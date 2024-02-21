"""
pretrain on IND Data
"""
import copy
import os

import torch
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from ind_model import BertForModel
from Utils.util import set_seed
from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm
import torch.nn.functional as F


class PretrainModelManager:
    def __init__(self, args):
        set_seed(args.seed)
        # build model
        self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="",
                                                  num_labels=args.n_ind_class)
        if args.freeze_bert_parameters:
            self.freeze_parameters()

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = self.get_optimizer(args)
        self.best_eval_score = 0

    def freeze_parameters(self):
        for name, param in self.model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = int(
            args.n_train_example / args.pretrain_batch_size) * args.pretrain_epoch
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr_pre,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        return optimizer

    def train(self, args, train_dataloader, val_dataloader):
        wait = 0
        best_model = None

        for epoch in trange(int(args.pretrain_epoch), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss, _ = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")
                    loss.backward()
                    tr_loss += loss.item()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss', loss)

            eval_score = self.eval(args, val_dataloader)
            print('eval_score', eval_score)

            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        self.save_model(args)

    def eval(self, args, val_dataloader):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.n_ind_class)).to(self.device)

        for batch in tqdm(val_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask, mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        return acc

    def test(self, args, test_dataloader):
        test_acc = self.eval(args, test_dataloader)
        print(test_acc)

    def save_model(self, args):
        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        if args.freeze_bert_parameters:
            model_name = 'pretrain_' + args.dataset + '_' + str(args.n_ind_class) + '_divide_seed_' + str(
                args.divide_seed) + '.bin'
        else:
            model_name = 'pretrain_' + args.dataset + '_' + str(args.n_ind_class) + '_divide_seed_' + str(
                args.divide_seed) + '_wo_freeze.bin'
        model_file = os.path.join(args.pretrain_dir, model_name)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())
