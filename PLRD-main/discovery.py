import copy
import os
import random
import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
from tqdm import tqdm, trange
from cgid_model import Prototypes,MultiHeadBERT
from Utils.contrast_loss import InstanceLoss
from Utils.eval import ClusterMetrics, ClassifyMetrics
from Utils.sinkhorn_knopp import SinkhornKnopp

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiscoveryModelManager:
    def __init__(self, pretrained_model_path, args):
        # Stage information
        self.n_ind_class = args.n_ind_class
        self.n_ood_classes = args.n_ood_classes
        self.stage = None
        self.n_new_class = None  # The number of new classes in the current stage
        self.n_old_class = None  # The number of known classes in the current stage
        self.n_all_class = None  # The total number of new and known classes in the current stage

        # initialize model
        self.model = MultiHeadBERT.from_pretrained(
            pretrained_model_name_or_path=args.bert_model,
            n_old_class=args.n_ind_class,
            n_head=args.n_head
        )

        self.n_head = args.n_head
        self.temperature = args.temperature
        self.n_view = args.n_view

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.load_pretrained_model(pretrained_model_path)
        if args.freeze_bert_parameters:
            self.freeze_parameters()
        self.best_head = 0

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=args.num_iters_sk, epsilon=args.epsilon_sk
        )

        self.cluster_metrics = ClusterMetrics()
        self.classify_metrics = ClassifyMetrics()

        self.train_batch_size = args.train_batch_size
        self.train_epoch = args.train_epoch

        self.test_results = None
        self.criterion_instance = InstanceLoss(args.instance_temperature, self.device).to(self.device)

    def update_manager(self, args):
        self.stage = args.stage
        # merge existing class headers
        if self.stage > 1:
            self.merge_classifier()
        # update class information
        self.n_new_class = self.n_ood_classes[args.stage - 1]
        self.n_old_class = self.n_ind_class + sum(self.n_ood_classes[:args.stage - 1])
        self.n_all_class = self.n_new_class + self.n_old_class
        # update optimizer
        self.n_train_opt_steps = self.train_epoch
        self.optimizer, self.scheduler = self.configure_optimizers(args)
        # Update old models
        old_model_state = copy.deepcopy(self.model.state_dict())
        self.old_model = MultiHeadBERT.from_pretrained(
            pretrained_model_name_or_path=args.bert_model,
            n_old_class=self.n_old_class,
            n_head=args.n_head
        )
        self.old_model.load_state_dict(old_model_state, strict=False)
        # self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.freeze_all(self.old_model)
        self.old_model.to(self.device)
        # add new class head
        self.model.add_new_head(self.n_new_class)
        # add or update projection layers
        if self.stage == 1:
            self.model.add_contrast_head(self.n_all_class)
        else:
            self.model.update_contrast_head(self.n_new_class)
        self.model = self.model.to(self.device)
        # initialize
        self.best_p_train_dataloader = None  # Dataloader containing pseudo labels
        self.best_head = 0

    # Freeze the parameters of the first 11 layers
    def freeze_parameters(self):
        for name, param in self.model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    # Freeze all parameters
    def freeze_all(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False

    # Load IND pre-trained model parameters
    def load_pretrained_model(self, pretrained_model_path):
        pretrained_dict = torch.load(pretrained_model_path)
        # Dealing with the issue of different classification header names for IND pretrian_model and OOD train_model
        pretrained_dict['head_old.prototypes.weight'] = copy.copy(pretrained_dict['classifier.weight'])
        pretrained_dict['head_old.prototypes.bias'] = copy.copy(pretrained_dict['classifier.bias'])
        del pretrained_dict['classifier.weight']
        del pretrained_dict['classifier.bias']

        self.model.load_state_dict(pretrained_dict)

    # set optimizer and scheduler
    def configure_optimizers(self, args):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr_train,
            momentum=args.momentum_opt,
            weight_decay=args.weight_decay_opt,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_proportion * self.n_train_opt_steps),
            num_training_steps=self.n_train_opt_steps,
            num_cycles=0.5
        )
        return optimizer, scheduler

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.temperature, dim=-1)
        return -torch.mean(torch.sum(targets * preds, dim=-1))

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.n_view):
            for other_view in np.delete(range(self.n_view), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.n_view * (self.n_view - 1))

    def train(self, args, train_dataloader, memory=None, val_dataloader=None, test_dataloader_list=None):
        self.loss_per_head = torch.zeros(self.n_head)  # the loss of each head [n_head]

        for epoch in trange(int(self.train_epoch), desc="Epoch"):
            tr_loss, tr_loss_ce, tr_loss_proto, tr_loss_instance, tr_loss_kd = 0, 0, 0, 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()
            for batch in tqdm(train_dataloader, desc="Training"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # Sample known class data from Memory and merge it with new class data batch
                if args.use_memory:
                    # 　Memory random sampling
                    input_ids_replay, input_mask_replay, segment_ids_replay, label_ids_replay = memory.get_random_batch(
                        args.train_batch_size)
                    # merge
                    input_ids = torch.cat([input_ids, input_ids_replay], dim=0)
                    input_mask = torch.cat([input_mask, input_mask_replay], dim=0)
                    segment_ids = torch.cat([segment_ids, segment_ids_replay], dim=0)
                    label_ids = torch.cat([label_ids, label_ids_replay], dim=0)

                input_ids, input_mask, segment_ids, label_ids, mask_old = self.unpack_batch(input_ids, input_mask,
                                                                                            segment_ids, label_ids)

                nlc = self.n_old_class

                # normalize prototypes
                self.model.normalize_prototypes()

                # forward
                outputs, outputs_contrast = self.model(input_ids, input_mask, segment_ids, mode="discovery")

                # Classifier branch
                outputs["logits_old"] = (outputs["logits_old"].unsqueeze(1).expand(-1, self.n_head, -1, -1))
                logits = torch.cat([outputs["logits_old"], outputs["logits_new"]],
                                   dim=-1)
                # Building targets that classifier for prototype layer
                targets = torch.zeros_like(logits)
                targets_old = (F.one_hot(label_ids[mask_old], num_classes=self.n_old_class).float().to(
                    self.device))
                # gather outputs
                for v in range(self.n_view):
                    for h in range(self.n_head):
                        targets[v, h, mask_old, :nlc] = targets_old.type_as(targets)
                        # Generate soft pseudo labels using sinkhorn_knop
                        targets[v, h, ~mask_old, nlc:] = self.sk(outputs["logits_new"][v, h, ~mask_old]).type_as(
                            targets)

                # Prototype layer branch
                z_i, z_j = outputs_contrast[0]["instance_features"], outputs_contrast[1]["instance_features"]
                prototypes = self.model.prototypes.clone().detach()
                # Calculate the distance between the representation and the prototype
                logits_prot_view1 = torch.mm(z_i, prototypes.t())
                logits_prot_view2 = torch.mm(z_j, prototypes.t())
                # Pseudo labels provided by class prototypes
                proto_label_view1 = logits_prot_view1[~mask_old, nlc:].argmax(dim=1) 
                proto_label_view2 = logits_prot_view2[~mask_old, nlc:].argmax(dim=1)
                proto_targets_new_view1 = (F.one_hot(proto_label_view1, num_classes=self.n_new_class).float()
                                            .to(self.device))
                proto_targets_new_view2 = (F.one_hot(proto_label_view2, num_classes=self.n_new_class).float()
                                            .to(self.device))

                # Build pseudo labels that prototypes provided to classifiers
                proto_targets = torch.zeros_like(logits)
                for v in range(self.n_view):
                    for h in range(self.n_head):
                        proto_targets[v, h, mask_old, :nlc] = targets_old.type_as(proto_targets)
                        if v == 0:
                            proto_targets[v, h, ~mask_old, nlc:] = proto_targets_new_view1.type_as(proto_targets)
                        else:
                            proto_targets[v, h, ~mask_old, nlc:] = proto_targets_new_view2.type_as(proto_targets)

                # Momentum update prototype
                target_label_ids = targets.argmax(dim=-1)  # [n_view,n_head,batch_size]
                for feat, label in zip(z_i, target_label_ids[0, 0, :]):
                    self.model.prototypes[label] = self.model.prototypes[label] * args.proto_m + (
                            1 - args.proto_m) * feat

                for feat, label in zip(z_j, target_label_ids[1, 0, :]):
                    self.model.prototypes[label] = self.model.prototypes[label] * args.proto_m + (
                            1 - args.proto_m) * feat

                self.model.prototypes = F.normalize(self.model.prototypes, dim=1)
                # classification loss(cross-entropy)
                loss_ce = self.swapped_prediction(logits, proto_targets)
                # prototypical contrastive learning loss
                loss_proto_1 = self.cross_entropy_loss(logits_prot_view1, targets[1, 0, :, :])
                loss_proto_2 = self.cross_entropy_loss(logits_prot_view2, targets[0, 0, :, :])
                loss_proto = (loss_proto_1 + loss_proto_2) / 2
                # instance-level contrastive loss
                loss_instance = self.criterion_instance(z_i, z_j)
                # distillation loss
                if args.use_memory:
                    loss_kd = self.distill_old_model(input_ids=input_ids_replay,
                                                    input_mask=input_mask_replay,
                                                    segment_ids=segment_ids_replay,
                                                    feature_new=outputs['feats'][:, mask_old, :],
                                                    )

                if args.use_memory:
                    loss = (loss_ce + loss_proto + loss_instance + loss_kd) / 4
                else:
                    loss = (loss_ce + loss_proto + loss_instance)/3
                loss.backward()

                tr_loss += loss.item()
                tr_loss_ce += loss_ce.item()
                tr_loss_proto += loss_proto.item()
                if args.use_memory:
                    tr_loss_kd += loss_kd.item()
                tr_loss_instance += loss_instance.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # update best head tracker
                self.loss_per_head += loss_ce.cpu().clone().detach()

                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()

            self.best_head = torch.argmin(self.loss_per_head)
            print(self.best_head)
            # epoch avg loss
            tr_loss = tr_loss / nb_tr_steps
            tr_loss_ce = tr_loss_ce / nb_tr_steps
            tr_loss_proto = tr_loss_proto / nb_tr_steps
            tr_loss_instance = tr_loss_instance / nb_tr_steps
            tr_loss_kd = tr_loss_kd / nb_tr_steps
            # wandb  logging
            wandb.log({'train/loss_ce': tr_loss_ce,
                       'train/loss_proto': tr_loss_proto,
                       'train/loss_instance': tr_loss_instance,
                       'train/loss_kd': tr_loss_kd,
                       'train/loss': tr_loss,
                       'train/lr': self.optimizer.param_groups[0]['lr']
                       })

        self.get_pseudo_dataloader(train_dataloader)

    def validation(self, val_dataloader):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="validation"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # forward
                outputs = self.model(input_ids, input_mask, segment_ids, mode="eval")
                logits = torch.cat(
                    [outputs["logits_old"], outputs["logits_new"][self.best_head]],
                    dim=-1)  
                preds = logits.argmax(dim=-1)  
                self.cluster_metrics.update(preds, label_ids)
            val_results = self.cluster_metrics.compute()

            # wandb  logging
            wandb.log({'val/acc': val_results['acc'],
                       'val/nmi': val_results['acc'],
                       'val/ari': val_results['ari'],
                       })

    def unpack_batch(self, input_ids, input_mask, segment_ids, label_ids):
        mask_old = label_ids < self.n_old_class
        return input_ids, input_mask, segment_ids, label_ids, mask_old

    def test(self, test_dataloader_list):
        n_preds_right = 0  # Number of test samples
        n_preds = 0  # The correct number in the test samples
        self.model.eval()
        # IND test
        ind_dataloader = test_dataloader_list[0]
        for batch in tqdm(ind_dataloader, desc="Ind Test"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # forward
            with torch.no_grad():
                outputs = self.model(input_ids, input_mask, segment_ids, mode="eval")
            logits = torch.cat(
                [outputs["logits_old"], outputs["logits_new"][self.best_head]],
                dim=-1) 
            preds = logits.argmax(dim=-1)
            self.classify_metrics.update(preds, label_ids)
        test_ind_results = self.classify_metrics.compute()

        n_preds_right += test_ind_results['n_preds_right']
        n_preds += test_ind_results['n_preds']

        # OOD test
        ood_dataloaders = test_dataloader_list[1]
        test_ood_results_list = []
        n_ood_preds_right = 0
        n_ood_preds = 0
        for i in range(self.stage):
            dataloader = ood_dataloaders[i]
            for batch in tqdm(dataloader, desc="Ood Test"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # forward
                with torch.no_grad():
                    outputs = self.model(input_ids, input_mask, segment_ids, mode="eval")
                logits = torch.cat(
                    [outputs["logits_old"], outputs["logits_new"][self.best_head]],
                    dim=-1) 
                preds = logits.argmax(dim=-1) 
                self.cluster_metrics.update(preds, label_ids)
            test_ood_results = self.cluster_metrics.compute()
            test_ood_results_list.append(test_ood_results)

            n_ood_preds_right += test_ood_results['n_preds_right']
            n_ood_preds += test_ood_results['n_preds']

        # Calculate average accuracy
        n_preds_right += n_ood_preds_right
        n_preds += n_ood_preds
        test_ood_avg_acc = round(n_ood_preds_right / n_ood_preds * 100, 2)
        test_avg_acc = round(n_preds_right / n_preds * 100, 2)

        # wandb  logging
        wandb.log({'test_ind/ind_acc': test_ind_results['acc'],
                   'test_ood_1/ood_acc_1': test_ood_results_list[0]['acc'],
                   'test_ood_2/ood_acc_2': test_ood_results_list[1]['acc'] if self.stage > 1 else 0,
                   'test_ood_3/ood_acc_3': test_ood_results_list[2]['acc'] if self.stage > 2 else 0,
                   'test_avg/ood_avg_acc': test_ood_avg_acc,
                   'test_avg/avg_acc': test_avg_acc,
                   })
        print("stage:", self.stage)
        print("test_ind_results:", test_ind_results)
        print("test_ood_acc:", test_ood_results_list)
        print("test_avg_acc:", test_avg_acc)
        print("test_ood_avg_acc:", test_ood_avg_acc)
        self.test_results = [test_ind_results, test_ood_results_list, test_ood_avg_acc, test_avg_acc]

    # Merge the train_dataloader data with the data in memory to return a new train_dataloader
    def merge_dataloader_memory(self, train_dataloader, memory):
        total_label_ids, total_input_ids, total_input_mask, total_segment_ids = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc="Extracting representation"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                total_input_ids.append(input_ids)
                total_input_mask.append(input_mask)
                total_segment_ids.append(segment_ids)
                total_label_ids.append(label_ids)

            total_input_ids.append(torch.tensor(memory.input_ids).to(self.device))
            total_input_mask.append(torch.tensor(memory.input_mask).to(self.device))
            total_segment_ids.append(torch.tensor(memory.segment_ids).to(self.device))
            total_label_ids.append(torch.tensor(memory.label_ids).to(self.device))

            total_input_ids = torch.cat(total_input_ids, dim=0)
            total_input_mask = torch.cat(total_input_mask, dim=0)
            total_segment_ids = torch.cat(total_segment_ids, dim=0)
            total_label_ids = torch.cat(total_label_ids, dim=0)

            train_data = TensorDataset(total_input_ids, total_input_mask, total_segment_ids, total_label_ids)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)
            return train_dataloader

    # After each stage of training, merge the old and new classifiers into a joint classifier
    def merge_classifier(self):
        # save the weights and bias of the last stage classifier
        old_head_weight = self.model.head_old.prototypes.weight.data.clone()
        old_head_bias = self.model.head_old.prototypes.bias.data.clone()
        new_head_weight = self.model.head_new.prototypes[self.best_head].prototypes.weight.data.clone()
        new_head_bias = self.model.head_new.prototypes[self.best_head].prototypes.bias.data.clone()
        # create the joint classifier
        self.model.head_old = Prototypes(self.model.hidden_size, self.n_all_class).to(self.device)
        # put the weight
        self.model.head_old.prototypes.weight.data[:self.n_old_class] = old_head_weight
        self.model.head_old.prototypes.weight.data[self.n_old_class:] = new_head_weight
        # put the bias
        self.model.head_old.prototypes.bias.data[:self.n_old_class] = old_head_bias
        self.model.head_old.prototypes.bias.data[self.n_old_class:] = new_head_bias

    # Construct a new dataloader with pseudo labels using the input of train_dataloader and the pseudo labels of the model output
    def get_pseudo_dataloader(self, train_dataloader):
        total_input_ids, total_input_mask, total_segment_ids = [], [], []
        total_p_label, total_t_label = [], []  # t_lable is only used for calculating metrics
        self.model.eval()
        for batch in tqdm(train_dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            # label_ids is only used to calculate the acc of pseudo labels and does not involve model training and selection
            input_ids, input_mask, segment_ids, label_ids = batch
            # forward
            with torch.no_grad():
                outputs = self.model(input_ids, input_mask, segment_ids, mode="eval")
            p_label = outputs["logits_new"][self.best_head].argmax(dim=-1) + self.n_old_class
            # append
            total_input_ids.append(input_ids)
            total_input_mask.append(input_mask)
            total_segment_ids.append(segment_ids)
            total_p_label.append(p_label)
            total_t_label.append(label_ids)
            self.cluster_metrics.update(p_label, label_ids)
        # concat
        total_input_ids = torch.cat(total_input_ids, dim=0)
        total_input_mask = torch.cat(total_input_mask, dim=0)
        total_segment_ids = torch.cat(total_segment_ids, dim=0)
        total_p_label = torch.cat(total_p_label, dim=0)
        total_t_label = torch.cat(total_t_label, dim=0)

        # compute pseudo_labels metrics
        train_pseudo_results = self.cluster_metrics.compute()
        wandb.log({'train/pseudo_label_acc': train_pseudo_results['acc']})
        # new dataloader with pseudo_labels
        p_train_data = TensorDataset(total_input_ids, total_input_mask, total_segment_ids, total_p_label,
                                     total_t_label)
        p_train_sampler = SequentialSampler(p_train_data)
        p_train_dataloader = DataLoader(p_train_data, sampler=p_train_sampler, batch_size=self.train_batch_size)

        self.best_p_train_dataloader = p_train_dataloader

    def save_results(self, args, memory_size):
        test_ind_results, test_ood_results_list, test_ood_avg_acc, test_avg_acc = self.test_results
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)
        ood_acc = []
        for i in range(args.max_stage):
            ood_acc.append(test_ood_results_list[i]['acc'] if self.stage > i else 0)
        var = [self.stage, test_ind_results['acc'], ood_acc[0], ood_acc[1], ood_acc[2], test_ood_avg_acc, test_avg_acc,
               args.n_ind_class, str(args.n_ood_classes),
               args.use_memory, args.n_exemplar_per_class, 
               memory_size,
               args.train_epoch, args.train_batch_size, args.lr_train, args.proto_m,
               args.seed, args.divide_seed
               ]
        names = ['stage', 'ind_acc', 'ood_acc_1', 'ood_acc_2', 'ood_acc_3', 'acc_ood_avg', 'avg_acc',
                 'n_ind_class', 'n_ood_classes',
                 'use_memory', 'n_exemplar_per_class',
                 'memory_size', 
                 'epoch', 'batch_size', 'learing_rate', 'proto_m',
                 'seed', 'divide_seed']
        vars_dict = {k: v for k, v in zip(names, var)}
        keys = list(vars_dict.keys())
        values = list(vars_dict.values())

        file_name = args.results_file_name + '.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(vars_dict, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)

    def distill_old_model(self, input_ids, input_mask, segment_ids, feature_new):
        with torch.no_grad():
            feature_old = self.old_model(input_ids, input_mask, segment_ids,
                                         mode="analysis")
        # Calculate the distillation loss (distance loss) of the output features of both old and new models
        n_sample, feat_dim = feature_old.size()  
        loss_kd = torch.dist(F.normalize(feature_old.view(n_sample * feat_dim, 1), dim=0),
                             F.normalize(feature_new[0].view(n_sample * feat_dim, 1), dim=0))
        return loss_kd