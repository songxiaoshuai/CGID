import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()
        self.prototypes = nn.Linear(output_dim, num_prototypes)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MultiHead(nn.Module):
    def __init__(
            self, input_dim, num_prototypes, n_head
    ):
        super().__init__()
        self.n_head = n_head

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(input_dim, num_prototypes) for _ in range(n_head)]
        )

        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = F.normalize(feats, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.n_head)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadBERT(BertPreTrainedModel):
    def __init__(self, config, n_old_class, n_head=4):
        super(MultiHeadBERT, self).__init__(config)

        # backbone
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        self.head_old = Prototypes(config.hidden_size, n_old_class)
        self.n_head = n_head
        self.config = config
        self.apply(self.init_bert_weights)
        self.proto_dim = 128

    def add_new_head(self, n_new_class):
        if self.n_head is not None:
            self.head_new = MultiHead(
                input_dim=self.config.hidden_size,
                num_prototypes=n_new_class,
                n_head=self.n_head
            )

    def add_contrast_head(self, n_class):
        self.instance_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 128)
        )
        self.register_buffer("prototypes", torch.zeros(n_class, self.proto_dim))

    # Expand the class prototype layer based on the number of new classes
    def update_contrast_head(self, n_new_class):
        old_prototypes = self.prototypes.cpu().clone().detach()
        new_prototypes = torch.zeros(n_new_class, self.proto_dim)
        self.prototypes = torch.cat((old_prototypes, new_prototypes), 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_old.normalize_prototypes()
        if getattr(self, "head_new", False):
            self.head_new.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"logits_old": self.head_old(F.normalize(feats))}
        if hasattr(self, "head_new"):
            logits_new, proj_feats_new = self.head_new(feats)
            out.update(
                {
                    "logits_new": logits_new,
                    "proj_feats_new": proj_feats_new,
                }
            )
        return out

    def forward_contrast(self, feats):
        out = {"instance_features": F.normalize(self.instance_projector(feats), dim=1)}

        return out

    def forward(self, input_ids, input_mask, segment_ids, mode="pretrain"):
        if mode == "pretrain":
            encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)
            pooled_output_01 = self.dense(encoded_layer_12_emb01.mean(dim=1))
            pooled_output_01 = self.activation(pooled_output_01)
            pooled_output_01 = self.dropout(pooled_output_01)
            logits = self.forward_heads(pooled_output_01)['logits_old']
            out_contrast = self.forward_contrast(pooled_output_01)['instance_features']
            out_dict_01 = {"feats": pooled_output_01, 'instance_features': out_contrast,'logits':logits}
            return out_dict_01
        
        elif mode == "discovery":
            encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)
            encoded_layer_12_emb02, pooled_output_02 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)

            pooled_output_01 = self.dense(encoded_layer_12_emb01.mean(dim=1))
            pooled_output_02 = self.dense(encoded_layer_12_emb02.mean(dim=1))

            pooled_output_01 = self.activation(pooled_output_01)
            pooled_output_02 = self.activation(pooled_output_02)

            pooled_output_01 = self.dropout(pooled_output_01)
            pooled_output_02 = self.dropout(pooled_output_02)

            feats = [pooled_output_01, pooled_output_02]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])

            out_contrast = [self.forward_contrast(f) for f in feats]
            return out_dict, out_contrast

        elif mode == "eval":
            encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)
            pooled_output_01 = self.dense(encoded_layer_12_emb01.mean(dim=1))
            pooled_output_01 = self.activation(pooled_output_01)
            pooled_output_01 = self.dropout(pooled_output_01)
            out_01 = self.forward_heads(pooled_output_01)
            out_dict_01 = {"feats": pooled_output_01}
            for key in out_01.keys():
                out_dict_01[key] = out_01[key]
            return out_dict_01


        elif mode == "analysis":
            encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)
            pooled_output_01 = self.dense(encoded_layer_12_emb01.mean(dim=1))
            pooled_output_01 = self.activation(pooled_output_01)
            pooled_output_01 = self.dropout(pooled_output_01)

            return pooled_output_01
