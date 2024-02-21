import numpy as np
import torch
from tqdm import tqdm


class Memory(object):
    def __init__(self, args):
        self.input_ids = []
        self.input_mask = []
        self.segment_ids = []
        self.label_ids = []
        self.label_ids_gt = []
        self.n_exemplar_per_class = args.n_exemplar_per_class  # The number of exemplars stored for each class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = len(self.label_ids)

    def append(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids.append(input_ids)
        self.input_mask.append(input_mask)
        self.segment_ids.append(segment_ids)
        self.label_ids.append(label_ids)

    def get_random_batch(self, batch_size):
        permutations = np.random.permutation(self.size)
        index = permutations[:batch_size]
        mini_input_ids = [self.input_ids[i] for i in index]
        mini_input_mask = [self.input_mask[i] for i in index]
        mini_segment_ids = [self.segment_ids[i] for i in index]
        mini_label_ids = [self.label_ids[i] for i in index]
        return torch.tensor(mini_input_ids).to(self.device), torch.tensor(mini_input_mask).to(self.device), \
                torch.tensor(mini_segment_ids).to(self.device), torch.tensor(mini_label_ids).to(self.device)

    def select_ind_exemplars_to_store(self, ind_dataloader):
        input_ids_list, input_mask_list, segment_ids_list = [], [], []
        label_list = []

        for batch in tqdm(ind_dataloader, desc="select_ind_exemplars_to_store"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_list.append(label_ids)
        input_ids_list = torch.cat(input_ids_list, dim=0).data.cpu().numpy()
        input_mask_list = torch.cat(input_mask_list, dim=0).data.cpu().numpy()
        segment_ids_list = torch.cat(segment_ids_list, dim=0).data.cpu().numpy()
        label_list = torch.cat(label_list, dim=0).data.cpu().numpy()

        # For each class, randomly select a fixed number of samples
        for i in range(max(label_list) + 1):
            permutations = np.random.permutation(len(label_list[label_list == i]))
            index = permutations[:self.n_exemplar_per_class]
            for j in index:
                self.append(input_ids_list[label_list == i][j], input_mask_list[label_list == i][j],
                            segment_ids_list[label_list == i][j], label_list[label_list == i][j])
        self.size = len(self.label_ids)


    def select_ood_exemplars_to_store(self, p_dataloader, model=None, n_new_class=None):
        input_ids_list, input_mask_list, segment_ids_list, feats_list = [], [], [], []
        p_label_list, t_label_list = [], []
        select_p_label_list, select_t_label_list = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(p_dataloader, desc="select_ood_exemplars_to_store"):
                batch = tuple(t.to(self.device) for t in batch)
                # t_label_ids is only used for calculating metrics
                input_ids, input_mask, segment_ids, p_label_ids, t_label_ids = batch
                # forward
                feats = model(input_ids, input_mask, segment_ids, mode="analysis")
                # append
                input_ids_list.append(input_ids)
                input_mask_list.append(input_mask)
                segment_ids_list.append(segment_ids)
                p_label_list.append(p_label_ids)
                t_label_list.append(t_label_ids)
                feats_list.append(feats)
        # concat
        input_ids_list = torch.cat(input_ids_list, dim=0).data.cpu().numpy()
        input_mask_list = torch.cat(input_mask_list, dim=0).data.cpu().numpy()
        segment_ids_list = torch.cat(segment_ids_list, dim=0).data.cpu().numpy()
        feats_list = torch.cat(feats_list, dim=0).data.cpu().numpy()
        p_label_list = torch.cat(p_label_list, dim=0).data.cpu().numpy()
        t_label_list = torch.cat(t_label_list, dim=0).data.cpu().numpy()

        # For each class, randomly select a fixed number of samples
        for i in range(max(p_label_list) + 1):
            permutations = np.random.permutation(len(p_label_list[p_label_list == i]))
            index = permutations[:self.n_exemplar_per_class]
            for j in index:
                self.append(input_ids_list[p_label_list == i][j], input_mask_list[p_label_list == i][j],
                            segment_ids_list[p_label_list == i][j], p_label_list[p_label_list == i][j])
                select_p_label_list.append(p_label_list[p_label_list == i][j])
                select_t_label_list.append(t_label_list[p_label_list == i][j])

        self.size = len(self.label_ids)

    def print_memory_info(self):
        print('Memory size:{}'.format(self.size))

