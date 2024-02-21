import torch
import numpy as np
import os
import csv

from torch.utils.data import Dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from argparse import ArgumentParser
from transformers import BertTokenizer




class OriginSamples(Dataset):
    """A class that stores raw text and label data"""

    def __init__(self, train_x, train_y):
        assert len(train_y) == len(train_x)
        self.train_x = train_x
        self.train_y = train_y


class InputFeatures(object):
    """Storing a set of data features"""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# Convert samples into input features for BERT, label mapping, and store them in a list containing a series of InputFeatures classes
def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer):
    features = []
    content_list = examples.train_x
    label_list = examples.train_y

    for i in range(len(content_list)):
        # input_ids
        tokens_a = tokenizer.tokenize(content_list[i])  
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]  # Truncation
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens) 
        # segment_ids
        segment_ids = [0] * len(tokens)
        # input_mask
        input_mask = [1] * len(input_ids)
        # Padding
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # Label Mapping
        label_id = label_map[label_list[i]]
        # Instantiate into the InputFeatures class
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def get_datamodule(args):
    if args.dataset == "banking":
        return BankingDataModule(args)
    elif args.dataset == "clinc":
        return ClincDataModule(args)
    else:
        raise ValueError()


class BankingDataModule:
    def __init__(self, args):
        self.stage = None  # The current stage, 0 denotes the training stage of IND classes
        self.max_stage = args.max_stage
        self.data_dir = os.path.join('dataset', args.dataset)
        self.n_ind_class = args.n_ind_class  # Number of IND classes
        self.n_ood_classes = args.n_ood_classes  # Number of OOD classes
        self.bert_model = args.bert_model  # BERT backbone
        self.max_seq_length = args.max_seq_length  # The maximum token length of sentence
        self.all_label_list = self.get_labels(self.data_dir)  # obtain all classes labels
        self.divide_seed = args.divide_seed
        self.ind_class, self.ood_classes = self.divide_labels(self.divide_seed)  # ind ood 类别划分
        self.label_map = self.label_map()  # classes to ids mapping
        self.pretrain_batch_size = args.pretrain_batch_size
        self.train_batch_size = args.train_batch_size

    def label_map(self):
        label_map = {}
        count = 0
        for label in self.ind_class:
            label_map[label] = count
            count += 1
        for ood_class in self.ood_classes:
            for label in ood_class:
                label_map[label] = count
                count += 1
        return label_map

    # Divide classess into IND and multi-stage OOD classes based on Seed
    def divide_labels(self, divide_seed):
        np.random.seed(divide_seed)
        ind_class = list(
            np.random.choice(np.array(self.all_label_list), self.n_ind_class, replace=False))  # IND Classes List
        print("ind_class:\n", ind_class)
        ood_class_all = [item for item in self.all_label_list if item not in set(ind_class)]
        print("ood_class_all")
        print(ood_class_all)
        ood_class = []  # OOD Classes List
        for i in range(self.max_stage):
            ood_class.append(list(
                np.random.choice(np.array(ood_class_all), self.n_ood_classes[i], replace=False)))
            print("ood_class_stage_" + str(i + 1) + ":\n", ood_class[i])
            ood_class_all = [item for item in ood_class_all if item not in set(ood_class[i])]
        # check
        assert len(ind_class) == self.n_ind_class
        for i in range(self.max_stage):
            assert len(ood_class[i]) == self.n_ood_classes[i]

        return ind_class, ood_class

    # Return all class labels in the training set
    def get_labels(self, data_dir):
        import pandas as pd
        train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(train_df['label']))
        return labels

    # Return the number of training samples
    def get_n_train_example(self, stage):
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        train_set = self.get_datasets(train_data_dir)
        train_ind, train_ood = self.divide_datasets(train_set)
        if stage == 0:
            return len(train_ind)
        else:
            return len(train_ood[stage - 1])

    # Read data file and return nested list：[[x_0,y_0],[x_1,y_1],……]
    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                if i == 0:
                    i += 1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines

    # Divide the [[x_0, y_0],...] into a two-dimensional array of IND data and a three-dimensional array of all OOD data, and the first dimension represents the OOD stage
    def divide_datasets(self, origin_data):
        ind_examples, ood_examples = [], []
        for i in range(self.max_stage):
            ood_examples.append([])
        for example in origin_data:
            if example[-1] in self.ind_class:
                ind_examples.append(example)
            else:
                for i in range(self.max_stage):
                    if example[-1] in self.ood_classes[i]:
                        ood_examples[i].append(example)
        return ind_examples, ood_examples

    # Convert [[x_1, y_1], [x_1, y_1],...] to [x_1, x_1,...] and [y_1, y_1,...] and instantiate them into the OriginSamples class
    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list, labels_list)

        return data

    # Process and encapsulate raw data into TensorDataset
    def get_loader(self, labelled_examples, batch_size, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        # Convert raw data into input tensors
        features = convert_examples_to_features(labelled_examples, self.label_map, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader

    def prepare_data(self):
        # dataset path
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        val_data_dir = os.path.join(self.data_dir, "eval.tsv")
        test_data_dir = os.path.join(self.data_dir, "test.tsv")
        # read data
        train_set = self.get_datasets(train_data_dir)
        val_set = self.get_datasets(val_data_dir)
        test_set = self.get_datasets(test_data_dir)
        # split IND/OOD
        self.train_ind, self.train_ood = self.divide_datasets(train_set)
        self.val_ind, self.val_ood = self.divide_datasets(val_set)
        self.test_ind, self.test_ood = self.divide_datasets(test_set)
        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of ind/ood train samples: ", len(self.train_ind),
                [len(self.train_ood[i]) for i in range(self.max_stage)])
        print("the numbers of ind/ood validation samples: ", len(self.val_ind),
                [len(self.val_ood[i]) for i in range(self.max_stage)])
        print("the numbers of ind/ood test samples: ", len(self.test_ind),
                [len(self.test_ood[i]) for i in range(self.max_stage)])

        self.test_all_data = self.get_samples(test_set)

    def setup(self, stage=None):
        self.stage = stage
        if self.stage == 0:
            self.train_data = self.get_samples(self.train_ind)
            self.val_data = self.get_samples(self.val_ind)
            self.test_ind_data = self.get_samples(self.test_ind)
        else:
            self.train_data = self.get_samples(self.train_ood[self.stage - 1])
            self.val_data = self.get_samples(self.val_ood[self.stage - 1])
            self.test_ind_data = self.get_samples(self.test_ind)
            self.test_ood_data = [self.get_samples(self.test_ood[i]) for i in range(self.max_stage)]

    def train_dataloader(self):
        if self.stage == 0:
            return self.get_loader(self.train_data, batch_size=self.pretrain_batch_size, mode="train")
        else:
            return self.get_loader(self.train_data, batch_size=self.train_batch_size, mode="train")

    def val_dataloader(self):
        if self.stage == 0:
            return self.get_loader(self.val_data, batch_size=self.pretrain_batch_size, mode="validation")
        else:
            return self.get_loader(self.val_data, batch_size=self.train_batch_size, mode="validation")

    def test_dataloader(self):
        if self.stage == 0:
            return self.get_loader(self.test_ind_data, batch_size=self.pretrain_batch_size, mode="test")
        else:
            test_ood_list = []
            test_ind_loader = self.get_loader(self.test_ind_data, batch_size=self.train_batch_size, mode="test")
            for i in range(self.stage):
                test_ood_list.append(
                    self.get_loader(self.test_ood_data[i], batch_size=self.train_batch_size, mode="validation"))
            test_all_loader = self.get_loader(self.test_all_data, batch_size=self.train_batch_size, mode="test")
            return [test_ind_loader, test_ood_list, test_all_loader]


class ClincDataModule:
    def __init__(self, args):
        self.stage = None  
        self.max_stage = args.max_stage
        self.data_dir = os.path.join('dataset', args.dataset)
        self.n_ind_class = args.n_ind_class  
        self.n_ood_classes = args.n_ood_classes 
        self.bert_model = args.bert_model  
        self.max_seq_length = args.max_seq_length  
        self.all_label_list = self.get_labels(self.data_dir)
        self.divide_seed = args.divide_seed
        self.ind_class, self.ood_classes = self.divide_labels(self.divide_seed) 
        self.label_map = self.label_map() 
        self.pretrain_batch_size = args.pretrain_batch_size
        self.train_batch_size = args.train_batch_size

    @property
    def dataloader_mapping(self):
        return {0: "IND", 1: "OOD", 2: "ALL"}

    def label_map(self):
        label_map = {}
        count = 0
        for label in self.ind_class:
            label_map[label] = count
            count += 1
        for ood_class in self.ood_classes:
            for label in ood_class:
                label_map[label] = count
                count += 1
        return label_map

    def divide_labels(self, divide_seed):
        np.random.seed(divide_seed)
        ind_class = list(
            np.random.choice(np.array(self.all_label_list), self.n_ind_class, replace=False))
        print("ind_class:\n", ind_class)
        ood_class_all = [item for item in self.all_label_list if item not in set(ind_class)]
        print("ood_class_all")
        print(ood_class_all)
        ood_class = [] 
        for i in range(self.max_stage):
            ood_class.append(list(
                np.random.choice(np.array(ood_class_all), self.n_ood_classes[i], replace=False)))
            print("ood_class_stage_" + str(i + 1) + ":\n", ood_class[i])
            ood_class_all = [item for item in ood_class_all if item not in set(ood_class[i])]
        # check
        assert len(ind_class) == self.n_ind_class
        for i in range(self.max_stage):
            assert len(ood_class[i]) == self.n_ood_classes[i]

        return ind_class, ood_class

    def get_labels(self, data_dir):
        import pandas as pd
        train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(train_df['label']))
        return labels

    def get_n_train_example(self, stage):
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        train_set = self.get_datasets(train_data_dir)
        train_ind, train_ood = self.divide_datasets(train_set)
        if stage == 0:
            return len(train_ind)
        else:
            return len(train_ood[stage - 1])

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                if i == 0:
                    i += 1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines

    def divide_datasets(self, origin_data):
        ind_examples, ood_examples = [], []
        for i in range(self.max_stage):
            ood_examples.append([])
        for example in origin_data:
            if example[-1] in self.ind_class:
                ind_examples.append(example)
            else:
                for i in range(self.max_stage):
                    if example[-1] in self.ood_classes[i]:
                        ood_examples[i].append(example)
        return ind_examples, ood_examples


    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list, labels_list)

        return data


    def get_loader(self, labelled_examples, batch_size, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, self.label_map, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader

    def prepare_data(self):
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        val_data_dir = os.path.join(self.data_dir, "dev.tsv")
        test_data_dir = os.path.join(self.data_dir, "test.tsv")

        train_set = self.get_datasets(train_data_dir)
        val_set = self.get_datasets(val_data_dir)
        test_set = self.get_datasets(test_data_dir)

        self.train_ind, self.train_ood = self.divide_datasets(train_set)
        self.val_ind, self.val_ood = self.divide_datasets(val_set)
        self.test_ind, self.test_ood = self.divide_datasets(test_set)
        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of ind/ood train samples: ", len(self.train_ind),
                [len(self.train_ood[i]) for i in range(self.max_stage)])
        print("the numbers of ind/ood validation samples: ", len(self.val_ind),
                [len(self.val_ood[i]) for i in range(self.max_stage)])
        print("the numbers of ind/ood test samples: ", len(self.test_ind),
                [len(self.test_ood[i]) for i in range(self.max_stage)])

        self.test_all_data = self.get_samples(test_set)

    def setup(self, stage=None):
        self.stage = stage
        if self.stage == 0:
            self.train_data = self.get_samples(self.train_ind)
            self.val_data = self.get_samples(self.val_ind)
            self.test_ind_data = self.get_samples(self.test_ind)
        else:
            self.train_data = self.get_samples(self.train_ood[self.stage - 1])
            self.val_data = self.get_samples(self.val_ood[self.stage - 1])
            self.test_ind_data = self.get_samples(self.test_ind)
            self.test_ood_data = [self.get_samples(self.test_ood[i]) for i in range(self.max_stage)]

    def train_dataloader(self):
        if self.stage == 0:
            return self.get_loader(self.train_data, batch_size=self.pretrain_batch_size, mode="train")
        else:
            return self.get_loader(self.train_data, batch_size=self.train_batch_size, mode="train")

    def val_dataloader(self):
        if self.stage == 0:
            return self.get_loader(self.val_data, batch_size=self.pretrain_batch_size, mode="validation")
        else:
            return self.get_loader(self.val_data, batch_size=self.train_batch_size, mode="validation")

    def test_dataloader(self):
        if self.stage == 0:
            return self.get_loader(self.test_ind_data, batch_size=self.pretrain_batch_size, mode="test")
        else:
            test_ood_list = []
            test_ind_loader = self.get_loader(self.test_ind_data, batch_size=self.train_batch_size, mode="test")
            for i in range(self.stage):
                test_ood_list.append(
                    self.get_loader(self.test_ood_data[i], batch_size=self.train_batch_size, mode="validation"))
            test_all_loader = self.get_loader(self.test_all_data, batch_size=self.train_batch_size, mode="test")
            return [test_ind_loader, test_ood_list, test_all_loader]


