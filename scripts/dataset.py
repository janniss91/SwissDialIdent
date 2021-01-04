import csv
import torch

from torch.utils.data import Dataset


class SwissDialectDataset(Dataset):
    def __init__(self, vec_file, txt_file):
        self.dialect_map = {"LU": 0, "BE": 1, "ZH": 2, "BS": 3}

        self.ivectors = self.load_ivectors(vec_file)
        self.labels = self.load_labels(txt_file)
        self.n_samples = self.ivectors.shape[0]
        self.n_features = self.ivectors.shape[1]
        self.n_classes = len(self.dialect_map)

    def load_ivectors(self, vec_file):
        with open(vec_file) as in_file:
            reader = csv.reader(in_file, delimiter=" ")
            ivectors = [[float(num) for num in ivec] for ivec in reader]

        return torch.FloatTensor(ivectors)

    def load_labels(self, txt_file):
        with open(txt_file) as in_file:
            reader = csv.reader(in_file, delimiter="\t")
            numeric_labels = [self.dialect_map[label] for _utterance, label in reader]

        return torch.LongTensor(numeric_labels)

    def __getitem__(self, index):
        return self.ivectors[index], self.labels[index]

    def __len__(self):
        return self.n_samples
