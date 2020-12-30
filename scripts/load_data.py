import csv
import torch


def load_ivectors(vec_file):
    with open(vec_file) as in_file:
        reader = csv.reader(in_file, delimiter=" ")
        ivectors = [[float(num) for num in ivec] for ivec in reader]

    return torch.FloatTensor(ivectors)


def load_labels(txt_file):
    with open(txt_file) as in_file:
        reader = csv.reader(in_file, delimiter="\t")
        return torch.FloatTensor([label for utterance, label in reader])
