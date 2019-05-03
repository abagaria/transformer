# Python imports.
import pdb
from collections import defaultdict
import pickle
import numpy as np
from copy import deepcopy

# PyTorch imports.
import torch
from torch.utils.data import Dataset

# Other imports.
from utils import *


class LMDataset(Dataset):
    """ Dataset module for our language model. """

    def __init__(self, sentences, vocab, reverse_vocab, window_length):
        """
        Args:
            sentences (list): list of sentences in training / dev data
            vocab (defaultdict)
            reverse_vocab (defaultdict)
            window_length (int)
        """
        np.random.seed(0)
        torch.manual_seed(0)

        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.sentences = sentences
        self.window_length = window_length
        self.training_data = self._create_string_windows()

    def _create_string_windows(self):
        """
        Convert the training data (list of sentences) into 1 string and then create
        chunks / windows of certain length.
        Returns:
            windows (list): list of lists of symbols in the current data window
        """
        all_words = " ".join(self.sentences).split()
        num_windows = len(all_words) // self.window_length
        windows = []
        for i in range(num_windows):
            start_idx = self.window_length * i
            end_idx = start_idx + self.window_length
            window = all_words[start_idx:end_idx]
            windows.append(window)
        return windows

    def __len__(self):
        assert isinstance(self.training_data[0], list), "Expected data as LoL, got {}".format(self.training_data)
        return len(self.training_data)

    def __getitem__(self, i):
        sub_string = self.training_data[i]

        input_sub_string = deepcopy(sub_string)
        label_sub_string = deepcopy(sub_string)

        # Convert strings to list of word ids.
        input_sequence = [self.vocab[word] for word in input_sub_string]
        label_sequence = [self.vocab[word] for word in label_sub_string]

        random_seq1 = np.random.uniform(0., 1., size=len(input_sequence))
        chosen_idx = [idx for idx, num in enumerate(random_seq1) if num < 0.15]
        random_seq2 = np.random.uniform(0., 1., size=len(chosen_idx))
        mask_idx = [chosen_idx[idx] for idx, num in enumerate(random_seq2) if num < 0.8]
        replace_idx = [chosen_idx[idx] for idx, num in enumerate(random_seq2) if 0.8 < num < 0.9]

        for mask_id in mask_idx:
            input_sequence[mask_id] = self.vocab["<MASK>"]

        for replace_id in replace_idx:
            input_sequence[replace_id] = np.random.randint(0, len(self.vocab))

        # Eventually we want to return tensors from the dataset class.
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        label_tensor = torch.tensor(label_sequence, dtype=torch.long)
        chosen_tensor = torch.tensor(chosen_idx, dtype=torch.long)

        return input_tensor, label_tensor, chosen_tensor

    def decode_line(self, word_ids):
        sentence = []
        for word_id in word_ids.tolist():
            sentence.append(self.reverse_vocab[word_id])
        return sentence


if __name__ == "__main__":
    data_file = "data/testUnk.txt"
    all_sentences = get_sentences(data_file)
    with open("vocab.pkl", "rb") as f:
        v = pickle.load(f)
    with open("inverse_vocab.pkl", "rb") as f:
        rv = pickle.load(f)
    d_set = LMDataset(all_sentences, v, rv, 20)
