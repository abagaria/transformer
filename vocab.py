# Python imports.
import pdb
from collections import defaultdict
import pickle


class Vocab(object):

    def __init__(self, training_file, testing_file):
        self.input_file = training_file
        self.output_file = testing_file
        self.training_data = self._get_training_data()
        self.vocab, self.reverse_vocab = self._create_vocab()
        self.save_vocab()

    def save_vocab(self):
        with open("vocab.pkl", "wb+") as f:
            pickle.dump(self.vocab, f)
        with open("inverse_vocab.pkl", "wb+") as f:
            pickle.dump(self.reverse_vocab, f)

    def _get_training_data(self):
        train_data = []
        with open(self.input_file) as _file:
            for line in _file:
                train_data.append(line)
        with open(self.output_file) as _file:
            for line in _file:
                train_data.append(line)
        return train_data

    def get_vocab_size(self):
        return len(self.vocab.keys())

    def _create_vocab(self):
        all_words = []
        for line in self.training_data:
            words = line.split()
            all_words += words
        word_set = set(all_words)
        vocab = defaultdict()
        reverse_vocab = defaultdict()
        for idx, word in enumerate(word_set):
            vocab[word] = idx
            reverse_vocab[idx] = word
        return vocab, reverse_vocab


if __name__ == "__main__":
    _train_file = "data/traintUnk.txt"
    _test_file = "data/testUnk.txt"
    v = Vocab(_train_file, _test_file)
