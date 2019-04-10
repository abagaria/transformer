import argparse
import torch
import pickle
from train import train
from hyperparameters import Hyperparameters
from tensorboardX import SummaryWriter
from evaluate import evaluate
from utils import *
from vocab import Vocab


def load_vocab():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("inverse_vocab.pkl", "rb") as f:
        reverse_vocab = pickle.load(f)
    return vocab, reverse_vocab


def create_vocab():
    Vocab(args.train_file, args.test_file)


def main():
    create_directory("saved_runs")

    device = torch.device(args.device)
    hy = Hyperparameters()
    writer = SummaryWriter() if args.tensor_log else None
    sentences = get_sentences(args.train_file)
    test_sentences = get_sentences(args.test_file)

    create_vocab()
    vocab, reverse_vocab = load_vocab()
    print("Loaded vocab of size {}".format(len(vocab)))

    train_perplexity = train(sentences, vocab, reverse_vocab, hy, writer, device)
    print("Training perplexity = {}".format(train_perplexity))

    evaluate(test_sentences, vocab, reverse_vocab, hy, writer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="training data", default="data/trainUnk.txt")
    parser.add_argument("--test_file", type=str, help="test data", default="data/testUnk.txt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tensor_log", type=bool, default=False)
    args = parser.parse_args()

    main()
