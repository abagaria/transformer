import argparse
import torch
from train import train
from hyperparameters import Hyperparameters
from tensorboardX import SummaryWriter
from evaluate import evaluate
from utils import *
from vocab import Vocab


def create_vocab():
    vocab_object = Vocab(args.train_file, args.test_file)
    vocab, reverse_vocab = vocab_object.vocab, vocab_object.reverse_vocab
    return vocab, reverse_vocab


def main():
    create_directory("saved_runs")

    device = torch.device(args.device)
    hy = Hyperparameters()
    writer = SummaryWriter() if args.tensor_log else None
    sentences = get_sentences(args.train_file)
    test_sentences = get_sentences(args.test_file)

    vocab, reverse_vocab = create_vocab()
    print("Loaded vocab of size {}".format(len(vocab)))

    model, train_perplexity = train(sentences, vocab, reverse_vocab, hy, writer, device)
    print("Training perplexity = {}".format(train_perplexity))

    evaluate(model, test_sentences, vocab, reverse_vocab, hy, writer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="training data", default="data/trainUnk.txt")
    parser.add_argument("--test_file", type=str, help="test data", default="data/testUnk.txt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tensor_log", type=bool, default=False)
    args = parser.parse_args()

    main()
