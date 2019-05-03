import argparse
import torch
from train import train
from hyperparameters import Hyperparameters
from evaluate import evaluate
from utils import *
from vocab import Vocab
import pickle


def create_vocab():
    vocab_object = Vocab(args.train_file, args.test_file)
    vocab, reverse_vocab = vocab_object.vocab, vocab_object.reverse_vocab
    return vocab, reverse_vocab


def main():
    create_directory("saved_runs")

    device = torch.device(args.device)
    hy = Hyperparameters()
    writer = None # SummaryWriter() if args.tensor_log else None
    sentences = get_sentences(args.train_file)
    test_sentences = get_sentences(args.test_file)

    vocab, reverse_vocab = create_vocab()
    print("Loaded vocab of size {}".format(len(vocab)))

    with open("vocab.pkl", "wb+") as f:
        pickle.dump(vocab, f)
    with open("reverse_vocab.pkl", "wb+") as f:
        pickle.dump(reverse_vocab, f)

    train_perplexities = []
    test_perplexities = []

    for epoch in range(hy.num_epochs):
        model, train_perplexity = train(sentences, vocab, reverse_vocab, hy, writer, device)
        test_perplexity = evaluate(model, test_sentences, vocab, reverse_vocab, hy, writer, device)
        train_perplexities.append(train_perplexity)
        test_perplexities.append(test_perplexity)
        torch.save(model.state_dict(), "saved_runs/bert_transformer_{}.pt".format(epoch))

    print("=" * 80)
    print("Final Train Perplexity = {:.2f}".format(min(train_perplexities)))
    print("Final Test Perplexity = {:.2f}".format(min(test_perplexities)))
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="training data", default="data/trainUnk.txt")
    parser.add_argument("--test_file", type=str, help="test data", default="data/testUnk.txt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tensor_log", type=bool, default=False)
    args = parser.parse_args()

    main()
