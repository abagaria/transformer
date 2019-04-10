# Python imports.
from tqdm import tqdm
import numpy as np
import pdb

# PyTorch imports.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Other imports.
from dataset import LMDataset
from model import LanguageModel


def evaluate(sentences, vocab, reverse_vocab, hy, writer, device):
    dataset = LMDataset(sentences, vocab, reverse_vocab, hy.window_size)
    loader = DataLoader(dataset, batch_size=hy.batch_size, shuffle=False, drop_last=True)
    vocab_size = len(vocab.keys())
    print("Loaded vocab of size {} for evaluation".format(vocab_size))

    model = LanguageModel(vocab_size, hy.embed_size, hy.window_size, hy.hidden_size, device, dataset)

    perplexities = []

    for epoch in range(1, hy.num_epochs + 1):
        model.load_state_dict(torch.load("saved_runs/transformer_{}_weights.pt".format(epoch)))
        perplexity = compute_model_accuracy(model, loader, device, epoch, writer)
        perplexities.append(perplexity)

    print("=" * 80)
    print("Evaluation metrics:")
    print("Final perplexity = {:.2f}".format(np.min(perplexities)))
    print("=" * 80)

    return perplexities


def compute_model_accuracy(model, loader, device, epoch, writer):
    loss_history = []
    n_iterations = 0

    # Using loss function to compute perplexity
    loss_function = nn.CrossEntropyLoss().to(device)

    model.eval()

    print("\rComputing validation accuracy model @ {} epoch..".format(epoch))

    for input_seq, label_seq in tqdm(loader):

        # Move the data to the GPU
        input_seq = input_seq.to(device)
        label_seq = label_seq.to(device)

        with torch.no_grad():
            logits = model(input_seq)
            loss = loss_function(logits, label_seq.squeeze(0))
            loss_history.append(loss.item())

        if writer is not None:
            writer.add_scalar("TestingLoss", loss.item(), n_iterations)
            n_iterations = n_iterations + 1

    perplexity = np.exp(np.mean(loss_history))

    return perplexity
