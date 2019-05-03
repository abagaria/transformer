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


def evaluate(model, sentences, vocab, reverse_vocab, hy, writer, device):
    dataset = LMDataset(sentences, vocab, reverse_vocab, hy.window_size)
    loader = DataLoader(dataset, batch_size=hy.batch_size, shuffle=True, drop_last=True)
    vocab_size = len(vocab.keys())
    print("Loaded vocab of size {} for evaluation".format(vocab_size))

    perplexity = compute_model_accuracy(model, loader, device, writer)

    return perplexity


def compute_model_accuracy(model, loader, device, writer):
    loss_history = []
    n_iterations = 0

    # Using loss function to compute perplexity
    loss_function = nn.CrossEntropyLoss().to(device)

    model.eval()

    for input_seq, label_seq, mask_idx in tqdm(loader):

        # Move the data to the GPU
        input_seq = input_seq.to(device)
        label_seq = label_seq.to(device)
        mask_idx = mask_idx.to(device).squeeze(0)

        with torch.no_grad():
            logits, _ = model(input_seq)
            loss = loss_function(logits[mask_idx, :], label_seq.squeeze(0)[mask_idx])
            loss_history.append(loss.item())

        if writer is not None:
            writer.add_scalar("TestingLoss", loss.item(), n_iterations)
            n_iterations = n_iterations + 1

    perplexity = np.exp(np.mean(loss_history))

    return perplexity
