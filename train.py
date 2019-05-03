# Python imports.
import pdb
from tqdm import tqdm
import numpy as np

# PyTorch imports.
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# Other imports.
from dataset import LMDataset
from model import LanguageModel


def train(sentences, vocab, reverse_vocab, hy, writer, device):
    dataset = LMDataset(sentences, vocab, reverse_vocab, hy.window_size)
    loader = DataLoader(dataset, batch_size=hy.batch_size, shuffle=True, drop_last=True)

    vocab_size = len(vocab.keys())
    model = LanguageModel(vocab_size, hy.embed_size, hy.window_size, hy.hidden_size, device, dataset)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hy.lr)

    n_iterations = 0
    loss_history = []

    model.train()

    for input_seq, label_seq, mask_idx in tqdm(loader):

        # Move the data to the GPU
        input_seq = input_seq.to(device)
        label_seq = label_seq.to(device)
        mask_idx = mask_idx.to(device).squeeze(0)

        optimizer.zero_grad()
        logits, _ = model(input_seq)
        loss = loss_function(logits[mask_idx, :], label_seq.squeeze(0)[mask_idx])
        loss.backward()
        optimizer.step()

        if writer is not None:
            writer.add_scalar("TrainingLoss", loss.item(), n_iterations)

        n_iterations = n_iterations + 1
        loss_history.append(loss.item())

    perplexity = np.exp(np.mean(loss_history))
    return model, perplexity
