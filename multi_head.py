# Python imports.
import pdb
import numpy as np

# PyTorch imports.
import torch
import torch.nn as nn
from single_head import SingleHeadedAttention


class MultiHeadedAttention(nn.Module):
    """ Multi Headed Attention Network. """

    def __init__(self, window_size, embedding_size, hidden_size, device, dataset):
        super(MultiHeadedAttention, self).__init__()
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.device = device
        self.dataset = dataset  # TODO

        self.sha1 = SingleHeadedAttention(window_size, embedding_size, hidden_size, device, dataset)
        self.sha2 = SingleHeadedAttention(window_size, embedding_size, hidden_size, device, dataset)
        self.sha3 = SingleHeadedAttention(window_size, embedding_size, hidden_size, device, dataset)
        self.lin1 = nn.Linear(3 * hidden_size, hidden_size)

    def forward(self, embedding_matrix):
        x1 = self.sha1(embedding_matrix)
        x2 = self.sha2(embedding_matrix)
        x3 = self.sha3(embedding_matrix)
        x4 = torch.cat((x1, x2, x3), dim=-1)
        x5 = self.lin1(x4)
        return x5
