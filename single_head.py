# Python imports.
import pdb
import numpy as np

# PyTorch imports.
import torch
import torch.nn as nn


class SingleHeadedAttention(nn.Module):
    """ Single Headed Attention Network. """

    def __init__(self, window_size, embedding_size, hidden_size, device, dataset):
        super(SingleHeadedAttention, self).__init__()
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.device = device
        self.dataset = dataset  # TODO

        # Collin's equation: softmax( QAQ' + mask ) QB
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, hidden_size)

        self.softmax = nn.Softmax(dim=1)
        self.to(device)

    def forward(self, embedding_matrix):
        """
        :param embedding_matrix: Q in the Collin's equation (window_size x embed_size)
        :return:
        """
        # Need to remove the batch dimension because of the way we are doing A and B
        embedding_matrix = embedding_matrix.squeeze(0)

        x = self.fc1(embedding_matrix) @ embedding_matrix.t()
        mask = torch.tril(x)
        mask = mask.masked_fill(mask == 0, -np.inf)
        x = x + mask
        x = self.softmax(x) @ self.fc2(embedding_matrix)
        return x
