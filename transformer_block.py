# Python imports.
import pdb

# PyTorch imports.
import torch.nn as nn

# Other imports.
from multi_head import MultiHeadedAttention


class TransformerBlock(nn.Module):
    """ Single unit of the transformer model. """

    def __init__(self, embedding_size, window_size, hidden_size, device, dataset):
        super(TransformerBlock, self).__init__()
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.device = device
        self.dataset = dataset  # TODO

        self.multi_headed_attention = MultiHeadedAttention(window_size, embedding_size, hidden_size, device, dataset)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        self.to(device)

    def forward(self, input_embeddings):
        hidden = self.multi_headed_attention(input_embeddings)
        skip = hidden + input_embeddings.squeeze(0)
        skipped_normed = self.norm1(skip)
        modified_embeddings = self.fc(skipped_normed)
        output_embeddings = modified_embeddings + skipped_normed
        return self.norm2(output_embeddings)
