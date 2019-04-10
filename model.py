# Python imports.
import pdb

# PyTorch imports.
import torch
import torch.nn as nn

# Other imports.
from single_head import SingleHeadedAttention


class LanguageModel(nn.Module):
    """ Encoder network. """

    def __init__(self, vocab_size, embedding_size, window_size, hidden_size, device, dataset):
        super(LanguageModel, self).__init__()
        self.input_vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.device = device
        self.dataset = dataset  # TODO

        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(window_size, embedding_size)
        self.single_headed_attention = SingleHeadedAttention(window_size, embedding_size, hidden_size, device, dataset)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)

        self.to(device)

    def forward(self, word_ids):
        position_ids = torch.arange(start=0, end=self.window_size-1, dtype=torch.long)
        word_embeddings = self.word_embedding(word_ids)
        position_embeddings = self.position_embedding(position_ids)

        input_embeddings = word_embeddings + position_embeddings.unsqueeze(0)
        hidden = self.single_headed_attention(input_embeddings)
        skip = hidden + input_embeddings.squeeze(0)
        skipped_normed = self.norm(skip)
        logits = self.classifier(skipped_normed)
        return logits
