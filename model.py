# Python imports.
import pdb

# PyTorch imports.
import torch
import torch.nn as nn

# Other imports.
from transformer_block import TransformerBlock


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
        self.transformer1 = TransformerBlock(embedding_size, window_size, hidden_size, device, dataset)
        self.transformer2 = TransformerBlock(hidden_size, window_size, hidden_size, device, dataset)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, vocab_size)
        )

        self.to(device)

    def forward(self, word_ids):
        position_ids = torch.arange(start=0, end=self.window_size, dtype=torch.long, device=self.device)
        word_embeddings = self.word_embedding(word_ids)
        position_embeddings = self.position_embedding(position_ids)

        input_embeddings = word_embeddings + position_embeddings.unsqueeze(0)
        hidden = self.transformer1(input_embeddings)
        hidden = self.transformer2(hidden)
        logits = self.classifier(hidden)
        return logits, hidden
