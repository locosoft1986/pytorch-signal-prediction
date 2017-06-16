import torch.nn as nn


class Embedder(nn.Module):

    def __init__(self, vocab_size, embedding_length):
        super(Embedder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_length)

    def forward(self, x):
        return self.embedding(x)