import torch
import torch.nn as nn
from torch import inf

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model, dtype=torch.float32)

    def forward(self, x):
        return self.embedding(x)