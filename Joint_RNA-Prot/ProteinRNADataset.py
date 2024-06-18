import torch
from torch.utils.data import Dataset
from torch import inf

class ProteinRNAPairsDataset(Dataset):
    def __init__(self, codon_aa_pairs, ntokens):
        self.codon_aa_pairs = codon_aa_pairs
        self.ntokens = ntokens

    def __len__(self):
        return len(self.codon_aa_pairs)

    def __getitem__(self, idx):
        pair_seq = torch.as_tensor(self.codon_aa_pairs[idx])
        return pair_seq
