import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from Bio import SeqIO
import numpy as np
import pandas as pd
import csv
import os
import pickle


class EmbeddingsAndSequences(Dataset):
    def __init__(self, weights_filepath, rna_fasta, protein_fasta):
        self.weights_filepath = weights_filepath #CodonBERT
        self.rna_fasta = rna_fasta
        self.protein_fasta = protein_fasta

        self.sequences = list(SeqIO.parse(open(self.rna_fasta), "fasta"))
        self.sequence_names = [seq.id for seq in self.sequences]

       # Determine maximum sequence lengths
        self.max_rna_length = self.get_max_length(self.rna_fasta)
    
    def load_embeddings(self, filepath, idx):
        #CodonBERT embeddings in picke format
        with open(filepath, 'rb') as f:
            data = torch.load(f, map_location=torch.device('cpu'))
        data = data[idx]

        return torch.tensor(data)

    def get_max_length(self, fasta_filepath):
        max_length = 0
        
        for seq_record in SeqIO.parse(fasta_filepath, "fasta"):
            max_length = max(max_length, len(seq_record))
                    
        return max_length

    def __len__(self):
        return len(self.sequence_names)
    
    def __getitem__(self, idx):   
        # Load embedding
        embedding = self.load_embeddings(self.weights_filepath, idx)
        
        return pad_tensor(embedding, self.max_rna_length)


def pad_tensor(tensor, max_rna_length):
    # Pad RNA tensor along sequence length dimension
    RNA_padded = F.pad(tensor, (0, 0, 0, max_rna_length - tensor.size(0)), mode='constant', value=0)    
    return RNA_padded.flatten()


def get_labels(csv_file, fasta_file, is_categorical=False):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Dictionary to store values for each sequence
    values = list()
    
    # Parse the fasta file
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Extract the sequence name and index
        sequence_name = record.id
        index = int(sequence_name.split("_")[-1])
        
        if is_categorical:
            # Keep labels as strings or integers
            values.append(df.at[index - 1, 'Value'])
        else:
            # Convert label to float for regression
            values.append(float(df.at[index - 1, 'Value']))

    return values