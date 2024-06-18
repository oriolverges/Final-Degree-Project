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
    def __init__(self, weights_filepath, rna_fasta, protein_fasta, esm_base_dir):
        self.weights_filepath = weights_filepath #CodonBERT
        self.rna_fasta = rna_fasta
        self.protein_fasta = protein_fasta
        self.esm_base_dir = esm_base_dir

        self.sequences = list(SeqIO.parse(open(self.rna_fasta), "fasta"))
        self.sequence_names = [seq.id for seq in self.sequences]

       # Determine maximum sequence lengths
        self.max_rna_length = self.get_max_length(self.rna_fasta)
        self.max_protein_length = self.get_max_length(self.protein_fasta)

    def get_rna_embedding_dim(self):
        # Load one RNA embedding to determine its dimension
        example_rna_embedding = self.load_rna_embeddings(self.weights_filepath, 0, 1)[0]
        return example_rna_embedding.size()[1]

    def get_protein_embedding_dim(self):
        # Load one protein embedding to determine its dimension
        example_protein_embedding = self.load_prot_embeddings(self.sequence_names[0])
        return example_protein_embedding.size(1)

    def load_rna_embeddings(self, filepath, idx):
        #CodonBERT embeddings in picke format
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        data = data[idx]

        return torch.tensor(data)
    
    def load_prot_embeddings(self, sequence_name):
        # Implement loading protein embeddings
        filepath = os.path.join(self.esm_base_dir, f"{sequence_name}.pt")

        return next(iter(torch.load(filepath)['representations'].values()))

    def get_max_length(self, fasta_filepath):
        max_length = 0
        
        for seq_record in SeqIO.parse(fasta_filepath, "fasta"):
            max_length = max(max_length, len(seq_record))
                    
        return max_length

    def __len__(self):
        return len(self.sequence_names)
    
    def __getitem__(self, idx):   
        sequence = self.sequences[idx]
        sequence_name = sequence.id

        # Load RNA embedding
        rna_embedding = self.load_rna_embeddings(self.weights_filepath, idx)

        # Load protein embedding
        protein_embedding = self.load_prot_embeddings(sequence_name)

        # Pad and concatenate embeddings
        combined_embeddings = []

        combined_embeddings = pad_and_concat(rna_embedding, protein_embedding, self.max_rna_length, self.max_protein_length)
        
        return combined_embeddings


def pad_and_concat(RNA_tensor, protein_tensor, max_rna_length, max_protein_length):

    # Pad RNA tensor along sequence length dimension
    RNA_padded = F.pad(RNA_tensor, (0, 0, 0, max_rna_length - RNA_tensor.size(0)), mode='constant', value=0)
    # Pad protein tensor along sequence length dimension
    protein_padded = F.pad(protein_tensor, (0, 0, 0, max_protein_length - protein_tensor.size(0)), mode='constant', value=0)
    
    # Determine the maximum of the padded lengths
    max_length = max(RNA_padded.size(1), protein_padded.size(1))    
    
    # Pad RNA tensor along embedding dimension
    RNA_padded = F.pad(RNA_padded, (0, max_length - RNA_padded.size(1)))  
    # Pad protein tensor along embedding dimension
    protein_padded = F.pad(protein_padded, (0, max_length - protein_padded.size(1)))
    
    # Concatenate tensors
    combined_tensor = torch.cat((RNA_padded, protein_padded), dim=0)
    
    return combined_tensor.flatten()


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