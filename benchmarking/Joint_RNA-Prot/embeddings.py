import os
import argparse
import torch
import pandas as pd

from lightning.fabric import Fabric

from torch.utils.data import DataLoader, Dataset
from Tokenizer import Tokenizer_Embedding, create_codon_aa_pairs, padding_function
from TransformerModel import TransformerModel

class ProteinRNAPairsDataset(Dataset):
    def __init__(self, codon_aa_pairs, ntokens, sequence_names):
        self.codon_aa_pairs = codon_aa_pairs
        self.ntokens = ntokens
        self.sequence_names = sequence_names

    def __len__(self):
        return len(self.codon_aa_pairs)

    def __getitem__(self, idx):
        pair_seq = torch.as_tensor(self.codon_aa_pairs[idx])
        sequence_name = self.sequence_names[idx]
        return pair_seq, sequence_name

def read_fasta_to_df(fasta_file):
    """
    Reads a FASTA file and returns a DataFrame with the entry names and sequences.

    Parameters:
    fasta_file (str): Path to the FASTA file.

    Returns:
    pd.DataFrame: DataFrame with 'entry_name', 'sequence', and 'protein_sequence' columns.
    """
    entry_names = []
    sequences = []
    with open(fasta_file, 'r') as file:
        sequence = ""
        entry_name = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
                entry_name = line[1:]  # Remove '>' character
                entry_names.append(entry_name)
            else:
                sequence += line
        # Append the last sequence
        if sequence:
            sequences.append(sequence)
    
    df = pd.DataFrame({"Entry": entry_names, "RNA": sequences})
    df['Protein'] = df['RNA'].apply(translate_rna_to_protein)
    return df

def translate_rna_to_protein(rna_sequence):
    """
    Translates an RNA sequence into a protein sequence using the standard genetic code.

    Parameters:
    rna_sequence (str): RNA sequence.

    Returns:
    str: Protein sequence.
    """
    genetic_code = {
        'AUG': 'M', 'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'UAU': 'Y',
        'UAC': 'Y', 'UGU': 'C', 'UGC': 'C', 'UGG': 'W', 'CUU': 'L',
        'CUC': 'L', 'CUA': 'L', 'CUG': 'L', 'CCU': 'P', 'CCC': 'P',
        'CCA': 'P', 'CCG': 'P', 'CAU': 'H', 'CAC': 'H', 'CAA': 'Q',
        'CAG': 'Q', 'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'ACU': 'T', 'ACC': 'T',
        'ACA': 'T', 'ACG': 'T', 'AAU': 'N', 'AAC': 'N', 'AAA': 'K',
        'AAG': 'K', 'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'GCU': 'A',
        'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'GAU': 'D', 'GAC': 'D',
        'GAA': 'E', 'GAG': 'E', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G',
        'GGG': 'G', 'UAA': '*', 'UAG': '*', 'UGA': '*'
    }

    protein_sequence = ""
    for i in range(0, len(rna_sequence), 3):
        codon = rna_sequence[i:i+3]
        if len(codon) == 3:
            protein_sequence += genetic_code.get(codon, 'X')  # 'X' for unknown codons
    return protein_sequence


# Main function to load data, preprocess, and get embeddings
def main(args):
    fabric = Fabric(accelerator="gpu",
                devices= args.devices,
                precision="16-mixed",
                strategy="deepspeed")
    fabric.launch()

    # Load the dataset
    dataset = read_fasta_to_df(args.data_path)

    # Preprocess the dataset
    sequence_names = dataset['Entry'].tolist()
    RNA_sequences = dataset['RNA']
    protein_sequences = dataset['Protein']

    codon_aa_pairs = [create_codon_aa_pairs(RNA_sequences[i], protein_sequences[i]) for i in range(len(dataset))]

    tokenized_sequences, ntokens = Tokenizer_Embedding(codon_aa_pairs)

    padded_sequences = padding_function(tokenized_sequences)

    data = ProteinRNAPairsDataset(padded_sequences, ntokens, sequence_names)
    
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    dataloader = fabric.setup_dataloaders(dataloader)

    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    model = fabric.to_device(model)
    fabric.load_raw(args.weights_path, model)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            sequence, names = batch
            embeddings = model(sequence)
            # Open the file in append mode for each batch
            with open(os.path.join(args.embedding_dir, f"{args.output_name}.pt"), 'ab') as embeddings_file:
                torch.save(embeddings, embeddings_file)

    print("Embeddings of: ", args.data_path, " saved to: ", embeddings_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file (file in tsv format)')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the pretrained weights file')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Path to store the Embeddings')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output file containing the embeddings')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--emsize', type=int, default=512, help='Embedding size (default: 512)')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--nhid', type=int, default=1024, help='Hidden size of the feedforward layers (default: 1024)')
    parser.add_argument('--nlayers', type=int, default=4, help='Number of encoder layers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability (default: 0.2)')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (default: 1)')

    args = parser.parse_args()
    main(args)
