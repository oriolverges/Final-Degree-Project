import pandas as pd
import numpy as np
import statistics

def Pairs_Tokenizer(df_sequences, tokenizer=None):
    '''
    Protein sequences tokenizer:
    - Input = Serie of pairs of RNA (codons) + Amino acid
    - Output = Sequences tokenized + number of tokens used
    '''
    if tokenizer is None:
        tokenizer = dict()

    tokenized_sequences = []

    # Iterate over all sequences
    for sequence in df_sequences:
        seq = []

        for pair in sequence:
            if pair not in tokenizer:          
                # +1 is used to avoid using token 0, to save it for padding.      
                tokenizer[pair] = len(tokenizer) + 1
            
            seq.append(tokenizer[pair])
        
        tokenized_sequences.append(seq)
    return tokenized_sequences, len(tokenizer), tokenizer


def padding_function(input_list, pad_value = 0):
    '''
    Function to pad a serie of sequences:
    - Input = Protein sequences tokenized (padding value is set to 0 by default, if necessary, change it)
    - Output = List containing padded protein sequences
    '''

    # Calculate lengths
    lengths = [len(seq) for seq in input_list]
    # use the mode as the expected length
    expected_length = int(statistics.mode(lengths))
    padded_sequences = []

    for sequence in input_list:

        current_length = len(sequence)

        if current_length < expected_length:
            padding = [pad_value] * (expected_length - current_length)
            padding = sequence + padding

        elif current_length > expected_length:
            padding = sequence[:expected_length]
        
        else:
            padding = sequence

        padded_sequences.append(padding)

    return padded_sequences

def create_codon_aa_pairs(rna_sequence, protein_sequence):
    codon_aa_pairs = []
    for i in range(0, len(protein_sequence)):
        codon = rna_sequence[i * 3: i * 3 + 3]
        aa = protein_sequence[i]
        codon_aa_pairs.append((codon + aa))
    return codon_aa_pairs


def Tokenizer_Embedding(df_sequences):
    '''
    Protein sequences tokenizer:
    - Input = Serie of pairs of RNA (codons) + Amino acid
    - Output = Sequences tokenized + number of tokens used
    '''

    tokenizer = {'AUGM': 1, 'CUAL': 2, 'UUCF': 3, 'AAAK': 4, 'CAGQ': 5, 
    'GACD': 6, 'GAUD': 7, 'UAUY': 8, 'UCAS': 9, 'ACCT': 10, 'CCGP': 11, 
    'GUAV': 12, 'UUUF': 13, 'GGGG': 14, 'ACAT': 15, 'AUCI': 16, 'GAGE': 17, 
    'UCCS': 18, 'AGUS': 19, 'AACN': 20, 'CCCP': 21, 'GUUV': 22, 'GGUG': 23, 
    'GCCA': 24, 'CCUP': 25, 'AAUN': 26, 'CGAR': 27, 'UUAL': 28, 'AUAI': 29, 
    'UUGL': 30, 'CGGR': 31, 'CUGL': 32, 'UGUC': 33, 'ACGT': 34, 'UACY': 35, 
    'GAAE': 36, 'UCUS': 37, 'AAGK': 38, 'GCGA': 39, 'AUUI': 40, 'CAAQ': 41, 
    'CGUR': 42, 'GUCV': 43, 'CUUL': 44, 'UGGW': 45, 'CCAP': 46, 'GGAG': 47, 
    'GCUA': 48, 'AGCS': 49, 'CAUH': 50, 'GCAA': 51, 'GGCG': 52, 'CACH': 53, 
    'UCGS': 54, 'GUGV': 55, 'AGGR': 56, 'CUCL': 57, 'AGAR': 58, 'UGCC': 59, 
    'CGCR': 60, 'ACUT': 61, 'UAGX': 62, 'UGAX': 63, 'UAAX': 64, 'XXXX':65}

    if tokenizer is None:
        tokenizer = dict()

    tokenized_sequences = []

    # Iterate over all sequences
    for sequence in df_sequences:
        seq = []

        for pair in sequence:
            if pair not in tokenizer:          
                # Unknown codon-amino acid      
                seq.append(tokenizer['XXXX'])
            else:
                seq.append(tokenizer[pair])
        
        tokenized_sequences.append(seq)
    return tokenized_sequences, 69