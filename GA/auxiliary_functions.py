import random
import math
import RNA
import argparse

codon_table = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAU', 'AAC'],
    'D': ['GAU', 'GAC'],
    'C': ['UGU', 'UGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],
    'H': ['CAU', 'CAC'],
    'I': ['AUU', 'AUC', 'AUA'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'K': ['AAA', 'AAG'],
    'M': ['AUG'],
    'F': ['UUU', 'UUC'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'W': ['UGG'],
    'Y': ['UAU', 'UAC'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'X': ['UAA', 'UGA', 'UAG']
}

codon_optimality = {
    'AAA': 0.9, 'AAC': 0.85, 'AAG': 0.9, 'AAU': 0.8,
    'ACA': 0.7, 'ACC': 0.75, 'ACG': 0.7, 'ACU': 0.73,
    'AGA': 0.6, 'AGC': 0.65, 'AGG': 0.6, 'AGU': 0.6,
    'AUA': 0.9, 'AUC': 0.95, 'AUG': 1.0, 'AUU': 0.93,
    
    'CAA': 0.8, 'CAC': 0.83, 'CAG': 0.8, 'CAU': 0.8,
    'CCA': 0.7, 'CCC': 0.75, 'CCG': 0.7, 'CCU': 0.73,
    'CGA': 0.6, 'CGC': 0.6, 'CGG': 0.65, 'CGU': 0.6,
    'CUA': 0.85, 'CUC': 0.85, 'CUG': 0.9, 'CUU': 0.87,
    
    'GAA': 0.9, 'GAC': 0.9, 'GAG': 0.93, 'GAU': 0.9,
    'GCA': 0.8, 'GCC': 0.8, 'GCG': 0.8, 'GCU': 0.8,
    'GGA': 0.75, 'GGC': 0.75, 'GGG': 0.78, 'GGU': 0.75,
    'GUA': 0.8, 'GUC': 0.8, 'GUG': 0.83, 'GUU': 0.8,
    
    'UAA': 0.1, 'UAC': 0.85, 'UAG': 0.1, 'UAU': 0.83,
    'UCA': 0.7, 'UCC': 0.73, 'UCG': 0.7, 'UCU': 0.7,
    'UGA': 0.1, 'UGC': 0.95, 'UGG': 1.0, 'UGU': 0.9,
    'UUA': 0.85, 'UUC': 0.95, 'UUG': 0.9, 'UUU': 0.93
}

codon_to_amino_acid = {}
for amino_acid, codons in codon_table.items():
    for codon in codons:
        codon_to_amino_acid[codon] = amino_acid


### LIST OF FUNCTIONS TO IDENTIFY AND OR GENERATE SEQUENCES/STRUCTURES: ###

# Differentiate betwen RNA and protein sequences:
def is_RNA(sequence):
    if 'U' in sequence:
        return True
    else:
        return False

# Obtain a random codon that codifies for the given amino acid
def random_codon_from_aa(aminoacid):

    return random.choice(codon_table[aminoacid])

# Obtain a random codon given a codon
def random_codon(codon):
    aminoacid = next(key for key, value in codon_table.items() if codon in value)
    synonymous_codons = codon_table[aminoacid]
    return random.choice(synonymous_codons)

# Generate a valid structure taking into account nucleotides pairing.
def generate_valid_structure(sequence):

    stack = []
    structure = ['.'] * len(sequence)

    for i in range(len(sequence)):
        nucleotide = sequence[i]
        if nucleotide in ["A", "G"]:
            stack.append(i)

        elif nucleotide in ["U", "C"]:
            if stack:
                j = stack.pop()
                if (sequence[j], nucleotide) in [('A', 'U'), ('G', 'C')]:
                    structure[j] = "("
                    structure[i] = ")"
    
    return ''.join(structure)

### LIST OF FUNCTIONS TO COMPUTE OPTIMALITY/FITNESS OF A GIVEN SEQUENCE AND STRUCTURE: ###

def refined_fitness(sequence, sw, ow, lw, ew):
    structure, energy = RNA.fold(sequence)
    AUP = structure.count(".")/len(structure)
    loop_length = compute_loop_length(structure)
    stacking_energy = compute_stacking_energy(structure, energy)

    return (sw * (1 - AUP)) + (ow * optimality(sequence)) + (lw * loop_length) + (ew * stacking_energy)

def compute_loop_length(structure):
    stack = []
    loop_length = [0]

    for char in structure:
        if char == "(":
            stack.append(char)
        
        elif char == ")":
            loop_length.append(len(stack))
            stack.pop()
        
        elif stack and char == ".":
            loop_length[-1] += 1
        
    if not loop_length:
        return 0
    else:
        return sum(loop_length)/len(loop_length)

def compute_stacking_energy(structure, energy):
    if structure.count("(") != structure.count(")"):
        return 0 

    else:
        return -energy


def fitness(structure, optimality_score, sw, ow):
    AUP = structure.count(".")/len(structure)

    return (sw * (1- AUP)) + (ow * optimality_score)

# Function to compute optimality scores for a sequence
def optimality(sequence):    
    score = 0

    for nuc in range(0, len(sequence), 3):
        if nuc + 3 <= len(sequence):
            score += codon_optimality[sequence[nuc:nuc + 3]]
    
    return score/(len(sequence)/3)


def uct_value(total_score, num_visits, total_visits, exploration_param):
    """UCT value calculation for node selection."""
    if num_visits == 0:
        return float(1e10)
    exploitation = total_score / num_visits
    exploration = exploration_param * math.sqrt(math.log(total_visits) / num_visits)
    return exploitation + exploration

def prob_codon_change(sequence, structure, new_codon, position, sw, ow, beta):
    if sequence[position: position + 3] != new_codon:
        codon1_optimality = optimality(sequence)
        codon1_fitness = fitness(structure, codon1_optimality, sw, ow)

        codon2_sequence = sequence[:position] + new_codon + sequence[position + 3:]
        codon2_optimality = optimality(codon2_sequence)
        codon2_fitness = fitness(structure, codon2_optimality, sw, ow)

        return math.exp(-beta*(codon2_fitness - codon1_fitness))

    else:
        return 1

def execute_refined_fitness(generation, G):
    ranges = [(20, 25), (40, 45), (60, 65), (80, 85)]
    percentage = generation * 100/G
    for percentage_start, percentage_end in ranges:
        if percentage > percentage_start and percentage < percentage_end:
            return True

### LIST OF FUNCTIONS TO COMPUTE OTHER STATISTICS RELATED WITH PERFORMANCE: ###

def common_letters(sequence1, sequence2):
    count = 0
    for char1 in range(len(sequence1)):
        if sequence1[char1] == sequence2[char1]:
            count += 1
    return count

def accuracy(sequence, target):
    translated = protein_translation(sequence)
    
    score = common_letters(translated, target)/len(target)
    return score

# function to translate from a RNA sequence to a protein sequence
def protein_translation(sequence):
    return ''.join(codon_to_amino_acid[sequence[nuc:nuc+3]] for nuc in range(0, len(sequence), 3))

### FUNCTION TO READ SEQUENCES FROM FILES AND RETURN THEM IN A DICTIONARY ###

def read_sequences_from_file(input_file):
    """
    Read sequences from an input file and construct the 'proteins' dictionary.

    Args:
    input_file (str): Path to the input file containing protein sequences.

    Returns:
    dict: Dictionary where keys are protein names and values are lists of sequences.
    """
    proteins = {}
    current_protein = None

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                current_protein = line[1:]
                if current_protein not in proteins:
                    proteins[current_protein] = []
            else:
                proteins[current_protein].append(line)    

    return proteins