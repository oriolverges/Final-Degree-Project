import random
import time
from auxiliary_functions import *
import math
import pandas as pd
import sys
import argparse

def main(args):
    N = 50
    G = 30
    mu = 0.01

    sw = 0.5
    ow = 0.5

    swr = 0.1
    owr = 0.1
    lw = 0.1
    ew = 0.7

    # parameters for UCT
    c = 100

    # Moves for minimizing AUP 
    b = 5000
    cool = 0.95

    def basic_GA(N, G, mu, proteins, structure_weight, optimality_weight, structure_weight_refined, optimality_weight_refined, loop_weight, energy_weight, beta, cooling, uct_c):

        total_visits = 1

        best_fitness = float('-inf')  

        generations_no_change_best = 0

        stop = (30 * G)//100

        start_time = time.time()

        if is_RNA(list(proteins.values())[0]) == True:
            protein_list = [protein_translation(proteins[protein]) for protein in proteins]
            sequences = [[''.join(random_codon_from_aa(aa) for aa in protein_list[protein]) for _ in range(N)]for protein in range(len(protein_list))]
            population = [[generate_valid_structure(proteins[protein]) for _ in range(N)]for protein in proteins]
        else:
            sequences = [[''.join(random_codon_from_aa(aa) for aa in proteins[protein]) for _ in range(N)]for protein in proteins]
            population = [[generate_valid_structure(sequence) for sequence in protein]for protein in sequences]

        optimality_scores = [[optimality(sequence) for sequence in protein] for protein in sequences]

        best_population = population
        best_sequence = sequences
        best_optimality = optimality_scores

        for generation in range(G + 1, 1):
            if execute_refined_fitness(generation, G):
                fitness_population = [[refined_fitness(sequences[protein][seq], structure_weight_refined, optimality_weight_refined, loop_weight, energy_weight)
                                    for seq in range(N)] for protein in range(len(proteins))]

            else:
                fitness_population = [[fitness(population[protein][structure], optimality_scores[protein][structure], structure_weight, optimality_weight)
                                    for structure in range(N)] for protein in range(len(proteins))]

            new_population_structures = []
            new_population_sequences = []

            for protein in range(len(proteins)):

                new_structure = []
                new_sequence = []

                uct_values = [uct_value(w, n, total_visits, c) for w, n in zip(fitness_population[protein], range(len(fitness_population[protein])))]

                for _ in range(N):

                    parents = random.choices(range(N), weights=uct_values, k=2)

                    parent1_structure = population[protein][parents[0]]
                    parent1_sequence = sequences[protein][parents[0]]
                    parent2_structure = population[protein][parents[1]]
                    parent2_sequence = sequences[protein][parents[1]]

                    crossover_point = random.randint(0, len(parent1_structure) - 1)

                    child_structure  = parent1_structure[:crossover_point] + parent2_structure[crossover_point:]
                    child_sequence = parent1_sequence[:crossover_point] + parent2_sequence[crossover_point:]

                    # Perform a mutation on the child structure with a probability of mutation_rate.
                    mutated_child_structure = ''.join(
                        c if random.random() < mu else random.choice(['(', ')', '.']) for c in child_structure)
                    
                    mutated_child_seq = ''
                    for nuc in range(0, len(child_sequence), 3):
                        new_codon = random_codon(child_sequence[nuc:nuc + 3])
                        prob_new_nuc = prob_codon_change(child_sequence, mutated_child_structure, new_codon, nuc, structure_weight, optimality_weight, beta)
                        if prob_new_nuc < random.random():
                            mutated_child_seq += new_codon
                        else:
                            mutated_child_seq += child_sequence[nuc:nuc+3]

                    new_structure.append(mutated_child_structure)
                    new_sequence.append(mutated_child_seq)

                new_population_structures.append(new_structure)
                new_population_sequences.append(new_sequence)

            population = new_population_structures
            sequences = new_population_sequences
            optimality_scores = [[optimality(sequence) for sequence in protein] for protein in sequences]

            current_best_fitness = max([max(seq_fitness) for seq_fitness in fitness_population])
            
            if current_best_fitness > best_fitness:

                best_fitness = current_best_fitness
                best_population = population
                best_sequence = sequences
                best_optimality = optimality_scores
                generations_no_change_best = 0
            
            elif current_best_fitness < best_fitness:
                if generations_no_change_best >= stop:
                    break

                else:
                    generations_no_change_best += 1

            total_visits += N * len(proteins)

            mu = random.uniform(0.01, 0.1) 

            beta *= cooling

        fitness_final = [[refined_fitness(best_sequence[protein][structure], structure_weight_refined, optimality_weight_refined, loop_weight, energy_weight) 
                        for structure in range(N)] for protein in range(len(proteins))]

        protein_indices = {protein_name: index for index, protein_name in enumerate(proteins.keys())}

        best_structure_indices = [sequence_fitness.index(max(sequence_fitness)) for sequence_fitness in fitness_final]

        end_time = time.time()

        best = []
        for protein_name, best_structure_index in zip(proteins.keys(), best_structure_indices):
            best_sequence = sequences[protein_indices[protein_name]][best_structure_index]
            best_structure = population[protein_indices[protein_name]][best_structure_index]
            best_fitness = fitness_final[protein_indices[protein_name]][best_structure_index]

            acc = 0.0
            if is_RNA(list(proteins.values())[0]) == True:
                acc = accuracy(best_sequence, protein_translation(proteins[protein_name]))
            
            else:
                acc = accuracy(best_sequence, proteins[protein_name])
            struc, energy = RNA.fold(best_sequence)
            best.extend([protein_name, best_sequence, best_structure, best_fitness, energy, acc])

        total_time = end_time - start_time

        return best, total_time

    results1 = []
    result1, total_time = basic_GA(N, G, mu, proteins, sw, ow, swr, owr, lw, ew, b, cool, c)
                                    
    for i in range(0, len(result1), 6):
        protein_name = result1[i]
        sequence = result1[i + 1]
        structure = result1[i + 2]
        fit = result1[i + 3]
        en = result1[i + 4]
        acc = result1[i + 5]
            
        results1.append({
            'Algorithm': 'BasicGA',
            'Protein': protein_name,
            'Fitness': fit,
            'Energy': en,
            'Accuracy': acc,
            'Poulation': N,
            'Generations': G,
            'sw': sw,
            'ow': ow,
            'swr': swr,
            'owr': owr,
            'lw': lw,
            'ew': ew,
            'Beta': b,
            'Cooling': cool,
            'c_uct': c,
            'Sequence': sequence,
            'Structure': structure,
            'Time': total_time
        })

    title = "Basic GA"

    df1 = pd.DataFrame(results1)

    output_file_name = args.output_file
    with open(output_file_name, "a") as file:
        keys_string = ", ".join(proteins.keys())
        file.write(title + '\n')
        file.write(df1.to_string(index=False) + '\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file containing protein sequences")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save results")
    
    args = parser.parse_args()

    proteins = read_sequences_from_file(args.input_file)
    main(args)