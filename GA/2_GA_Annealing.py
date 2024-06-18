import random
import time
from auxiliar_functions import *
import math
import pandas as pd
import sys
import argparse

def main(args):

    N = 20
    G = 5
    mu = 0.01
    T = 100.0
    C = 0.95

    sw = 0.4
    ow = 0.6

    swr = 0.1
    owr = 0.1
    lw = 0.1
    ew = 0.7

    c = 10

    b = 10
    cool = 0.6

    stop = 3
    def annealing_GA(N, G, mu, T, C, proteins, structure_weight, optimality_weight, structure_weight_refined, optimality_weight_refined, loop_weight, energy_weight, beta, cooling):
        best_fitness = float('-inf')  

        generations_no_change_best = 0
        
        start_time = time.time()

        if is_RNA(list(proteins.values())[0]) == True:
            sequences = [[proteins[protein] for sequence in range(N)] for protein in proteins]
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

                for _ in range(N):

                    individual_id = random.randint(0, N - 1)
                    individual_structure = population[protein][individual_id]
                    individual_sequence = sequences[protein][individual_id]

                    neighbor_id = (individual_id + random.randint(1, N - 1)) % N
                    neighbor_structure = population[protein][neighbor_id]
                    neighbor_sequence = sequences[protein][neighbor_id]

                    difference = fitness(neighbor_structure, optimality_scores[protein][neighbor_id], structure_weight, optimality_weight) - fitness(individual_structure, optimality_scores[protein][individual_id], structure_weight, optimality_weight)
                    if difference >= 0 or random.random() < math.exp(difference/T):
                        selected_structure = neighbor_structure
                        selected_sequence = neighbor_sequence
                    else:
                        selected_structure = individual_structure
                        selected_sequence = individual_sequence
                    
                    crossover_point = random.randint(0, len(selected_sequence) - 1)

                    child_structure  = selected_structure[:crossover_point] + selected_structure[crossover_point:]
                    child_sequence = selected_sequence[:crossover_point] + selected_sequence[crossover_point:]

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

            T *= C

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
    
    results2 = []
    result2, total_time = annealing_GA(N, G, mu, T, C, proteins, sw, ow, swr, owr, lw, ew, b, cool)

    # Loop through the result2 list and append each element to the results2 list
    for i in range(0, len(result2), 6):
        protein_name = result2[i]
        sequence = result2[i + 1]
        structure = result2[i + 2]
        fit = result2[i + 3]
        en = result2[i + 4]
        acc = result2[i + 5]

        # Append the results as a dictionary
        results2.append({
            'Algorithm': 'Annealing', 
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
            'Temperature': T,
            'Cooling Annealing': C,
            'Sequence': sequence,
            'Structure': structure,
            'Time': total_time
        })
    
    title = "GA with SA"

    df2 = pd.DataFrame(results2)

    output_file_name = args.output_file
    # Save the dataframes and titles to a text file
    with open(output_file_name, "a") as file:
        keys_string = ", ".join(proteins.keys())
        file.write(title + '\n')
        file.write(df2.to_string(index=False) + '\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file containing protein sequences")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save results")
    args = parser.parse_args()

    proteins = read_sequences_from_file(args.input_file)
    main(args)