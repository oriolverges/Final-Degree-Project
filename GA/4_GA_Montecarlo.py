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

    sw = 0.4
    ow = 0.6

    swr = 0.1
    owr = 0.1
    lw = 0.1
    ew = 0.7

    b = 10
    cool = 0.6

    stop = 3
    def montecarlo_GA(N, G, mu, proteins, structure_weight, optimality_weight, structure_weight_refined, optimality_weight_refined, loop_weight, energy_weight, beta, cooling):

        best_fitness = float('-inf')  

        generations_no_change_best = 0

        stop = 3

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

                cumulative_distribution = [sum(fitness_population[protein][:i+1]) for i in range(N)]
                total_fitness = sum(fitness_population[protein])
                normalized_distribution = [cumulative / total_fitness for cumulative in cumulative_distribution]

                new_structure = []
                new_sequence = [] 

                for _ in range(N):

                    number1 = random.random()
                    number2 = random.random()

                    select_id1 = None
                    select_id2 = None

                    for i, cumulative_prop in enumerate(normalized_distribution):
                        if number1 <= cumulative_prop:
                            selected_id1 = i
                            break
                    
                    selected_structure1 = population[protein][selected_id1]
                    selected_sequence1 = sequences[protein][selected_id1]

                    # Select another individual using the same method 
                    for i, cumulative_prob in enumerate(normalized_distribution):
                        if number2 <= cumulative_prob:
                            selected_id2 = i
                            break
                
                    selected_structure2 = population[protein][selected_id2]
                    selected_sequence2 = sequences[protein][selected_id2]

                    crossover_point = random.randint(0, len(selected_structure1) - 1)

                    child_structure  = selected_structure1[:crossover_point] + selected_structure2[crossover_point:]
                    child_sequence = selected_sequence1[:crossover_point] + selected_sequence2[crossover_point:]

                    mutated_child_structure = ''.join(
                        c if random.random() > mu else random.choice(['(', ')', '.']) for c in child_structure)
                
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

    results4 = []
    result4, total_time = montecarlo_GA(N, G, mu, proteins, sw, ow, swr, owr, lw, ew, b, cool)

    # Loop through the 4 list and append each element to the results4 list
    for i in range(0, len(result4), 6):
        protein_name = result4[i]
        sequence = result4[i + 1]
        structure = result4[i + 2]
        fit = result4[i + 3]
        en = result4[i + 4]
        acc = result4[i + 5]
        # Append the results as a dictionary
        results4.append({
                'Algorithm': 'Montecarlo',
                'Protein': protein_name,
                'Sequence': sequence,
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
                'Sequence': sequence,
                'Structure': structure,
                'Time': total_time
            })

    title = "Monte Carlo GA"

    df4 = pd.DataFrame(results4)

    output_file_name = args.output_file
    with open(output_file_name, "a") as file:
        keys_string = ", ".join(proteins.keys())
        file.write(title + '\n')
        file.write(df4.to_string(index=False) + '\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file containing protein sequences")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save results")
    args = parser.parse_args()

    # Read sequences from the input file
    proteins = read_sequences_from_file(args.input_file)
    main(args)