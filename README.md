# Prediction and optimization of mRNA stability via Language Models
## Final Degree Project

- **Title:** Prediction and optimization of mRNA stability via Language Models
- **Author:** Oriol Vergés (oriol.verges@alum.esci.upf.edu)
- **Degree:** Bachelor`s Degree in Bioinformatics (BDBI)
- **Institution:** ESCI-UPF
- **Course:** 2023-2024
- **Date:** 18/06/2024
- **Link:**

This repository contains the main scripts and files used and generated for this Final Degree Project.

### Abstract


This repository is organized into three main folders, each corresponding to a specific task performed in the project:

- **GA**: This folder contains all Python scripts related to the execution of four different genetic algorithm approaches. It also includes the sequences used for parametrization (protein1 and protein2) and the sequence used for comparing Elitism and Monte Carlo methods (4FC).
- **joint_RNA-Prot**: This folder includes all Python scripts used to train the encoder-only transformer model. The model generates a single embedding composed of sequences of pairs, each consisting of a codon and its encoded amino acid.
- **benchmarking**: This folder contains all files and executables required for benchmarking tasks. It covers both the concatenated embeddings and the embeddings obtained from the joint RNA-Protein model.
