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
***
### Abstract
mRNA vaccines have revolutionized vaccinology, yet their efficacy is constrained by RNA hydrolysis. The stability of mRNA sequences crucially depends on strategic codon selection and minimizing free energy. Traditional optimization methods, such as Genetic Algorithms (GAs), while effective, face challenges in efficiently navigating the vast search space inherent to mRNA sequences.
This study explores the feasibility of leveraging Natural Language Processing (NLP), specifically transformer architectures, to assess the informativeness of mRNA and protein sequences for predictive tasks. Through extensive experimentation and benchmarking, we investigate whether NLP-driven approaches can effectively harness the rich information encoded in mRNA and protein sequences.
Two primary methodologies are examined: embedding concatenation and transformer-based models. These approaches are applied to predict various downstream properties of mRNA sequences using Random Forest and Multi-Layer Perceptron machine learning algorithms.
Our findings suggest that NLP-driven techniques hold promise in extracting meaningful insights from mRNA and protein sequences. While not yet optimizing sequence stability directly, these preliminary results lay the groundwork for future development of encoder-decoder models tailored for mRNA optimization tasks.
This research contributes to advancing the understanding of mRNA sequence dynamics and informs future directions in RNA-based therapeutic design and vaccine development, highlighting the potential of NLP in enhancing predictive models for mRNA stability.

**Supplementary Materials**: 

***
This repository is organized into three main folders, each corresponding to a specific task performed in the project:

- **GA**: This folder contains all Python scripts related to the execution of four different genetic algorithm approaches. It also includes the sequences used for parametrization (protein1 and protein2) and the sequence used for comparing Elitism and Monte Carlo methods (4FC).
- **joint_RNA-Prot**: This folder includes all Python scripts used to train the encoder-only transformer model. The model generates a single embedding composed of sequences of pairs, each consisting of a codon and its encoded amino acid.
- **benchmarking**: This folder contains all files and executables required for benchmarking tasks. It covers both the concatenated embeddings and the embeddings obtained from the joint RNA-Protein model. Data to benchmark can be downloaded [here](https://drive.google.com/drive/folders/1k_-aU9SQS_ZmfRlpTfNnGuSxtFhmaC7m?usp=sharing).
