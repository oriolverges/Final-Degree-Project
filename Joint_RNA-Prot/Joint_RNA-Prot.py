import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import argparse
import math

from torch.utils.data import Dataset
from torch import inf
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from distutils.util import strtobool

from Tokenizer import *
from ProteinRNADataset import ProteinRNAPairsDataset
from PositionalEncoding import PositionalEncoding
from FeatureEmbedding import FeatureEmbedding
from TransformerModel import TransformerModel
from utils import estimate_parameters, train, validation

def main(args):
    
    # Initialize Wandb and Fabric
    wandb_logger = WandbLogger(
        project = args.wandb_project,
        name = args.wandb_name,
        config = {
        "nodes": args.nodes,
        "devices": args.devices,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "emsize": args.emsize,
        "nhead": args.nhead,
        "nlayers": args.nlayers,
        "nhid": args.nhid,
        "dropout": args.dropout,
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate
        })
    
    
    fabric = Fabric(accelerator="gpu",
                    devices= args.devices,
                    precision="16-mixed",
                    strategy="deepspeed",
                    loggers=wandb_logger)


    fabric.launch()

    # Define task
    directory_path = os.path.join(args.project_dir, args.wandb_name + "-" + args.jobid)
    if fabric.global_rank == 0:
        os.makedirs(directory_path)

    if args.subset_size != None:
        training_file = f'training_{args.subset_size//1000}k.txt'
        validation_file = f'validation_{args.subset_size//1000}k.txt'

    else:
        training_file = f'training_whole_data.txt'
        validation_file = f'validation_whole_data.txt'

    '''
    Input data frame Uniref50 containing protein sequence in position 4 and RNA sequence in position 3.
    '''
    train_path = os.path.join(args.data_dir, args.train_data)
    val_path = os.path.join(args.data_dir, args.val_data)

    training_dataset = pd.read_csv(train_path, sep='\t', header = 'infer')
    validation_dataset = pd.read_csv(val_path, sep='\t', header = 'infer')

    if args.subset_size != None:
        training_dataset = training_dataset.iloc[:args.subset_size]
        subset_size = int(args.subset_size * 0.15)
        validation_dataset = validation_dataset.iloc[:subset_size]


    # FOR RNA SEQUENCES:
    training_RNA = training_dataset.iloc[:, 1]
    validation_RNA = validation_dataset.iloc[:, 1]


    # FOR PROTEIN SEQUENCES:
    training_protein = training_dataset.iloc[:, 2]
    validation_protein = validation_dataset.iloc[:, 2]


    # CREATE PAIRS OF CODON-AA:
    training_codon_aa_pairs = []
    for seq in range(len(training_dataset)):
        training_codon_aa_pairs.append(create_codon_aa_pairs(training_RNA[seq], training_protein[seq]))

    validation_codon_aa_pairs = []
    for seq in range(len(validation_dataset)):
        validation_codon_aa_pairs.append(create_codon_aa_pairs(validation_RNA[seq], validation_protein[seq]))


    # TOKENIZE
    training_tokenized, ntokens, tokenizer = Pairs_Tokenizer(training_codon_aa_pairs)
    validation_tokenized, ntokens, tokenizer = Pairs_Tokenizer(validation_codon_aa_pairs, tokenizer)

    # To the obtained number of tokens from the sequences we add: token for padding (0)
    ntokens = ntokens + 1

    # PADDING
    training_padded = padding_function(training_tokenized)
    validation_padded = padding_function(validation_tokenized)

    # Datasets and DataLoaders
    train_dataset = ProteinRNAPairsDataset(training_padded, ntokens)
    val_dataset = ProteinRNAPairsDataset(validation_padded, ntokens)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)


    # Show Metrics
    fabric.print("SaPROT like Model\n", "-" * 89)
    fabric.print("Training dataset: ", len(train_dataset), "sequences")
    fabric.print("Training batches: ", len(train_dataloader) * args.devices * args.nodes)
    fabric.print("Validation dataset: ", len(val_dataset), "sequences")
    fabric.print("Validation batches: ", len(val_dataloader))
    fabric.print("Batch size: ", args.batch_size)
    fabric.print("Number of tokens: ", ntokens)
    fabric.print("Sequence length after padding: ", len(train_dataset[0]))
    fabric.print("Sequence length after padding: ", len(train_dataset[3]))
    fabric.print("Tokenizer: ", tokenizer)

    
    # Model
    model = TransformerModel(ntoken=ntokens, d_model=args.emsize, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    if args.pretrained:
        state = {"model": model, "optimizer": optimizer}
        fabric.load(args.weights, state)
        fabric.print("Loaded weights from:", args.weights)

    # Count trainable parameters
    total_params = estimate_parameters(ntokens, args.emsize, args.nlayers, args.nhead, args.nhid)
    fabric.print(f"Trainable parameters: {total_params}", "\n", "-" * 89)

    # Train/validation loop
    train_losses = list()
    validation_losses = list()
    best_validation_loss = float('inf')  # Initialize with a high value

    for epoch in range(1, args.num_epochs + 1):
        with open(os.path.join(directory_path, training_file), 'a') as f:

            train_loss = train(fabric, model, train_dataloader, optimizer, ntokens, epoch)
            train_losses.append(train_loss)
            
            print(f"Epoch {epoch}/{args.num_epochs}, Train Loss: {train_loss}")
            f.write(f"{epoch}\t{train_loss}\n")       
 
        f.close()

        with open(os.path.join(directory_path, validation_file), 'a') as f:
            val_loss = validation(model, val_dataloader, ntokens)
            validation_losses.append(val_loss)
            
            print(f"Epoch {epoch}/{args.num_epochs}, Validation Loss: {val_loss}")
            f.write(f"{epoch}\t{val_loss}\n")

        f.close()

        fabric.log_dict({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if args.subset_size != None:
            subset_dir = os.path.join(directory_path, f'subset_{args.subset_size//1000}', 'k_model')
        else:
            subset_dir = os.path.join(directory_path, f'whole_data')
        os.makedirs(subset_dir, exist_ok=True)  # Create directory if it doesn't exist
        model_weights_path = os.path.join(subset_dir, f'model_weights_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_weights_path)

        # Save the best model based on validation loss
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_model_weights_path = os.path.join(subset_dir, f'best_model_weights.pt')
            torch.save(model.state_dict(), best_model_weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("jobid", type=str, help="job id (e.g. 123456)")

    wandb = parser.add_argument_group('wandb configuration')
    wandb.add_argument("--wandb_project", type=str, required=True, help="name of the wandb project")
    wandb.add_argument("--wandb_name", type=str, required=True, help="name of the wandb run")

    files = parser.add_argument_group('Files information: input and output files, plots...')
    files.add_argument("--project_dir", type=str, default="/home/overges", help="location where project results will be stored")
    files.add_argument("--data_dir", type=str, default="/home/overges/Data", help="location where training data is stored")
    files.add_argument('--train_data', metavar="t", type=str, required=True, help='Path to the training data input file (file in tsv format)')
    files.add_argument('--val_data', metavar="t", type=str, required=True, help='Path to the validation data input file (file in tsv format)')
    files.add_argument('--subset_size', type=int, default=None, help='Subset size (default: None)')
    files.add_argument('--pretrained', type=lambda b:bool(strtobool(b)), default=True, help="True if pretrained weights are provided")
    files.add_argument('--weights', metavar="w", type=str, help="path of the pretrained weights")

    resources = parser.add_argument_group('computational resources')
    resources.add_argument("--nodes", type=int, default=1, help="number of nodes")
    resources.add_argument("--devices", type=int, default=1, help="number of devices per node")
    resources.add_argument("--num_workers", type=int, default=1, help="number of workers (cpus) for data loading")

    hyperparams = parser.add_argument_group('training hyperparameters')
    hyperparams.add_argument('--emsize', type=int, default=512, help='Embedding size (default: 512)')
    hyperparams.add_argument('--nhead', type=int, default=8, help='Number of attention heads (default: 8)')
    hyperparams.add_argument('--nhid', type=int, default=1024, help='Hidden size of the feedforward layers (default: 1024)')
    hyperparams.add_argument('--nlayers', type=int, default=4, help='Number of encoder layers (default: 4)')
    hyperparams.add_argument('--dropout', type=float, default=0.2, help='Dropout probability (default: 0.2)')
    hyperparams.add_argument('--num_epochs', type=int, default=30, help='Number of epochs (default: 30)')
    hyperparams.add_argument('--learning_rate', type=float, default=0.000001, help='Learning rate (default: 0.000001)')
    hyperparams.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')

    args = parser.parse_args()
    main(args)