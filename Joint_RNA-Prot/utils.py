import torch
import torch.nn as nn
from torch import inf

def estimate_parameters(vocab_size, embedding_dim, num_layers, num_heads, dff):
   
    # Embedding parameters
    embedding_params = vocab_size * embedding_dim

    # Parameters per transformer layer
    per_layer_params = (
        # Multi-head attention parameters (query, key, value matrices, and output projection)
        3 * embedding_dim * embedding_dim * num_heads + embedding_dim * embedding_dim +
        # Feed-forward network parameters
        dff * embedding_dim + dff +
        # Layer normalization parameters (two per layer)
        2 * embedding_dim * 2
    )

    # Total parameters
    total_params = embedding_params + per_layer_params * num_layers

    return total_params

def train(fabric, model, dataloader, optimizer, ntoken, epoch):
    model.train()
    total_loss = 0.0
    
    criterion_sequence = nn.CrossEntropyLoss()  # For sequence predictions

    for batch_idx, (sequence) in enumerate(dataloader):
        print("Sequence: ", sequence)
        optimizer.zero_grad()
        output = model(sequence)

        print(output.size())
        # Calculate loss for sequence predictions
        loss = criterion_sequence(output.transpose(1, 2), sequence)

        fabric.backward(loss)        
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validation(model, dataloader, ntoken):
    model.eval()
    total_loss = 0.0

    criterion_sequence = nn.CrossEntropyLoss()  # For sequence predictions

    with torch.no_grad():  
        for batch_idx, (sequence) in enumerate(dataloader):

            sequence_output = model(sequence)

            # Calculate loss for sequence predictions
            loss = criterion_sequence(sequence_output.transpose(1, 2), sequence)

            total_loss += loss.item()

    return total_loss / len(dataloader)