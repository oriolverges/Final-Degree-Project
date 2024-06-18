import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Embeddings_Dataset import EmbeddingsAndSequences, get_labels

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import argparse

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def main(args):
    hidden_dim = 50
    output_dim = 1  
    num_epochs = 10  
    batch_size = args.batch_size
    learning_rate = 0.001

    print("Data: ", args.data_set)
    training_dataset = EmbeddingsAndSequences(
        weights_filepath=args.Model2_train,
        rna_fasta=args.training_data_rna_fasta,
        protein_fasta=args.training_data_protein_fasta,
    )

    train_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False)

    if args.continuous:
        train_labels = get_labels(args.data_set, args.training_data_rna_fasta, is_categorical=False)
        test_labels = get_labels(args.data_set, args.test_data_rna_fasta, is_categorical=False)
    else:
        train_labels = get_labels(args.data_set, args.training_data_rna_fasta, is_categorical=True)
        test_labels = get_labels(args.data_set, args.test_data_rna_fasta, is_categorical=True)
    
    sample_batch = next(iter(train_dataloader))[0]
    input_dim = sample_batch.size(-1)

    mlp = MLP(input_dim, hidden_dim, output_dim)
    print(mlp)
    criterion = nn.MSELoss() if args.continuous else nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    mlp.train()
    for epoch in range(num_epochs):
        for batch_idx, batch_combined_embeddings in enumerate(train_dataloader):
            X_batch = batch_combined_embeddings.cpu().detach().numpy()
            start_idx = batch_idx * batch_size
            end_idx = start_idx + X_batch.shape[0]  
            y_batch = train_labels[start_idx:end_idx]
            optimizer.zero_grad()
            outputs = mlp(torch.tensor(X_batch, dtype=torch.float32))

            if args.continuous:
                y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1)
            else:
                y_batch_tensor = torch.tensor(y_batch, dtype=torch.long)
            
            loss = criterion(outputs, y_batch_tensor)
            loss.backward()
            optimizer.step() 
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}')

    test_dataset = EmbeddingsAndSequences(
        weights_filepath=args.Model2_test,
        rna_fasta=args.test_data_rna_fasta,
        protein_fasta=args.test_data_protein_fasta,
    )
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 

    test_features = []
    
    mlp.eval()
    with torch.no_grad():
        for batch_idx, batch_combined_embeddings in enumerate(test_loader):
            test_features.append(batch_combined_embeddings)
    
        X_test_reduced = np.concatenate(test_features, axis=0)
        y_test = test_labels

        if args.continuous:
            predictions = mlp(torch.tensor(X_test_reduced, dtype=torch.float32)).numpy()

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            print(f'MLP Model MSE: {mse:.4f}')
            print(f'MLP Model MAE: {mae:.4f}')
            print(f'MLP Model R²: {r2:.4f}')
        else:
            outputs = mlp(torch.tensor(X_test_reduced, dtype=torch.float32))
            _, final_predictions = torch.max(outputs, 1)
            final_predictions = final_predictions.numpy()

            accuracy = accuracy_score(y_test, final_predictions)
            precision = precision_score(y_test, final_predictions, average='weighted')
            recall = recall_score(y_test, final_predictions, average='weighted')
            f1 = f1_score(y_test, final_predictions, average='weighted')

            print(f'MLP Model Accuracy: {accuracy:.4f}')
            print(f'MLP Model Precision: {precision:.4f}')
            print(f'MLP Model Recall: {recall:.4f}')
            print(f'MLP Model F1 Score: {f1:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--training_data_rna_fasta", type=str, help="Path of the RNA training data fasta file.")
    parser.add_argument("--training_data_protein_fasta", type=str, help="Path of the protein training data fasta file.")
    parser.add_argument("--Model2_train", type=str, help="Path of the Model2 weight file prefix.")
    parser.add_argument("--test_data_rna_fasta", type=str, help="Path of the RNA test data fasta file.")
    parser.add_argument("--test_data_protein_fasta", type=str, help="Path of the protein test data fasta file.")
    parser.add_argument("--Model2_test", type=str, help="Path of the Model2 weight file prefix for testing.")
    parser.add_argument("--data_set", type=str, help="Path of the CSV dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size value.")
    parser.add_argument("--continuous", action="store_true", help="Indicate if labels are continuous (for regression).")

    args = parser.parse_args()
    main(args)
