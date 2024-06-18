import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from Embeddings_Dataset import EmbeddingsAndSequences, get_labels

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
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
    output_dim = 10  
    num_epochs = 10  
    batch_size = args.batch_size
    learning_rate = 0.001

    training_dataset = EmbeddingsAndSequences(
        weights_filepath=args.Model2_train,
        rna_fasta=args.training_data_rna_fasta,
        protein_fasta=args.training_data_protein_fasta,
    )

    train_dataloader = DataLoader(dataset=training_dataset, batch_size=args.batch_size, shuffle=False)

    if args.continuous:
        # Load labels
        train_labels = get_labels(args.data_set, args.training_data_rna_fasta, is_categorical=False)
        test_labels = get_labels(args.data_set, args.test_data_rna_fasta, is_categorical=False)

    else:
        train_labels = get_labels(args.data_set, args.training_data_rna_fasta, is_categorical=True)
        test_labels = get_labels(args.data_set, args.test_data_rna_fasta, is_categorical=True)
    
    sample_batch = next(iter(train_dataloader))[0]
    input_dim = sample_batch.size(-1)

    # MLP to reduce dimensionality
    mlp = MLP(input_dim, hidden_dim, output_dim)
    
    if args.continuous:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    models = []

    mlp.train()
    for batch_idx, batch_combined_embeddings in enumerate(train_dataloader):
        X_batch = batch_combined_embeddings.cpu().detach().numpy()
        start_idx = batch_idx * batch_size
        end_idx = start_idx + X_batch.shape[0]  
        y_batch = train_labels[start_idx:end_idx] 

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = mlp(torch.tensor(X_batch, dtype=torch.float32))
            if args.continuous:
                # Ensure y_batch is converted to torch tensor and reshaped as necessary
                y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1)
                loss = criterion(outputs, y_batch_tensor)
            else:
                # For classification, ensure y_batch is converted to torch tensor with correct dtype
                y_batch_tensor = torch.tensor(y_batch, dtype=torch.long)
                loss = criterion(outputs, y_batch_tensor)

            loss.backward()
            optimizer.step() 
        
        mlp.eval()
        with torch.no_grad():
            batch_features = mlp(torch.tensor(X_batch, dtype=torch.float32)).numpy()

        if args.continuous:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(batch_features, y_batch)
        models.append(model)
        mlp.train()

    test_dataset = EmbeddingsAndSequences(
        weights_filepath=args.Model2_test,
        rna_fasta=args.test_data_rna_fasta,
        protein_fasta=args.test_data_protein_fasta,
    )
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False) 

    test_features = []
    
    mlp.eval()
    with torch.no_grad():
        for batch_idx, batch_combined_embeddings in enumerate(test_loader):
            # Process the whole batch at once
            batch_features = mlp(batch_combined_embeddings).numpy()
            test_features.append(batch_features)
        
        X_test_reduced = np.concatenate(test_features, axis=0)
        y_test = test_labels
        predictions = np.zeros((X_test_reduced.shape[0], len(models)))

    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_test_reduced)
    
    if args.continuous:
        final_predictions = np.mean(predictions, axis=1)

        mse = mean_squared_error(y_test, final_predictions)
        mae = mean_absolute_error(y_test, final_predictions)
        r2 = r2_score(y_test, final_predictions)

        print(f'Aggregated Model MSE: {mse:.4f}')
        print(f'Aggregated Model MAE: {mae:.4f}')
        print(f'Aggregated Model R²: {r2:.4f}')

        for i, model in enumerate(models):
            individual_mse = mean_squared_error(y_test, model.predict(X_test_reduced))
            individual_mae = mean_absolute_error(y_test, model.predict(X_test_reduced))
            individual_r2 = r2_score(y_test, model.predict(X_test_reduced))
            print(f'Model {i+1} MSE: {individual_mse:.4f}')
            print(f'Model {i+1} MAE: {individual_mae:.4f}')
            print(f'Model {i+1} R²: {individual_r2:.4f}')
    
    else:
        final_predictions = np.mean(predictions, axis=1).round().astype(int)

        accuracy = accuracy_score(y_test, final_predictions)
        print(f'Aggregated Model Accuracy: {accuracy:.4f}')

        for i, model in enumerate(models):
            individual_accuracy = accuracy_score(y_test, model.predict(X_test_reduced))
            print(f'Model {i+1} Accuracy: {individual_accuracy:.4f}')

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
