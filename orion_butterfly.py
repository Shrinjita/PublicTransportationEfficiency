import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from orion.client import report, Client

# Define file paths
csv_files = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data/processed"
processed_data_path = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data"

# Load preprocessed data
data = pd.read_csv(f"{processed_data_path}/public_transportation_data.csv")

# Features and target (CO2 emissions)
X = data.drop(columns=['co2_emissions_gPERkm'])
y = data['co2_emissions_gPERkm']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom Dataset Class
class VehicleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# DataLoader
train_dataset = VehicleDataset(X_train_scaled, y_train.values)
test_dataset = VehicleDataset(X_test_scaled, y_test.values)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define Neural Network Model
class CO2EmissionModel(nn.Module):
    def __init__(self, input_size):
        super(CO2EmissionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
input_size = X_train.shape[1]
model = CO2EmissionModel(input_size)

# Loss and optimizer (parameters will be tuned via Orion)
criterion = nn.MSELoss()

# Butterfly Optimization Algorithm (BOA)
class ButterflyOptimization:
    def __init__(self, population_size, dimensions, bounds, iterations):
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.iterations = iterations
        self.butterflies = np.random.uniform(bounds[0], bounds[1], (population_size, dimensions))
        self.best_position = None
        self.best_fitness = float('inf')

    def fitness_function(self, position):
        # This will be the validation loss of the model
        learning_rate, batch_size = position[0], int(position[1])
        validation_loss = train_and_evaluate({'learning_rate': learning_rate, 'batch_size': batch_size})
        return validation_loss

    def optimize(self):
        for _ in range(self.iterations):
            for i in range(self.population_size):
                fitness = self.fitness_function(self.butterflies[i])
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = self.butterflies[i]
            self.update_positions()

    def update_positions(self):
        # Update butterfly positions based on their fitness
        for i in range(self.population_size):
            # Placeholder for position update logic
            self.butterflies[i] += np.random.randn(self.dimensions) * 0.1

        # Keep the positions within bounds
        self.butterflies = np.clip(self.butterflies, self.bounds[0], self.bounds[1])

# Training and evaluation function used for both BOA and Orion
def train_and_evaluate(params):
    # Set up optimizer with the given learning rate
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop
    model.train()
    for epoch in range(10):  # Adjust the number of epochs as needed
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

    # Validation step
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            validation_loss += loss.item()

    validation_loss /= len(test_loader)
    return validation_loss

# Orion configuration
client = Client()
client.create_experiment(
    name="Hyperparameter Tuning with Orion",
    space={
        'learning_rate': {'type': 'uniform', 'low': 0.0001, 'high': 0.1},
        'batch_size': {'type': 'choice', 'options': [16, 32, 64, 128]}
    },
    objective_metric='validation_loss',  # Metric to optimize
    objective='minimize',  # Minimize the objective metric
    max_trials=50  # Number of trials to run
)

# Run optimization using Orion
for trial in client.fetch_trials():
    params = trial.params
    validation_loss = train_and_evaluate(params)
    report(trial, validation_loss)

# Run BOA optimization
boa = ButterflyOptimization(population_size=10, dimensions=2, bounds=[(0.0001, 0.1), (16, 128)], iterations=20)
boa.optimize()
print(f"Best BOA hyperparameters: Learning rate: {boa.best_position[0]}, Batch size: {int(boa.best_position[1])}")
