import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from orion_butterfly import load_model  # Ensure you have this function defined

# Check if Intel GPU is available and set device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Training on Intel GPU")
else:
    device = torch.device('cpu')
    print("Training on CPU")

# Custom Dataset Class
class VehicleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Load preprocessed data from CSV files
csv_files = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data/processed"
file_path = os.path.join(csv_files, 'vehicles_cleaned.csv')

# Read the dataset
data = pd.read_csv(file_path)

# Check the first few rows of the data
print(data.head())

# Features and target (CO2 emissions)
X = data.drop(columns=['co2_emissions_gPERkm'])
y = data['co2_emissions_gPERkm']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch Dataset
train_dataset = VehicleDataset(X_train_scaled, y_train.values)
test_dataset = VehicleDataset(X_test_scaled, y_test.values)

# DataLoader
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

# Instantiate the model and move it to the selected device
input_size = X_train.shape[1]
model = CO2EmissionModel(input_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print loss per epoch
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Make prediction function
def make_prediction(input_data):
    model = load_model()  # Load your trained model
    model.eval()
    
    # Preprocess input data as required
    # Assuming input_data is preprocessed here to match model input
    tensor_input = torch.tensor(input_data, dtype=torch.float32)  # Adjust based on your input
    
    with torch.no_grad():
        prediction = model(tensor_input.unsqueeze(0))  # Adjust based on your model's expected input shape
    
    return prediction.item()  # Return prediction result

# Train the model
train_model(model, train_loader, criterion, optimizer, device, epochs=20)

# Save the trained model
model_path = os.path.join(csv_files, 'co2_emission_model.pth')
torch.save(model.state_dict(), model_path)

print(f"Model training complete and saved to {model_path}!")
