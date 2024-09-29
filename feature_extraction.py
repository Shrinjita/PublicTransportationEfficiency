import pandas as pd
import numpy as np

# Load dataset
csv_files = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data/processed/cleaned_New York City Bus Data1.csv"
data = pd.read_csv(csv_files)

# Feature extraction: Selecting key columns
selected_columns = [
    'car_id', 'manufacturer', 'model', 'transmission', 'transmission_type', 
    'engine_size_cm3', 'fuel', 'powertrain', 'power_ps', 'co2_emissions_gPERkm'
]

# Subset the data
data_subset = data[selected_columns]

# Feature engineering

# Create new feature: emission intensity (per PS unit)
data_subset['emission_per_ps'] = data_subset['co2_emissions_gPERkm'] / data_subset['power_ps']

# Create new feature: engine size in liters
data_subset['engine_size_liters'] = data_subset['engine_size_cm3'] / 1000

# One-hot encode the categorical variables (manufacturer, transmission, fuel, etc.)
data_encoded = pd.get_dummies(data_subset, columns=['manufacturer', 'transmission', 'fuel', 'powertrain'])

# Handle missing values (optional depending on your dataset)
data_encoded.fillna(data_encoded.mean(), inplace=True)

# Save the preprocessed data
data_encoded.to_csv('vehicles.csv', index=False)

print("Feature extraction and engineering complete!")
