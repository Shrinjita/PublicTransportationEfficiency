import os
import sys
import pandas as pd
from orion import Orion  # Importing the Orion module for route optimization

# Append the path where your preprocess_data.py is located
script_path = r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiency\models"
sys.path.append(script_path)

# Import the preprocess_data module
import preprocess_data

# Define the list of file paths
file_paths = [
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\New York City Bus Data\uk_gov_data_dense_preproc.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\New York City Bus Data\uk_gov_data_sparse_preproc.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\NYC Yellow Taxi Trip Data\yellow_tripdata_2016-03.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\NYC Yellow Taxi Trip Data\yellow_tripdata_2016-02.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\NYC Yellow Taxi Trip Data\yellow_tripdata_2016-01.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\NYC Yellow Taxi Trip Data\yellow_tripdata_2015-01.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\Vehicle Emissions Data Set\mta_1706.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\Vehicle Emissions Data Set\mta_1708.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\Vehicle Emissions Data Set\mta_1710.csv",
    r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\raw\Vehicle Emissions Data Set\mta_1712.csv"
]

# Define the output directory where cleaned files will be saved
output_dir = r"C:\Users\Shrinjita Paul\Documents\GitHub\PublicTransportationEfficiencyData\data\processed"

# Preprocess all files
preprocess_data.preprocess_multiple_files(file_paths, output_dir)

# Route Optimization using ORION
def optimize_routes(data):
    """Optimize routes using ORION."""
    orion = Orion()

    # Assuming data has latitude and longitude columns for stops
    # Replace 'latitude' and 'longitude' with your actual column names
    if 'latitude' in data.columns and 'longitude' in data.columns:
        routes = orion.optimize(data[['latitude', 'longitude']])
        return routes
    else:
        print("Data must contain 'latitude' and 'longitude' columns for optimization.")
        return None

if __name__ == "__main__":
    print("Starting the preprocessing...")

    # Assuming you want to read cleaned data from the interim folder
    cleaned_data_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv')]
    
    # Collecting all cleaned data into a single DataFrame for route optimization
    all_data = pd.concat((pd.read_csv(file) for file in cleaned_data_paths), ignore_index=True)

    # Optimize routes based on the cleaned data
    optimized_routes = optimize_routes(all_data)

    if optimized_routes is not None:
        # Save optimized routes to a file
        optimized_routes_file = os.path.join(output_dir, "optimized_routes.csv")
        optimized_routes.to_csv(optimized_routes_file, index=False)
        print(f"Optimized routes saved to {optimized_routes_file}")
