import os
from preprocess_data import preprocess_file

# Define base path where CSV files are stored and where processed files should be saved
raw_data_path = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data/raw"
processed_data_path = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data/processed"

# Create the processed data directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

# Get list of all CSV files in the raw data directory
csv_files = [file for file in os.listdir(raw_data_path) if file.endswith('.csv')]

# Debug: List CSV files found
print(f"Found CSV files: {csv_files}")

# Iterate over each CSV file and call preprocess_file
for csv_file in csv_files:
    csv_file_path = os.path.join(raw_data_path, csv_file)
    print(f"Processing file: {csv_file_path}")
    
    # Define the output path for the processed file
    output_file_path = os.path.join(processed_data_path, f"cleaned_{csv_file}")

    # Process the file
    try:
        preprocess_file(csv_file_path, output_file_path)
    except Exception as e:
        print(f"Error while processing {csv_file}: {e}")
