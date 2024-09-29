import subprocess
import os

# Define the path to the analysis script
analysis_script_path = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/analysis.py"
csv_files = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data/processed"
processed_data_path = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data"
# Iterate over each processed CSV file and call analysis.py
for csv_file in csv_files:
    processed_file_path = os.path.join(processed_data_path, f"cleaned_{csv_file}")
    
    # Debug: Display the file being analyzed
    print(f"Running analysis on: {processed_file_path}")
    
    # Call the analysis script with the processed file as an argument
    try:
        result = subprocess.run(
            ['python', analysis_script_path, processed_file_path],
            check=True, capture_output=True, text=True
        )
        print(f"Analysis completed for {processed_file_path}")
        print(result.stdout)  # Debug: Display output from the analysis script
    except subprocess.CalledProcessError as e:
        print(f"Error while analyzing {processed_file_path}: {e.stderr}")
