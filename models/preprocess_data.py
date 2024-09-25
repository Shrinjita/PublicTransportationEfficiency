import os
import pandas as pd

def load_data(file_path):
    """Loads data from a CSV file with error handling for encoding."""
    try:
        # Attempt to read the CSV with UTF-8 encoding first
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If there's a decoding error, fall back to a different encoding
        print(f"UTF-8 encoding failed for {file_path}. Trying ISO-8859-1 encoding...")
        return pd.read_csv(file_path, encoding='ISO-8859-1')

def clean_data(df):
    """Performs basic cleaning on the dataframe by dropping missing values."""
    df_cleaned = df.dropna()  # Drop missing values
    return df_cleaned

def preprocess_multiple_files(file_paths, output_dir, interim_folder):
    """Loads, cleans, and preprocesses multiple CSV files."""
    
    # Ensure interim folder exists
    os.makedirs(interim_folder, exist_ok=True)

    for file_path in file_paths:
        try:
            # Load and clean the data
            df = load_data(file_path)
            df_cleaned = clean_data(df)

            # Extract column names for logging/debugging
            column_names = df.columns.tolist()
            print(f"Processing {os.path.basename(file_path)} with columns: {column_names}")

            # Save the cleaned file to the interim folder
            file_name = os.path.basename(file_path)
            cleaned_file_path = os.path.join(interim_folder, file_name)
            df_cleaned.to_csv(cleaned_file_path, index=False)

            print(f"Cleaned file saved to {cleaned_file_path}")

            # Save the final cleaned file to the output directory
            final_output_path = os.path.join(output_dir, f"cleaned_{file_name}")
            df_cleaned.to_csv(final_output_path, index=False)
            print(f"Processed and saved: {final_output_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# No need for the __main__ block, as the variables will be passed from the notebook.
