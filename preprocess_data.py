import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path, chunksize=10000):
    """Loads data from a CSV file in chunks with error handling for encoding."""
    try:
        # Read in chunks
        for chunk in pd.read_csv(file_path, encoding='utf-8', chunksize=chunksize):
            yield chunk
    except UnicodeDecodeError:
        print(f"UTF-8 encoding failed for {file_path}. Trying ISO-8859-1 encoding...")
        for chunk in pd.read_csv(file_path, encoding='ISO-8859-1', chunksize=chunksize):
            yield chunk

def clean_data(df, method='interpolate'):
    """Performs basic cleaning on the dataframe by applying selected methods."""
    if method == 'interpolate':
        print("Interpolating missing values...")  # Debug statement
        df_cleaned = df.interpolate(method='linear', limit_direction='forward', axis=0)
    else:
        print("Unknown cleaning method. Defaulting to interpolation.")
        df_cleaned = df.interpolate(method='linear', limit_direction='forward', axis=0)
    
    return df_cleaned

def normalize_data(df):
    """Normalizes the numerical features in the dataframe using Min-Max scaling."""
    print("Normalizing data...")  # Debug statement
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) > 0:  # Ensure there are numerical columns to normalize
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        print("No numerical columns to normalize.")  # Debug statement
    return df

def preprocess_file(input_file, output_file):
    """Preprocess a single file."""
    # Initialize an empty DataFrame for concatenation
    df_list = []
    
    # Load the CSV file in chunks
    for df_chunk in load_data(input_file):
        # Preprocess the data: Clean and normalize
        df_cleaned = clean_data(df_chunk, method='interpolate')
        df_normalized = normalize_data(df_cleaned)
        df_list.append(df_normalized)  # Collect each processed chunk

    # Concatenate all chunks into a single DataFrame and save
    full_df = pd.concat(df_list, ignore_index=True)
    full_df.to_csv(output_file, index=False)
    print(f"Preprocessed file saved at: {output_file}")

def main():
    # Ensure enough arguments are provided (input file and output file)
    if len(sys.argv) != 3:
        print("Usage: python preprocess_data.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Preprocess the single file
    preprocess_file(input_file, output_file)

if __name__ == "__main__":
    main()
