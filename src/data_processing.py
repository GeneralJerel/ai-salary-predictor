"""
Data Processing Module
Loads, cleans, and validates the DS salaries dataset.
"""

import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"Loaded data shape: {df.shape}")
    return df

def clean_data(df):
    """
    Clean the dataset:
    - Drop the unnamed index column
    - Remove duplicates
    - Handle missing values
    """
    # Drop unnamed index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    print(f"Shape after dropping index column: {df.shape}")

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Duplicates removed: {duplicates_removed}")

    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values found:")
        print(missing_counts[missing_counts > 0])
        # Drop rows with missing values if any
        df = df.dropna()
        print(f"Shape after removing missing values: {df.shape}")
    else:
        print("No missing values found")

    # Data type validation
    print("\nData types:")
    print(df.dtypes)

    print("\nBasic statistics:")
    print(df.describe())

    return df

def save_processed_data(df, output_dir='data/processed'):
    """Save processed data to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ds_salaries_clean.csv')
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    return output_path

def main(input_path='data/raw/ds_salaries.csv', output_dir='data/processed'):
    """Main processing pipeline."""
    df = load_data(input_path)
    df = clean_data(df)
    output_path = save_processed_data(df, output_dir)
    return df

if __name__ == '__main__':
    main()
