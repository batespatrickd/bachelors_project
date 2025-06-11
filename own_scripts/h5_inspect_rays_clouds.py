import h5py
import pandas as pd
import numpy as np

# Function to save datasets as CSV
def save_dataset_to_csv(name, obj, file_prefix):
    if isinstance(obj, h5py.Dataset):  # Check if it is a dataset
        data = obj[()]  # Read the dataset
        if isinstance(data, np.ndarray):  # Ensure it's an array
            df = pd.DataFrame(data)  # Convert to DataFrame
            csv_filename = f"../own_scripts/{file_prefix}_{name.replace('/', '_')}.csv"  # Replace '/' in names for valid filenames
            df.to_csv(csv_filename, index=False, header=False)
            print(f"Saved {name} to {csv_filename}")

# List and save datasets
def process_file(file_path, file_prefix):
    with h5py.File(file_path, "r") as f:
        # Save datasets to CSV
        f.visititems(lambda name, obj: save_dataset_to_csv(name, obj, file_prefix))

        # List all groups and datasets (optional, for debugging or inspection)
        def print_structure(name, obj):
            print(name, ":", obj)

        f.visititems(print_structure)  # Recursively lists all groups and datasets

# Process both files
process_file("../src/clouds.h5", "clouds")
process_file("../src/rays.h5", "rays")
