import h5py
import csv
import numpy as np
from itertools import zip_longest

def extract_h5_to_csv(h5_filename, csv_filename):
    """
    Extracts all datasets (ray_numbers, tau_data, x_data) from an HDF5 file
    and writes them into a CSV file.
    """
    with h5py.File(h5_filename, 'r') as h5_file:
        # Read datasets
        ray_numbers = h5_file['ray_numbers'][:]
        tau_data = h5_file['tau_data'][:]
        x_data = h5_file['x_data'][:]

        # Ensure all lists are the same length using zip_longest
        combined_data = zip_longest(ray_numbers, tau_data, x_data, fillvalue=np.nan)

        # Write data to CSV
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Ray Number', 'Tau Value', 'X Value'])  # Header
            writer.writerows(combined_data)

# Example usage
extract_h5_to_csv('../src/all_tau_data.h5', 'tau_data.csv')
