
# Script debugged and modified with the help of OpenAI. (2025). ChatGPT (June 11 version) [Large language model]. https://chat.openai.com/

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

def get_zero_rows_from_rays_n_clouds(file_path):
    df = pd.read_csv(file_path, header=None)
    zero_rows = df[df.iloc[:, 0].isin([0, 0.0])].index + 1
    return zero_rows

def calculate_distances_for_zero_rows(ray_coords_file, zero_rows):
    df = pd.read_csv(ray_coords_file, header=None)
    distances = []

    for row_num in zero_rows:
        x = df.iloc[row_num - 1, 0]
        y = df.iloc[row_num - 1, 1]
        distance = math.sqrt(x**2 + y**2)
        distances.append([row_num, distance])

    return distances

def save_distances_to_csv(distances, output_file):
    df = pd.DataFrame(distances, columns=['ray number', 'distance'])
    df.to_csv(output_file, index=False)

def plot_distance_histogram(input_file, output_image, bins=20):
    df = pd.read_csv(input_file)
    distances = df['distance']

    plt.figure(figsize=(8, 6))
    counts, bin_edges, _ = plt.hist(distances, bins=bins, color='blue', alpha=0.7, edgecolor='black')

    for count, bin_edge in zip(counts, bin_edges[:-1]):
        if count > 0:
            plt.text(bin_edge + (bin_edges[1] - bin_edges[0]) / 2, count, str(int(count)),
                     ha='center', va='bottom', fontsize=10, color='black')

    plt.xlabel('Distance from Center [kpc]')
    plt.ylabel('Frequency')
    plt.title('Histogram of Missed Ray Distances')

    plt.savefig(output_image, dpi=300)
    plt.show()
    print(f"Histogram saved as {output_image}")

def main():
    rays_n_clouds_file = 'rays_n_clouds.csv'
    ray_coords_file = 'rays_coords.csv'
    output_csv = 'missed_distances.csv'
    output_hist = 'missed_distances_histogram.png'

    print("Processing missed rays...")
    zero_rows = get_zero_rows_from_rays_n_clouds(rays_n_clouds_file)
    distances = calculate_distances_for_zero_rows(ray_coords_file, zero_rows)
    save_distances_to_csv(distances, output_csv)
    print(f"Processed data saved to {output_csv}")

    print("Generating histogram...")
    plot_distance_histogram(output_csv, output_hist, bins=20)

if __name__ == "__main__":
    main()
