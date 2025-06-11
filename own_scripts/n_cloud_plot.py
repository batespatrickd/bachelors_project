# Script debugged and modified with the help of OpenAI. (2025). ChatGPT (June 11 version) [Large language model]. https://chat.openai.com/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_cloud_intersections(csv_file):
    # Step 1: Read the CSV file
    data = pd.read_csv(csv_file, header=None)  # No header, single-column data

    # Step 2: Convert to integer (assuming all values are valid floats representing whole numbers)
    cloud_counts = data[0].astype(int)

    # Step 3: Count occurrences of each unique number of clouds passed through
    unique_counts = cloud_counts.value_counts().sort_index()

    # Step 4: Print results
    print("Number of rays per cloud count:")
    for num_clouds, count in unique_counts.items():
        print(f"Rays passing through {num_clouds} clouds: {count}")

    # Step 5: Plot histogram
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_counts.index, unique_counts.values, color='skyblue', edgecolor='black')

    # Add labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{height}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel("Number of Clouds Passed Through")
    plt.ylabel("Number of Rays")
    plt.title("Distribution of Rays Passing Through Clouds")
    plt.xticks(np.arange(0, unique_counts.index.max() + 1, step=1))  # Ensure integer x-axis ticks
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig("clouds_histogram.png")
    plt.show()

# Example usage:
analyze_cloud_intersections("rays_n_clouds.csv")
