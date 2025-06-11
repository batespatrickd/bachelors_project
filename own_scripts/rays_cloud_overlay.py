# Script debugged and modified with the help of OpenAI. (2025). ChatGPT (June 11 version) [Large language model]. https://chat.openai.com/


import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.collections import PatchCollection
from tqdm import tqdm

# Constants for unit conversions
kpc = 3.086e21  # in cm
cm = 1e-2  # cm for conversion

# Function to load clouds from an HDF5 file
def load_clouds_from_h5(clouds_file):
    with h5py.File(clouds_file, 'r') as f:
        centers = np.array(f['centers'])  # Assuming centers are stored in 'centers'
        radii = np.array(f['radii'])  # Assuming radii are stored in 'radii'
        velocities = np.array(f['velocities'])  # Assuming velocities are stored in 'velocities'
        masses = np.array(f['masses'])  # Assuming masses are stored in 'masses'
    return centers, radii, velocities, masses

# Function to load rays from an HDF5 file
def load_rays_from_h5(rays_file):
    with h5py.File(rays_file, 'r') as f:
        coords = np.array(f['coords'])  # Assuming ray coordinates are stored in 'coords'
        n_clouds = np.array(f['n_clouds'])  # Assuming ray-cloud hit/miss data is stored in 'n_clouds'
    return coords, n_clouds

# Function to plot clouds and rays
def plot_clouds(rays_coords=None, rays_n_clouds=None, clouds_centers=None, clouds_radii=None, number_labels=False, axis_labels=True, filename=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect(1)

    # Axis labels
    if axis_labels:
        ax.set_xlabel('x (kpc)')
        ax.set_ylabel('y (kpc)')

    # Plot clouds
    if clouds_centers is not None and clouds_radii is not None:
        print("Plotting Clouds")
        n_clouds = len(clouds_radii)
        for i in tqdm(range(n_clouds), "Plotting Clouds"):
            circle = plt.Circle((clouds_centers[i, 0], clouds_centers[i, 1]), radius=clouds_radii[i], alpha=0.5, color='blue')
            ax.add_artist(circle)

    # Plot rays
    if rays_coords is not None and rays_n_clouds is not None:
        print("Plotting Rays")
        n_rays = len(rays_coords)
        ray_hits = [plt.Circle((rays_coords[i, 0], rays_coords[i, 1]), radius=0.02) 
                    for i in range(n_rays) if rays_n_clouds[i] > 0]
        collection_hits = PatchCollection(ray_hits)
        collection_hits.set_alpha(0.7)
        collection_hits.set_color('black')  # Black color for hit rays
        ax.add_collection(collection_hits)

        ray_misses = [plt.Circle((rays_coords[i, 0], rays_coords[i, 1]), radius=0.02) 
                      for i in range(n_rays) if rays_n_clouds[i] == 0]
        collection_misses = PatchCollection(ray_misses)
        collection_misses.set_alpha(0.3)
        collection_misses.set_color('gray')  # Light gray for missed rays
        ax.add_collection(collection_misses)

    # Save the plot
    if filename is None:
        filename = 'clouds_with_rays.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close("all")

# Main function for loading clouds, rays, and plotting
def main():
    # Load clouds and rays from the .h5 files
    clouds_file = '../src/clouds.h5'  # Replace with your actual clouds file
    rays_file = '../src/rays.h5'  # Replace with your actual rays file
    
    # Load data from .h5 files
    clouds_centers, clouds_radii, _, _ = load_clouds_from_h5(clouds_file)
    rays_coords, rays_n_clouds = load_rays_from_h5(rays_file)

    # Plot and save the image
    plot_clouds(rays_coords=rays_coords, rays_n_clouds=rays_n_clouds, 
                clouds_centers=clouds_centers, clouds_radii=clouds_radii, 
                number_labels=True, filename='clouds_and_rays_overlay.png')

if __name__ == "__main__":
    main()
