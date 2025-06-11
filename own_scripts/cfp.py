# Script debugged and modified with the help of OpenAI. (2025). ChatGPT (June 11 version) [Large language model]. https://chat.openai.com/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_hit_to_missed_ratio(
    max_distance,
    num_increments,
    total_rays,
    use_log_scale=True,
    save_plot=True,
    plot_filename='hit_to_missed_ratio_plot.png',
    min_rays_per_bin=5  # Minimum rays required in a bin to compute ratio
):
    # Load data
    rays_n_clouds = pd.read_csv('rays_n_clouds.csv', header=None)
    rays_coords = pd.read_csv('rays_coords.csv', header=None)

    # Compute 2D distance from z-axis
    distances = np.sqrt(rays_coords[0]**2 + rays_coords[1]**2)

    # Mask for rays that hit
    rays_hit_mask = rays_n_clouds[0] > 0

    # Distance bins
    if use_log_scale:
        increments = np.logspace(0, np.log10(max_distance), num=num_increments)
    else:
        increments = np.linspace(0, max_distance, num=num_increments)

    hit_to_missed_ratios = []
    tick_labels = []

    for i in range(len(increments) - 1):
        lower_bound = increments[i]
        upper_bound = increments[i + 1]

        # Include lower bound for the first bin only
        if i == 0:
            rays_within_d = (distances >= lower_bound) & (distances <= upper_bound)
        else:
            rays_within_d = (distances > lower_bound) & (distances <= upper_bound)

        rays_hit_within_d = rays_within_d & rays_hit_mask
        rays_missed_within_d = rays_within_d & ~rays_hit_mask

        total_rays_in_bin = np.sum(rays_hit_within_d) + np.sum(rays_missed_within_d)

        if total_rays_in_bin < min_rays_per_bin:
            print(f"Skipping bin ({lower_bound:.2f}, {upper_bound:.2f}]: too few rays ({total_rays_in_bin})")
            continue  # Skip bins with too few rays

        if np.sum(rays_missed_within_d) > 0:
            ratio = np.sum(rays_hit_within_d) / np.sum(rays_missed_within_d)
        else:
            ratio = np.inf  # All hit, no misses

        hit_to_missed_ratios.append(ratio)
        tick_labels.append(f"({lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"Distance: ({lower_bound:.2f}, {upper_bound:.2f}] kpc, Hit/Missed Ratio: {ratio:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(hit_to_missed_ratios)), hit_to_missed_ratios, marker='o', linestyle='-', color='b')

    plt.xticks(range(len(tick_labels)), tick_labels, rotation=60, ha="right", fontsize=10)
    plt.xlabel("Distance Interval [kpc]", fontsize=12)
    plt.ylabel("Hit Rays / Missed Rays Ratio", fontsize=12)
    plt.title("Hit to Missed Rays Ratio vs Distance", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved as {plot_filename}")

    plt.show()

# Example usage
compute_hit_to_missed_ratio(
    max_distance=50,
    num_increments=9,
    total_rays=10000,
    use_log_scale=False,
    save_plot=True
)
# need rays_n_clouds.csv and rays_coords.csv files
########## problems with first bin, zero at first bin and then drastic increase on second bin ##################

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def compute_hit_to_missed_ratio(max_distance, num_increments, total_rays, use_log_scale=True, save_plot=True, plot_filename='hit_to_missed_ratio_plot.png'):
#     # Load data
#     rays_n_clouds = pd.read_csv('rays_n_clouds.csv', header=None)  # Number of clouds each ray passed through
#     rays_coords = pd.read_csv('rays_coords.csv', header=None)  # Coordinates of rays
    
#     # Compute 2D distance from z-axis using rays_coords (x, y columns)
#     distances = np.sqrt(rays_coords[0]**2 + rays_coords[1]**2)
    
#     # Determine rays that hit (non-zero clouds) and rays that missed (zero clouds)
#     rays_hit_mask = rays_n_clouds[0] > 0  # Rays that hit (non-zero cloud count)
#     rays_hit = np.sum(rays_hit_mask)  # Rays that hit
#     rays_missed = total_rays - rays_hit  # Rays that missed
    
#     # Define distance bins
#     if use_log_scale:
#         increments = np.logspace(0, np.log10(max_distance), num=num_increments)
#     else:
#         increments = np.linspace(0, max_distance, num=num_increments)
    
#     hit_to_missed_ratios = []
    
#     for i in range(len(increments) - 1):
#         # Find the rays within this distance increment
#         lower_bound = increments[i]
#         upper_bound = increments[i + 1]
#         rays_within_d = (distances > lower_bound) & (distances <= upper_bound)
        
#         # Separate rays that hit vs missed within this distance increment
#         rays_hit_within_d = rays_within_d & rays_hit_mask
#         rays_missed_within_d = rays_within_d & ~rays_hit_mask
        
#         # Calculate ratio of hit rays to missed rays in this distance bin
#         if np.sum(rays_missed_within_d) > 0:  # Avoid division by zero
#             ratio = np.sum(rays_hit_within_d) / np.sum(rays_missed_within_d)
#         else:
#             ratio = 0  # No missed rays within this distance range
        
#         hit_to_missed_ratios.append(ratio)
#         print(f"Distance: ({lower_bound:.2f}, {upper_bound:.2f}] kpc, Hit/Missed Ratio: {ratio:.4f}")
    
#     # Plot results as a line plot
#     plt.figure(figsize=(10, 6))  # Adjust figure size for better spacing
#     plt.plot(range(len(hit_to_missed_ratios)), hit_to_missed_ratios, marker='o', linestyle='-', color='b')

#     # Set x-axis ticks to correspond to distance bin ranges
#     tick_labels = [f"({increments[i]:.2f}, {increments[i+1]:.2f}]" for i in range(len(hit_to_missed_ratios))]
#     plt.xticks(range(len(hit_to_missed_ratios)), tick_labels, rotation=60, ha="right", fontsize=10)  # Rotate and resize labels

#     plt.xlabel("Distance Interval [kpc]", fontsize=12)
#     plt.ylabel("Hit Rays / Missed Rays Ratio", fontsize=12)
#     plt.title("Hit to Missed Rays Ratio vs Distance", fontsize=14)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)

#     # Adjust layout to prevent label clipping
#     plt.tight_layout()

#     # Save plot if save_plot is True
#     if save_plot:
#         plt.savefig(plot_filename, dpi=300)
#         print(f"Plot saved as {plot_filename}")

#     # Show the plot
#     plt.show()

# # Example usage
# compute_hit_to_missed_ratio(max_distance=50, num_increments=20, total_rays=10000, use_log_scale=False, save_plot=True)



#### bar graph #######
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def compute_hit_to_missed_ratio(max_distance, num_increments, total_rays, use_log_scale=True, save_plot=False, plot_filename='hit_to_missed_ratio_plot.png'):
#     # Load data
#     rays_n_clouds = pd.read_csv('rays_n_clouds.csv', header=None)  # Number of clouds each ray passed through
#     rays_coords = pd.read_csv('rays_coords.csv', header=None)  # Coordinates of rays
    
#     # Compute 2D distance from z-axis using rays_coords (x, y columns)
#     distances = np.sqrt(rays_coords[0]**2 + rays_coords[1]**2)
    
#     # Determine rays that hit (non-zero clouds) and rays that missed (zero clouds)
#     rays_hit_mask = rays_n_clouds[0] > 0  # Rays that hit (non-zero cloud count)
#     rays_hit = np.sum(rays_hit_mask)  # Rays that hit
#     rays_missed = total_rays - rays_hit  # Rays that missed
    
#     # Define distance bins
#     if use_log_scale:
#         increments = np.logspace(0, np.log10(max_distance), num=num_increments)
#     else:
#         increments = np.linspace(0, max_distance, num=num_increments)
    
#     hit_to_missed_ratios = []
    
#     for i in range(len(increments) - 1):
#         # Find the rays within this distance increment
#         lower_bound = increments[i]
#         upper_bound = increments[i + 1]
#         rays_within_d = (distances > lower_bound) & (distances <= upper_bound)
        
#         # Separate rays that hit vs missed within this distance increment
#         rays_hit_within_d = rays_within_d & rays_hit_mask
#         rays_missed_within_d = rays_within_d & ~rays_hit_mask
        
#         # Calculate ratio of hit rays to missed rays in this distance bin
#         if np.sum(rays_missed_within_d) > 0:  # Avoid division by zero
#             ratio = np.sum(rays_hit_within_d) / np.sum(rays_missed_within_d)
#         else:
#             ratio = 0  # No missed rays within this distance range
        
#         hit_to_missed_ratios.append(ratio)
#         print(f"Distance: ({lower_bound:.2f}, {upper_bound:.2f}] kpc, Hit/Missed Ratio: {ratio:.4f}")
    
#     # Plot results as a bar graph
#     plt.figure(figsize=(10, 6))  # Adjust figure size for better spacing
#     plt.bar(range(len(hit_to_missed_ratios)), hit_to_missed_ratios, width=0.7, color='g', align='center')

#     # Set x-axis ticks to correspond to distance bin ranges
#     tick_labels = [f"({increments[i]:.2f}, {increments[i+1]:.2f}]" for i in range(len(hit_to_missed_ratios))]
#     plt.xticks(range(len(hit_to_missed_ratios)), tick_labels, rotation=60, ha="right", fontsize=10)  # Rotate and resize labels

#     plt.xlabel("Distance Bins [kpc]", fontsize=12)
#     plt.ylabel("Hit Rays / Missed Rays Ratio", fontsize=12)
#     plt.title("Hit to Missed Rays Ratio vs Distance Bins", fontsize=14)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)

#     # Adjust layout to prevent label clipping
#     plt.tight_layout()

#     # Save plot if save_plot is True
#     if save_plot:
#         plt.savefig(plot_filename, dpi=300)
#         print(f"Plot saved as {plot_filename}")

#     # Show the plot
#     plt.show()

# # Example usage
# compute_hit_to_missed_ratio(max_distance=5, num_increments=10, total_rays=10000, use_log_scale=False, save_plot=True)
