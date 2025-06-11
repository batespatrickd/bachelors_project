#### added filtering column density == 0 out ########


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_column_densities(use_binned_statistics=True, num_bins=10, fit_median=True, include_mean=True, include_median=True, use_log_log=True):
    # Step 1: Load the CSV files
    ray_coords = pd.read_csv('rays_coords.csv', header=None)
    column_densities = pd.read_csv('rays_column_densities.csv', header=None)

    # Step 2: Ensure both datasets are properly aligned
    min_length = min(len(ray_coords), len(column_densities))
    ray_coords = ray_coords.iloc[:min_length]
    column_densities = column_densities.iloc[:min_length]

    # Step 3: Filter out data points with column density == 0
    non_zero_mask = (column_densities[0] != 0)
    ray_coords = ray_coords[non_zero_mask]
    column_densities = column_densities[non_zero_mask]

    # Step 4: Compute 2D distances from the z-axis
    distances = np.sqrt(ray_coords[0]**2 + ray_coords[1]**2)
    
    # Step 5: Compute binned statistics if enabled
    median_densities, mean_densities, bin_centers = [], [], []
    
    if use_binned_statistics:
        if len(distances) > 0:
            bin_edges = np.logspace(np.log10(min(distances)), np.log10(max(distances)), num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            for i in range(len(bin_edges) - 1):
                mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
                if np.any(mask):
                    median_densities.append(np.median(column_densities[mask]))
                    mean_densities.append(np.mean(column_densities[mask]))
                else:
                    median_densities.append(np.nan)
                    mean_densities.append(np.nan)
        else:
            print("Warning: No data left after filtering column densities == 0")
            return

    # Choose median or mean for plotting trend
    densities_to_plot = median_densities if fit_median else mean_densities
    plot_distances = bin_centers[~np.isnan(densities_to_plot)] if use_binned_statistics and len(distances) > 0 else []
    plot_densities = np.array(densities_to_plot)[~np.isnan(densities_to_plot)] if use_binned_statistics and len(distances) > 0 else []

    # Step 6: Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(distances, column_densities, color='b', alpha=0.25, label="All Data")
    
    if use_binned_statistics and len(distances) > 0:
        if include_median:
            plt.scatter(bin_centers, median_densities, color='black', marker='s', s=80, label="Median in Bins")
        if include_mean:
            plt.scatter(bin_centers, mean_densities, color='r', marker='o', s=80, label="Mean in Bins")
    
    if use_log_log and len(distances) > 0:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.xlabel('Distance from Z-axis (kpc)')
    plt.ylabel('Column Density [cm^-2]')
    plt.title('Column Density vs. Distance' + (' (Log-Log Scale)' if use_log_log and len(distances) > 0 else ''))
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('column_density_vs_distance.png')
    plt.close()
    
    print("Plot saved as 'column_density_vs_distance.png'")

# Example usage:
plot_column_densities(use_binned_statistics=True, num_bins=10, fit_median=False, include_mean=True, include_median=False, use_log_log=True)

#need rays_coords.csv and rays_column_densities.csv files

# ##### added median / mean binning and plotting ########

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_column_densities(use_binned_statistics=True, num_bins=10, fit_median=True, include_mean=True, include_median=True, use_log_log=True):
#     # Step 1: Load the CSV files
#     ray_coords = pd.read_csv('rays_coords.csv', header=None)
#     column_densities = pd.read_csv('rays_column_densities.csv', header=None)

#     # Step 2: Compute 2D distances from the z-axis
#     distances = np.sqrt(ray_coords[0]**2 + ray_coords[1]**2)
    
#     # Step 3: Ensure both datasets are properly aligned
#     min_length = min(len(distances), len(column_densities))
#     distances = distances.iloc[:min_length]
#     column_densities = column_densities.iloc[:min_length]
    
#     # Step 4: Compute binned statistics if enabled
#     median_densities, mean_densities, bin_centers = [], [], []
    
#     if use_binned_statistics:
#         bin_edges = np.logspace(np.log10(min(distances)), np.log10(max(distances)), num_bins + 1)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
#         for i in range(len(bin_edges) - 1):
#             mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
#             if np.any(mask):
#                 median_densities.append(np.median(column_densities[mask]))
#                 mean_densities.append(np.mean(column_densities[mask]))
#             else:
#                 median_densities.append(np.nan)
#                 mean_densities.append(np.nan)
    
#     # Choose median or mean for plotting trend
#     densities_to_plot = median_densities if fit_median else mean_densities
#     plot_distances = bin_centers[~np.isnan(densities_to_plot)]
#     plot_densities = np.array(densities_to_plot)[~np.isnan(densities_to_plot)]
    
#     # Step 5: Create the plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(distances, column_densities, color='b', alpha=0.25, label="All Data")
    
#     if use_binned_statistics:
#         if include_median:
#             plt.scatter(bin_centers, median_densities, color='black', marker='s', s=80, label="Median in Bins")
#         if include_mean:
#             plt.scatter(bin_centers, mean_densities, color='r', marker='o', s=80, label="Mean in Bins")
    
#     if use_log_log:
#         plt.xscale('log')
#         plt.yscale('log')
    
#     plt.xlabel('Distance from Z-axis (kpc)')
#     plt.ylabel('Column Density [cm^-2]')
#     plt.title('Column Density vs. Distance' + (' (Log-Log Scale)' if use_log_log else ''))
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot
#     plt.tight_layout()
#     plt.savefig('column_density_vs_distance.png')
#     plt.close()
    
#     print("Plot saved as 'column_density_vs_distance.png'")

# # Example usage:
# plot_column_densities(use_binned_statistics=True, num_bins=10, fit_median=False, include_mean=True, include_median=False, use_log_log=True)


#need rays_coords.csv and rays_column_densities.csv files

#### original program below #######
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Step 1: Read the ray coordinates from the rays_coords.csv file
# ray_coords = pd.read_csv('rays_coords.csv', header=None)  

# # Step 2: Compute 2D distances from the z-axis
# distances = np.sqrt(ray_coords[0]**2 + ray_coords[1]**2)  

# # Step 3: Read the column densities from rays_column_densities.csv
# column_densities = pd.read_csv('rays_column_densities.csv', header=None)

# # Step 4: Ensure both datasets are properly aligned
# min_length = min(len(distances), len(column_densities))
# distances = distances.iloc[:min_length]
# column_densities = column_densities.iloc[:min_length]

# # Boolean for log-log plot
# use_log_log = True  # Set to True for log-log plot

# # Step 5: Plot column density vs. distance
# plt.figure(figsize=(8, 6))

# if use_log_log:
#     plt.loglog(distances, column_densities, marker='o', linestyle='None', color='b')
#     plt.xlabel('Log(Distance from Z-axis) (kpc)', fontsize=12)
#     plt.ylabel('Log(Column Density)', fontsize=12)
#     plt.title('Log-Log Plot: Column Density as a function of Distance', fontsize=14)
# else:
#     plt.plot(distances, column_densities, marker='o', linestyle='None', color='b')
#     plt.xlabel('Distance from Z-axis (kpc)', fontsize=12)
#     plt.ylabel('Column Density ${\\rm cm^{-2}}$', fontsize=12)
#     plt.title('Column Density as a function of Distance', fontsize=14)

# plt.grid(True)
# plt.savefig('column_density_distance.png')  # Save as PNG file

# plt.show()
