import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_equivalent_widths(use_binned_statistics=True, num_bins=10, fit_median=True, include_mean=True, include_median=True, include_fit=True):
    # Step 1: Load the CSV files
    ews = pd.read_csv('rays_EWs.csv', header=None)  # No header specified, single column of EWs
    ray_coords = pd.read_csv('rays_coords.csv', header=None)  # No header specified, first column = x, second column = y
    
    # Step 2: Compute 2D distances from the z-axis
    distances = np.sqrt(ray_coords[0]**2 + ray_coords[1]**2)  # sqrt(x^2 + y^2)
    
    # Step 3: Filter out zero EWs for statistics calculations
    non_zero_mask = ews[0] > 0.0  # Only keep rays with EW > 0
    ews_non_zero = ews[non_zero_mask]
    distances_non_zero = distances[non_zero_mask]
    
    # Step 4: Compute binned statistics if enabled
    median_ews, mean_ews, bin_centers = [], [], []
    
    if use_binned_statistics:
        bin_edges = np.logspace(np.log10(min(distances_non_zero)), np.log10(max(distances_non_zero)), num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers

        for i in range(len(bin_edges) - 1):
            mask = (distances_non_zero >= bin_edges[i]) & (distances_non_zero < bin_edges[i+1])
            if np.any(mask):
                median_ews.append(np.median(ews_non_zero[mask]))
                mean_ews.append(np.mean(ews_non_zero[mask]))
            else:
                median_ews.append(np.nan)
                mean_ews.append(np.nan)
    
    # Choose whether to plot the median or mean points for fitting
    ews_to_plot = median_ews if fit_median else mean_ews
    fit_data = bin_centers[~np.isnan(ews_to_plot)]  # Remove NaN values
    fit_ews = np.array(ews_to_plot)[~np.isnan(ews_to_plot)]  # Remove NaN values
    
    # Step 5: Compute the fixed fit line
    fixed_fit_ews = np.exp(0.27 - 0.015 * fit_data)
    
    # Step 6: Create the log-log scale plot
    plt.figure(figsize=(6, 9))

    plt.scatter(distances_non_zero, ews_non_zero[0], color='b', alpha=0.25, label="All Data")
    
    if use_binned_statistics:
        if include_median:
            plt.scatter(bin_centers, median_ews, color='black', marker='s', s=80, label="Median in Bins")
        if include_mean:
            plt.scatter(bin_centers, mean_ews, color='r', marker='o', s=80, label="Mean in Bins")
    
    # Plot the fixed fit if enabled
    if include_fit:
        plt.plot(fit_data, fixed_fit_ews, color='green', label='Fixed Fit: log(EW) = 0.27 - 0.015*distance')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance from Center [kpc]')
    plt.ylabel('Equivalent Width (EW) [Å]')
    plt.title('Equivalent Widths vs. Distance (Log-Log Scale)')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig('equivalent_widths_vs_distance_loglog_fixed_fit.png')  # Save as PNG file
    plt.close()  # Close the plot to free up memory

    print("Log-log plot with fixed fit saved as 'equivalent_widths_vs_distance_loglog_fixed_fit.png'")

# Example usage:
plot_equivalent_widths(use_binned_statistics=True, num_bins=10, fit_median=False, include_mean=True, include_median=True, include_fit=False)



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_equivalent_widths(use_binned_statistics=True, num_bins=10, fit_median=True):
#     # Step 1: Load the CSV files
#     ews = pd.read_csv('rays_EWs.csv', header=None)  # No header specified, single column of EWs
#     ray_coords = pd.read_csv('rays_coords.csv', header=None)  # No header specified, first column = x, second column = y
    
#     # Step 2: Compute 2D distances from the z-axis
#     distances = np.sqrt(ray_coords[0]**2 + ray_coords[1]**2)  # sqrt(x^2 + y^2)
    
#     # Step 3: Filter out zero EWs for statistics calculations
#     non_zero_mask = ews[0] > 0.0  # Only keep rays with EW > 0
#     ews_non_zero = ews[non_zero_mask]
#     distances_non_zero = distances[non_zero_mask]
    
#     # Step 4: Compute binned statistics if enabled
#     if use_binned_statistics:
#         bin_edges = np.logspace(np.log10(min(distances_non_zero)), np.log10(max(distances_non_zero)), num_bins + 1)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers

#         median_ews = []
#         mean_ews = []
        
#         for i in range(len(bin_edges) - 1):
#             mask = (distances_non_zero >= bin_edges[i]) & (distances_non_zero < bin_edges[i+1])
#             if np.any(mask):
#                 median_ews.append(np.median(ews_non_zero[mask]))
#                 mean_ews.append(np.mean(ews_non_zero[mask]))
#             else:
#                 median_ews.append(np.nan)
#                 mean_ews.append(np.nan)
    
#     # Choose whether to plot the median or mean points
#     ews_to_plot = median_ews if fit_median else mean_ews
#     fit_data = bin_centers[~np.isnan(ews_to_plot)]  # Remove NaN values
#     fit_ews = np.array(ews_to_plot)[~np.isnan(ews_to_plot)]  # Remove NaN values
    
#     # Step 5: Compute the fixed fit line
#     fixed_fit_ews = np.exp(0.27 - 0.015 * fit_data)
    
#     # Step 6: Create the log-log scale plot
#     # plt.figure(figsize=(16, 12))
#     plt.figure(figsize=(6, 9))

#     plt.scatter(distances_non_zero, ews_non_zero[0], color='b', alpha=0.25, label="All Data")
    
#     if use_binned_statistics:
#         plt.scatter(bin_centers, median_ews, color='black', marker='s', s=80, label="Median in Bins")
#         plt.scatter(bin_centers, mean_ews, color='r', marker='o', s=80, label="Mean in Bins")
    
#     # Plot the fixed fit
#     plt.plot(fit_data, fixed_fit_ews, color='green', label='Fixed Fit: log(EW) = 0.27 - 0.015*distance')
    
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('Distance from Center [kpc]')
#     plt.ylabel('Equivalent Width (EW) [Å]')
#     plt.title('Equivalent Widths vs. Distance (Log-Log Scale)')
#     plt.legend()
#     plt.grid(True)

#     # Save the plot to a file
#     plt.tight_layout()
#     plt.savefig('equivalent_widths_vs_distance_loglog_fixed_fit.png')  # Save as PNG file
#     plt.close()  # Close the plot to free up memory

#     print("Log-log plot with fixed fit saved as 'equivalent_widths_vs_distance_loglog_fixed_fit.png'")

# # Example usage:
# plot_equivalent_widths(use_binned_statistics=True, num_bins=10, fit_median=False)

