# # ########################## WORKS ALSO WITH LARGER SIMULATIONS, color code,  boolean for both max observed distance and show observed distances, histograms, filter min d_v90 values, histograms ####### 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ——————————————————————————————
# Load & align data
ray_coords = pd.read_csv('rays_coords.csv', header=None)
delta_v90_df = pd.read_csv("delta_v90.csv")
rays_n_clouds_df = pd.read_csv("rays_n_clouds.csv", header=None)

# Diagnostics
print("✅ Loaded data")
print(f"  - ray_coords: {len(ray_coords)} rows")
print(f"  - delta_v90_df: {len(delta_v90_df)} rows")
print(f"  - rays_n_clouds_df: {len(rays_n_clouds_df)} rows")

# Extract ray indices
ray_indices = delta_v90_df["Ray Number"].values

# Check for out-of-bounds indices
max_index = ray_indices.max()
print(f"  - Max Ray Number in delta_v90_df: {max_index}")
if max_index >= len(ray_coords):
    print("⚠️  Warning: Some ray indices exceed ray_coords size. Filtering them out.")

# Filter invalid indices
valid_mask = ray_indices < len(ray_coords)
delta_v90_df = delta_v90_df[valid_mask].reset_index(drop=True)
ray_indices = delta_v90_df["Ray Number"].values

# Align other data
selected_coords = ray_coords.iloc[ray_indices].reset_index(drop=True)
selected_n_clouds = rays_n_clouds_df.iloc[ray_indices].reset_index(drop=True)
distances = np.sqrt(selected_coords[0]**2 + selected_coords[1]**2)

df = pd.DataFrame({
    "Distance": distances,
    "Delta V_90": delta_v90_df["Delta V_90"],
    "Num Clouds": selected_n_clouds[0]
})

# ——————————————————————————————
# Filtering options
include_n = None
exclude_n_below_or_equal_to = 0
exclude_delta_v90_below_or_equal_to = 20

if include_n is not None:
    df = df[df["Num Clouds"] == include_n]

if exclude_n_below_or_equal_to is not None:
    df = df[df["Num Clouds"] > exclude_n_below_or_equal_to]

if exclude_delta_v90_below_or_equal_to is not None:
    df = df[df["Delta V_90"] > exclude_delta_v90_below_or_equal_to]

# ——————————————————————————————
# Load observed data
include_observed_data = True
filter_lise_by_max_distance = True

if include_observed_data:
    lise_df = pd.read_csv("../../observed_table/cleaned_table.csv")
    lise_df = lise_df.dropna(subset=["b", "errb", "v90"])
    if filter_lise_by_max_distance:
        max_distance = df["Distance"].max()
        lise_df = lise_df[lise_df["b"] <= max_distance]

# ——————————————————————————————
# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot + colorbar
pts = ax.scatter(
    df["Distance"], df["Delta V_90"],
    c=df["Num Clouds"], cmap='viridis', alpha=0.7,
    label='Simulated (CloudFlex)'
)
cbar = fig.colorbar(pts, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Number of Clouds")

# Overlay observed data
if include_observed_data:
    ax.errorbar(
        lise_df["b"], lise_df["v90"],
        xerr=lise_df["errb"],
        fmt='o', color='orange', label='Observed data'
    )

ax.set_xlabel('Distance (kpc)')
ax.set_ylabel('ΔV₉₀ (km/s)')
ax.grid(True)
ax.legend()

# ——————————————————————————————
# Attach histograms
divider = make_axes_locatable(ax)

# Top histogram (Distance)
ax_histx = divider.append_axes("top", size="15%", pad=0.1, sharex=ax)
ax_histx.hist(df["Distance"], bins=30, color='gray', alpha=0.6)
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histx.set_ylabel('Count')

# Right histogram (Delta V_90)
ax_histy = divider.append_axes("right", size="15%", pad=0.1, sharey=ax)
ax_histy.hist(df["Delta V_90"], bins=30, orientation='horizontal', color='gray', alpha=0.6)
ax_histy.tick_params(axis="y", labelleft=False)
ax_histy.set_xlabel('Count')

# ——————————————————————————————
fig.suptitle('ΔV₉₀ vs Distance (Simulated & Observed)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save & show
plt.savefig('delta_v90_jointplot_with_observed.png', dpi=300)
plt.show()

# need rays_coords.csv, delta_v90.csv, rays_n_clouds.csv files


# # ########################## color code,  boolean for both max observed distance and show observed distances, histograms, filter min d_v90 values, histograms ####### 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# # ——————————————————————————————
# # Load & align data
# ray_coords = pd.read_csv('rays_coords.csv', header=None)
# delta_v90_df = pd.read_csv("delta_v90.csv")
# rays_n_clouds_df = pd.read_csv("rays_n_clouds.csv", header=None)

# ray_indices = delta_v90_df["Ray Number"].values
# selected_coords = ray_coords.iloc[ray_indices].reset_index(drop=True)
# selected_n_clouds = rays_n_clouds_df.iloc[ray_indices].reset_index(drop=True)
# distances = np.sqrt(selected_coords[0]**2 + selected_coords[1]**2)

# df = pd.DataFrame({
#     "Distance": distances,
#     "Delta V_90": delta_v90_df["Delta V_90"],
#     "Num Clouds": selected_n_clouds[0]
# })

# # ——————————————————————————————
# # Filtering options
# include_n = None
# exclude_n_below_or_equal_to = 0
# exclude_delta_v90_below_or_equal_to = 0

# if include_n is not None:
#     df = df[df["Num Clouds"] == include_n]

# if exclude_n_below_or_equal_to is not None:
#     df = df[df["Num Clouds"] > exclude_n_below_or_equal_to]

# if exclude_delta_v90_below_or_equal_to is not None:
#     df = df[df["Delta V_90"] > exclude_delta_v90_below_or_equal_to]

# # ——————————————————————————————
# # Load observed data
# include_observed_data = True
# filter_lise_by_max_distance = True

# if include_observed_data:
#     lise_df = pd.read_csv("/Users/patrickbates/bachelors_cloudflex/lise_table/cleaned_table.csv")
#     lise_df = lise_df.dropna(subset=["b", "errb", "v90"])
#     if filter_lise_by_max_distance:
#         max_distance = df["Distance"].max()
#         lise_df = lise_df[lise_df["b"] <= max_distance]

# # ——————————————————————————————
# # Plotting
# fig, ax = plt.subplots(figsize=(8, 6))

# # Scatter plot + colorbar
# pts = ax.scatter(
#     df["Distance"], df["Delta V_90"],
#     c=df["Num Clouds"], cmap='viridis', alpha=0.7,
#     label='Simulated (CloudFlex)'
# )
# cbar = fig.colorbar(pts, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label("Number of Clouds")

# # Overlay observed data
# if include_observed_data:
#     ax.errorbar(
#         lise_df["b"], lise_df["v90"],
#         xerr=lise_df["errb"],
#         fmt='o', color='orange', label='Observed data'
#     )

# ax.set_xlabel('Distance (kpc)')
# ax.set_ylabel('ΔV₉₀ (km/s)')
# ax.grid(True)
# ax.legend()

# # ——————————————————————————————
# # Attach histograms
# divider = make_axes_locatable(ax)

# # Top histogram (Distance)
# ax_histx = divider.append_axes("top", size="15%", pad=0.1, sharex=ax)
# ax_histx.hist(df["Distance"], bins=30, color='gray', alpha=0.6)
# ax_histx.tick_params(axis="x", labelbottom=False)
# ax_histx.set_ylabel('Count')

# # Right histogram (Delta V_90)
# ax_histy = divider.append_axes("right", size="15%", pad=0.1, sharey=ax)
# ax_histy.hist(df["Delta V_90"], bins=30, orientation='horizontal', color='gray', alpha=0.6)
# ax_histy.tick_params(axis="y", labelleft=False)
# ax_histy.set_xlabel('Count')

# # ——————————————————————————————
# fig.suptitle('ΔV₉₀ vs Distance (Simulated & Observed)', fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.98])

# # Save & show
# plt.savefig('delta_v90_jointplot_with_observed.png', dpi=300)
# plt.show()




# # ########################## color code,  boolean for both max observed distance and show observed distances, histograms, filter min d_v90 values ####### 

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Step 1: Read the ray coordinates
# ray_coords = pd.read_csv('rays_coords.csv', header=None)

# # Step 2: Read the delta_v_90 values with header
# delta_v90_df = pd.read_csv("delta_v90.csv")

# # Step 3: Read the number of clouds per ray
# rays_n_clouds_df = pd.read_csv("rays_n_clouds.csv", header=None)

# # Step 4: Use the "Ray Number" to align everything
# ray_indices = delta_v90_df["Ray Number"].values
# selected_coords = ray_coords.iloc[ray_indices].reset_index(drop=True)
# selected_n_clouds = rays_n_clouds_df.iloc[ray_indices].reset_index(drop=True)

# # Step 5: Compute distances
# distances = np.sqrt(selected_coords[0]**2 + selected_coords[1]**2)

# # Step 6: Create main plot dataframe
# distance_delta_v_df = pd.DataFrame({
#     "Distance": distances,
#     "Delta V_90": delta_v90_df["Delta V_90"],
#     "Num Clouds": selected_n_clouds[0]
# })

# # Step 7: Filtering options
# include_n = None                      # Optional: show only n = this value
# exclude_n_below_or_equal_to = 0      # Optional: exclude n <= this value
# exclude_delta_v90_below_or_equal_to = 0  # Optional: exclude Delta V_90 <= x (example value of x = 100)

# if include_n is not None:
#     distance_delta_v_df = distance_delta_v_df[distance_delta_v_df["Num Clouds"] == include_n]

# if exclude_n_below_or_equal_to is not None:
#     distance_delta_v_df = distance_delta_v_df[distance_delta_v_df["Num Clouds"] > exclude_n_below_or_equal_to]

# # Filter by Delta V_90 value
# if exclude_delta_v90_below_or_equal_to is not None:
#     distance_delta_v_df = distance_delta_v_df[distance_delta_v_df["Delta V_90"] > exclude_delta_v90_below_or_equal_to]

# # Step 8: Include observed data (optional)
# include_observed_data = True
# if include_observed_data:
#     lise_df = pd.read_csv("/Users/patrickbates/bachelors_cloudflex/lise_table/cleaned_table.csv")
#     lise_df = lise_df.dropna(subset=["b", "errb", "v90"])
#     filter_lise_by_max_distance = True
#     if filter_lise_by_max_distance:
#         max_distance = distance_delta_v_df["Distance"].max()
#         lise_df = lise_df[lise_df["b"] <= max_distance]

# # Step 9: Plot
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     distance_delta_v_df["Distance"],
#     distance_delta_v_df["Delta V_90"],
#     c=distance_delta_v_df["Num Clouds"],
#     cmap='viridis',
#     alpha=0.7,
#     label='CloudFlex',
# )

# cbar = plt.colorbar(scatter)
# cbar.set_label("Number of Clouds")

# if include_observed_data:
#     plt.errorbar(lise_df["b"], lise_df["v90"],
#                  xerr=lise_df["errb"],
#                  fmt='o', color='orange', label='Observed data')

# plt.xlabel('Distance from Z-axis / Doppler b (kpc / km/s)', fontsize=12)
# plt.ylabel('Delta V_90 (km/s)', fontsize=12)
# plt.title('Delta V_90 Comparison', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.savefig('delta_v_90_vs_distance_combined.png', format='png')
# plt.show()

# ########################## color code,  boolean for both max observed distance and show observed distances ####### 

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Step 1: Read the ray coordinates
# ray_coords = pd.read_csv('rays_coords.csv', header=None)

# # Step 2: Read the delta_v_90 values with header
# delta_v90_df = pd.read_csv("delta_v90.csv")

# # Step 3: Read the number of clouds per ray
# rays_n_clouds_df = pd.read_csv("rays_n_clouds.csv", header=None)

# # Step 4: Use the "Ray Number" to align everything
# ray_indices = delta_v90_df["Ray Number"].values
# selected_coords = ray_coords.iloc[ray_indices].reset_index(drop=True)
# selected_n_clouds = rays_n_clouds_df.iloc[ray_indices].reset_index(drop=True)

# # Step 5: Compute distances
# distances = np.sqrt(selected_coords[0]**2 + selected_coords[1]**2)

# # Step 6: Create main plot dataframe
# distance_delta_v_df = pd.DataFrame({
#     "Distance": distances,
#     "Delta V_90": delta_v90_df["Delta V_90"],
#     "Num Clouds": selected_n_clouds[0]
# })

# # Step 7: Filtering options
# include_n = None                      # Optional: show only n = this value
# exclude_n_below_or_equal_to = 0      # Optional: exclude n <= this value

# if include_n is not None:
#     distance_delta_v_df = distance_delta_v_df[distance_delta_v_df["Num Clouds"] == include_n]

# if exclude_n_below_or_equal_to is not None:
#     distance_delta_v_df = distance_delta_v_df[distance_delta_v_df["Num Clouds"] > exclude_n_below_or_equal_to]

# # Step 8: Include observed data (optional)
# include_observed_data = True
# if include_observed_data:
#     lise_df = pd.read_csv("/Users/patrickbates/bachelors_cloudflex/lise_table/cleaned_table.csv")
#     lise_df = lise_df.dropna(subset=["b", "errb", "v90"])
#     filter_lise_by_max_distance = True
#     if filter_lise_by_max_distance:
#         max_distance = distance_delta_v_df["Distance"].max()
#         lise_df = lise_df[lise_df["b"] <= max_distance]

# # Step 9: Plot
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     distance_delta_v_df["Distance"],
#     distance_delta_v_df["Delta V_90"],
#     c=distance_delta_v_df["Num Clouds"],
#     cmap='viridis',
#     alpha=0.7,
#     label='CloudFlex',
# )

# cbar = plt.colorbar(scatter)
# cbar.set_label("Number of Clouds")

# if include_observed_data:
#     plt.errorbar(lise_df["b"], lise_df["v90"],
#                  xerr=lise_df["errb"],
#                  fmt='o', color='orange', label='Observed data')

# plt.xlabel('Distance from Z-axis / Doppler b (kpc / km/s)', fontsize=12)
# plt.ylabel('Delta V_90 (km/s)', fontsize=12)
# plt.title('Delta V_90 Comparison', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.savefig('delta_v_90_vs_distance_combined.png', format='png')
# plt.show()



# ##### boolean for both max observed distance and show observed distances ##########

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Step 1: Read the ray coordinates
# ray_coords = pd.read_csv('rays_coords.csv', header=None)

# # Step 2: Compute 2D distances
# distances = np.sqrt(ray_coords[0]**2 + ray_coords[1]**2)

# # Step 3: Read the delta_v_90 values
# delta_v_90_df = pd.read_csv("delta_v90.csv")

# # Step 4: Read the rays_n_clouds.csv file
# rays_n_clouds_df = pd.read_csv("rays_n_clouds.csv", header=None)

# # Step 5: Ensure all dataframes have the same length
# min_length = min(len(ray_coords), len(delta_v_90_df), len(rays_n_clouds_df))
# ray_coords = ray_coords.iloc[:min_length]
# distances = distances.iloc[:min_length]
# delta_v_90_df = delta_v_90_df.iloc[:min_length]
# rays_n_clouds_df = rays_n_clouds_df.iloc[:min_length]

# # User-defined filter for cloud count
# include_n = None  # Set to an integer to filter by specific number of clouds
# if include_n is not None:
#     include_indices = rays_n_clouds_df[0] == include_n
# else:
#     include_indices = np.ones(len(rays_n_clouds_df), dtype=bool)

# # Apply filter
# distances = distances[include_indices]
# delta_v_90_df = delta_v_90_df[include_indices]

# # Step 6: Create primary plot dataframe
# distance_delta_v_df = pd.DataFrame({
#     "Distance": distances,
#     "Delta V_90": delta_v_90_df["Delta V_90"]
# })

# # Optional: Include observed data (Lise table)
# include_observed_data = True  # <<< Toggle this to include/exclude observed data
# if include_observed_data:
#     lise_df = pd.read_csv("/Users/patrickbates/bachelors_cloudflex/lise_table/cleaned_table.csv")
#     lise_df = lise_df.dropna(subset=["b", "errb", "v90"])
#     filter_lise_by_max_distance = True
#     if filter_lise_by_max_distance:
#         max_distance = distances.max()
#         lise_df = lise_df[lise_df["b"] <= max_distance]
#     lise_df = lise_df.dropna(subset=["b", "errb", "v90"])

# # Step 8: Plot
# use_log_log = False
# plt.figure(figsize=(8, 6))

# if use_log_log:
#     plt.loglog(distance_delta_v_df["Distance"], distance_delta_v_df["Delta V_90"],
#                marker='o', linestyle='None', color='b', label='CloudFlex')
#     if include_observed_data:
#         plt.errorbar(lise_df["b"], lise_df["v90"],
#                      xerr=lise_df["errb"],
#                      fmt='o', color='orange', label='Observed data')
#     plt.xlabel('Log(X-axis)', fontsize=12)
#     plt.ylabel('Log(Y-axis)', fontsize=12)
#     plt.title('Log-Log Plot: Delta V_90 Comparison', fontsize=14)
# else:
#     plt.plot(distance_delta_v_df["Distance"], distance_delta_v_df["Delta V_90"],
#              marker='o', linestyle='None', color='b', label='CloudFlex')
#     if include_observed_data:
#         plt.errorbar(lise_df["b"], lise_df["v90"],
#                      xerr=lise_df["errb"],
#                      fmt='o', color='orange', label='Observed data')
#     plt.xlabel('Distance from Z-axis / Doppler b (kpc / km/s)', fontsize=12)
#     plt.ylabel('Delta V_90 (km/s)', fontsize=12)
#     plt.title('Delta V_90 Comparison', fontsize=14)

# plt.legend()
# plt.grid(True)
# plt.savefig('delta_v_90_vs_distance_combined.png', format='png')
# plt.show()



######### boolean to select max distance for observed distances ##########

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Step 1: Read the ray coordinates from the rays_coords.csv file
# ray_coords = pd.read_csv('rays_coords.csv', header=None)  # No header specified, first column = x, second column = y

# # Step 2: Compute 2D distances from the z-axis (using Euclidean distance formula)
# distances = np.sqrt(ray_coords[0]**2 + ray_coords[1]**2)  # sqrt(x^2 + y^2)

# # Step 3: Read the delta_v_90 values from the previously computed file
# delta_v_90_df = pd.read_csv("delta_v90.csv")

# # Step 4: Read the rays_n_clouds.csv file to check for rays with a value of 1.0
# rays_n_clouds_df = pd.read_csv("rays_n_clouds.csv", header=None)  # Assuming 1.0 means to exclude

# # Step 5: Boolean to exclude rays with a value of 1.0 in rays_n_clouds.csv
# exclude_n_1 = True  # Set to True to exclude rays with 1.0, False to include all rays

# # Step 6: Filter the rays if 'exclude_n_1' is True
# if exclude_n_1:
#     # Get the indices of rays that should be included (where the value is not 1.0)
#     include_indices = rays_n_clouds_df[0] != 1.0
#     # Apply the filter to both the distances and delta_v_90 values
#     distances = distances[include_indices]
#     delta_v_90_df = delta_v_90_df[include_indices]

# # Step 7: Combine the delta_v_90 values with their respective distances
# distance_delta_v_df = pd.DataFrame({
#     "Distance": distances,
#     "Delta V_90": delta_v_90_df["Delta V_90"]
# })

# # Boolean to choose between linear or log-log plot
# use_log_log = False  # Set to False for linear plot, True for log-log plot

# # Step 8: Plot the graph of delta_v_90 vs. distance
# plt.figure(figsize=(8, 6))

# if use_log_log:
#     # Log-log plot
#     plt.loglog(distance_delta_v_df["Distance"], distance_delta_v_df["Delta V_90"], marker='o', linestyle='None', color='b')
#     plt.xlabel('Log(Distance from Z-axis) (kpc)', fontsize=12)
#     plt.ylabel('Log(Delta V_90) (km/s)', fontsize=12)
#     plt.title('Log-Log Plot: Delta V_90 as a function of Distance from Z-axis', fontsize=14)
# else:
#     # Linear plot
#     plt.plot(distance_delta_v_df["Distance"], distance_delta_v_df["Delta V_90"], marker='o', linestyle='None', color='b')
#     plt.xlabel('Distance from Z-axis (kpc)', fontsize=12)
#     plt.ylabel('Delta V_90 (km/s)', fontsize=12)
#     plt.title('Delta V_90 as a function of Distance from Z-axis', fontsize=14)

# plt.grid(True)
# plt.savefig('delta_v_90_vs_distance.png', format='png')

# plt.show()




# ######## choose n clouds to use for plot ############

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Step 1: Read the ray coordinates from the rays_coords.csv file
# ray_coords = pd.read_csv('rays_coords.csv', header=None)  

# # Step 2: Compute 2D distances from the z-axis
# distances = np.sqrt(ray_coords[0]**2 + ray_coords[1]**2)  

# # Step 3: Read the delta_v_90 values
# delta_v_90_df = pd.read_csv("delta_v90.csv")

# # Step 4: Read the rays_n_clouds.csv file
# rays_n_clouds_df = pd.read_csv("rays_n_clouds.csv", header=None)  

# # Step 5: Ensure all dataframes have the same length
# min_length = min(len(ray_coords), len(delta_v_90_df), len(rays_n_clouds_df))

# ray_coords = ray_coords.iloc[:min_length]
# distances = distances.iloc[:min_length]
# delta_v_90_df = delta_v_90_df.iloc[:min_length]
# rays_n_clouds_df = rays_n_clouds_df.iloc[:min_length]

# # User-defined filter: Set to None to include all rays, or specify a value to filter
# include_n = None         # None, otherwise enter float for # of clouds

# # Step 6: Apply filtering
# if include_n is not None:
#     include_indices = rays_n_clouds_df[0] == include_n
# else:
#     include_indices = np.ones(len(rays_n_clouds_df), dtype=bool)  

# # Apply filter
# distances = distances[include_indices]
# delta_v_90_df = delta_v_90_df[include_indices]

# # Step 7: Create dataframe
# distance_delta_v_df = pd.DataFrame({
#     "Distance": distances,
#     "Delta V_90": delta_v_90_df["Delta V_90"]
# })

# # Boolean for log-log plot
# use_log_log = False  

# # Step 8: Plot
# plt.figure(figsize=(8, 6))

# if use_log_log:
#     plt.loglog(distance_delta_v_df["Distance"], distance_delta_v_df["Delta V_90"], marker='o', linestyle='None', color='b')
#     plt.xlabel('Log(Distance from Z-axis) (kpc)', fontsize=12)
#     plt.ylabel('Log(Delta V_90) (km/s)', fontsize=12)
#     plt.title('Log-Log Plot: Delta V_90 as a function of Distance from Z-axis', fontsize=14)
# else:
#     plt.plot(distance_delta_v_df["Distance"], distance_delta_v_df["Delta V_90"], marker='o', linestyle='None', color='b')
#     plt.xlabel('Distance from Z-axis (kpc)', fontsize=12)
#     plt.ylabel('Delta V_90 (km/s)', fontsize=12)
#     plt.title('Delta V_90 as a function of Distance from Z-axis', fontsize=14)

# plt.grid(True)

# # Save the plot as a .png file
# plt.savefig('delta_v_90_vs_distance.png', format='png')

# # Show the plot
# plt.show()
