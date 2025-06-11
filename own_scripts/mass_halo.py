# Script debugged and modified with the help of OpenAI. (2025). ChatGPT (June 11 version) [Large language model]. https://chat.openai.com/


import csv
import numpy as np
import math

# Gravitational constant in m^3 kg^-1 s^-2
G = 6.67430e-11  

# Conversion factor: 1 kpc = 3.085677581e19 meters
KPC_TO_METERS = 3.085677581e19

# Solar mass in kg
SOLAR_MASS_KG = 1.98847e30

def compute_halo_mass(radius_kpc, sim_mass=None, filename="clouds_velocities.csv"):
    """
    Compute halo mass along x, y, and z velocity components and optionally compare to simulation mass.

    Parameters:
        radius_kpc (float): Radius in kiloparsecs.
        sim_mass (float, optional): Mass to compare against, in solar masses.
        filename (str): Path to the CSV file with velocity components in km/s.

    Returns:
        tuple: M_halo_x, M_halo_y, M_halo_z in solar masses.
    """
    velocities_x, velocities_y, velocities_z = [], [], []

    # Read the CSV file and extract the columns
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 3:
                x, y, z = map(float, row)
                velocities_x.append(x * 1e3)  # convert km/s to m/s
                velocities_y.append(y * 1e3)
                velocities_z.append(z * 1e3)

    # Calculate standard deviations (in m/s)
    sigma_x = np.std(velocities_x, ddof=1)
    sigma_y = np.std(velocities_y, ddof=1)
    sigma_z = np.std(velocities_z, ddof=1)

    # Convert radius to meters
    r_meters = radius_kpc * KPC_TO_METERS

    # Calculate mass in kg for each component
    M_halo_x_kg = (5 * sigma_x**2 * r_meters) / G
    M_halo_y_kg = (5 * sigma_y**2 * r_meters) / G
    M_halo_z_kg = (5 * sigma_z**2 * r_meters) / G

    # Convert to solar masses
    M_halo_x = M_halo_x_kg / SOLAR_MASS_KG
    M_halo_y = M_halo_y_kg / SOLAR_MASS_KG
    M_halo_z = M_halo_z_kg / SOLAR_MASS_KG

    # Print results
    print(f"x component mass: {M_halo_x:.2e} M☉")
    print(f"y component mass: {M_halo_y:.2e} M☉")
    print(f"z component mass: {M_halo_z:.2e} M☉")

    # If sim_mass is provided, print order of magnitude difference
    if sim_mass is not None:
        def orders_of_magnitude(m1, m2):
            return math.log10(m1 / m2)

        print(f"x component is {orders_of_magnitude(M_halo_x, sim_mass):.2f} orders of magnitude greater than sim_mass")
        print(f"y component is {orders_of_magnitude(M_halo_y, sim_mass):.2f} orders of magnitude greater than sim_mass")
        print(f"z component is {orders_of_magnitude(M_halo_z, sim_mass):.2f} orders of magnitude greater than sim_mass")

    return M_halo_x, M_halo_y, M_halo_z

# Example usage
radius_kpc = 5
sim_mass = 1e6  # in solar masses
compute_halo_mass(radius_kpc, sim_mass=sim_mass, filename="clouds_velocities.csv")
