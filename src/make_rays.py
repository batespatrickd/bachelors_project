import csv

"""
Run ray experiments sending N sightlines through the cloud cluster
"""

from cloudflex import \
    SpectrumGenerator, \
    plot_histogram, \
    plot_clouds, \
    Clouds, \
    plot_covering_frac, \
    plot_clouds_buffer
import sys
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit('Usage: make_rays.py clouds.h5')
    fn = sys.argv[1]

    clouds = Clouds()
    clouds.load(fn)
    params = clouds.params
    np.random.seed(seed=params['seed'])
    #plot_clouds_buffer(clouds, filename='clouds_3.png')
    #distances = np.linalg.norm(np.array([0,0,0]) - clouds.centers, axis=1)
    #velocities = np.linalg.norm(np.array([0,0,0]) - clouds.velocities, axis=1)
    #plot_histogram(distances, "cloud radial distances", clouds, log=True, xrange=[-1, 1])
    #plot_histogram(clouds.masses, "cloud masses", clouds, log=True, xrange=[-3, 5])
    #plot_histogram(velocities, "cloud total velocities", clouds, log=True, xrange=[-1,2.5])

    ## Generate and Plot Spectra for N random rays passing through domain
    ## Also plot distribution of clouds and rays

    # Create N random Rays passing through domain
    N = 10000
    sg = SpectrumGenerator(clouds, subcloud_turb=True)
    rays = sg.generate_ray_sample(N, params['center'],  params['dclmax'],
                                  first_n_to_plot=N, attach_spectra=True)
    #plot_clouds_buffer(clouds, filename='clouds_4.png', rays=rays)
    
    
    def save_ray_data(rays, filename='ray_data.csv'):
        # Open the CSV file in write mode
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(['Ray Index', 'Cloud Center z', 'Cloud Center y', 'Cloud Radius', 'Velocity z', 'Velocity y'])

            # Iterate over each ray and save its data
            for i in range(rays.N):
                cloud_indices = rays.clouds.get(i, [])  # Get cloud indices for this ray
                for cloud_idx in cloud_indices:
                    center_z, center_y = clouds.centers[cloud_idx, 2], clouds.centers[cloud_idx, 1]
                    radius = clouds.radii[cloud_idx]
                    velocity_z, velocity_y = clouds.velocities[cloud_idx, 2], clouds.velocities[cloud_idx, 1]
                    # Write the data for each cloud
                    #writer.writerow([i, center_z, center_y, radius, velocity_z, velocity_y])
                    writer.writerow([i+1, center_z, center_y, radius, velocity_z, velocity_y])


    # save data
    save_ray_data(rays)  

    rays.save('rays.h5')
    #plot_clouds(rays=rays, clouds=clouds, filename='clouds_rays.png')


    # Continue with the plotting
    plot_histogram(rays.column_densities, "column densities", clouds, log=True, xrange=[11, 18])
    plot_histogram(rays.EWs, "equivalent widths", clouds, xrange=[0, 1.0], n_bins=40)

    # Create histograms of EW, clouds intersected, and column density of all rays
    #plot_histogram(rays.n_clouds, "clouds intersected", clouds)
    plot_histogram(rays.column_densities, "column densities", clouds, log=True, xrange=[11,18])
    plot_histogram(rays.EWs, "equivalent widths", clouds, xrange=[0, 1.0], n_bins=40)#, annotate='Subcloud Turbulence')


