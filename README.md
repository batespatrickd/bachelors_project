# bachelors_project

The source code of this version of CloudFlex has been modified to extract extra data, which the original version did not provide.

## Dependencies 

```
astropy==7.0.1
cmocean==4.0.3
cmyt==2.0.2
h5py==3.12.1
matplotlib==3.10.3
matplotlib_scalebar==0.9.0
numpy==2.3.0
scipy==1.15.3
tqdm==4.67.1
trident==1.4.2
unyt==3.0.3
yt==4.4.0
```


## Usage

- Ensure that the `src`, `own_scripts` and `observed_table` folders are located in the same directory.
- Download the Trident ion balance file, hm2012_hr.h5.gz found at https://trident-project.org/data/ion_table/ and insert in the `src` folder.
- Adjust parameters and generate clouds in:

```
src/make_clouds.py
```

- Determine the amount of rays and send them through clouds:

```
src/make_rays.py clouds.h5
```


- Extract extra data (run the following scripts in this order):

```
own_scripts/h5_inspect_rays_clouds.py
own_scripts/h5_inspecta_spectra.py
own_scripts/delta_v90.py
```


The necessary `.csv` files will now be generated, and the rest of the scripts in `own_scripts` will be able to run.

## Contact

For questions, suggestions, or issues:

**Patrick Bates**  
Email: gpk712@alumni.ku.dk  






