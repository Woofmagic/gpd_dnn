# How to use these files:

Steps:

1. Generate a kinematic grid

i) Run `initiate_kinematic_grid.slurm`.
ii) Generates *all possible data* including unphysical values.

2. Combine all the local fit data into a global datafile.

i) Run `combine_kinematic_set_data.slurm`.
ii) Download the resulting file: 

3. Make *local* plot of kinematic grid.

i) Run `datafile_analysis.ipynb`.
ii) Create huge `.csv` file across *all* kinematic sets.

4. Upload the `.csv` file that you just created to the correct scratch folder.

5. Run `replica_script.slurm`

i) To properly multithread, calculate number of replicas and total number of kinematic settings.
ii) Multiply the two and then pass this number into the SLURM file.
iii) Will early exit if kinematics are unphysical.
iv) Generates all the replicas.

6. Run `replica_average.slurm`

i) The slurm parameter is the number of good kinematic settings.
