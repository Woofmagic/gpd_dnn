## Notebooks:

The Jupyter notebooks are for locally testing various workflows and programs before we need to run them on the HPC. We explain their use below:

1. `cross_section_surrogate.ipynb` is only for the analysis of a *single kinematic setting* and not global fit to the dataset.

2. `cross_section_surrogate_replicas.ipynb` is for a global fit to the dataset using many replicas.

## HPC Scripts:

1. `cross_section_replica_datasort.py` is a script we run to generate individual replica training, validation, and testing data.

2. `cross_section_surrogate_model.py` is the *actual training script* that runs and fits the DNN. In order to run this script, we must have run the prior one.

3. `plot_replica_learning_curves.py` is the script we run once the HPC has finished training several (or all, ideally) replicas. It generates loss plots based on the `.csv` loss data.

4. `cross_section_smooth_evaluation.py` is the script to run to generate novel predictions with the DNN on data that was not featured in the original dataset. This includes interpolation and extrapolation depending on what we want.