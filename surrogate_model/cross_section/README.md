## Mathematica:

The Mathematica notebooks are for locally testing the DNN workflow that is implemented in the Python script. It is a Mathematica cross-check with the results that we got from the Python scripts. [NOTE]: So far, the notebook has not helped much, and was really for ensuring that the workflow is correct more than anything. It was also used to check if there were indeed "topological issues" in the raw dataset that prevented a standard DNN from learning the data. The question is still outstanding...

1. `cross_section_analytical_dnn.nb`

## Notebooks:

The Jupyter notebooks are for locally testing various workflows and programs before we need to run them on the HPC. We explain their use below:

1. `cross_section_surrogate_replicas.ipynb` is for a global fit to the dataset containing $d^{4}\sigma^{UU}$ as a function of $k$, $t$, $x_{\text{B}}$, $Q^{2}$, and $u(\phi)$ using many replicas.

## Local Scripts:

These scripts are ones that can be run locally.

1. `cross_section_replicas_surrogate_script` is a simple script that runs a DNN on a collection of data and tries to fit $d^{4}\sigma^{UU}$ as a function of $k$, $t$, $x_{\text{B}}$, $Q^{2}$, and $u(\phi)$. The final variable $u(\phi)$ is a transformation from $\phi$ data to another variable that respects the topological feature of an azimuthal angle, namely $\phi \sim \phi + 2\pi$.

## HPC Scripts:

These scripts are used in an HPC context only.

1. `cross_section_surrogate_model.py` is the *actual training script* that runs and fits the DNN. It (i) generates the DNN replica data (which may or may not involved pseudodata sampling) and then (ii) actually runs the TF DNN on that data. Remember that it is learning $d^{4}\sigma^{UU}$ as a function of $k$, $t$, $x_{\text{B}}$, $Q^{2}$, and $u(\phi)$.

2. `plot_local_fits.py` generates the plot corresponding to local fits to the dataset. That is, for a unique combination of $k$, $t$, $x_{\text{B}}$, $Q^{2}$, we fit $d^{4}\sigma^{UU}$ as a function of $u(\phi)$. This script analyzes the DNN performance on *both* (i) the actual data in the dataset and (ii) interpolated regions of the dataset. Not only does it make local fit plots, but it also makes a collection of surface plots, representing (again) the DNN interpolation between the discrete points that were featured in the original dataset.