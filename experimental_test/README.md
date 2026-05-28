## What is in this folder?

1. `prototyping.ipynb`
A terrible name for what this notebook is about, but **this notebook creates**  the huge dataset that we used to do the simultaneous fitting procedure.

2. `preparatory_analysis.py`
This file is really for HPC stuff: once you have created a massive dataset equipped with the `set` column, delineating unique kinematic settings, you need to construct individual folders that contain only their respective kinematic setting data. We do this so local fitting routines can operate easily and keep all of their analysis confined to that single directory.

3. `initialize_experimental_data_scan.py`
This is the HPC Python script version of the `prototyping.ipynb` notebook that we made earlier.

4. `refine_experimental_data_scan.py`
Run this **after** `initialize_experimental_data_scan.py`; it analyzes the just-created mega datafile for redundancies and missing data. The major purpose is, again, to have this available on an HPC. This script is also a small block of the `prototyping.ipynb` notebook.