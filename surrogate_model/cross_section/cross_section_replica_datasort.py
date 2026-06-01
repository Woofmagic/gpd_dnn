#################################################################################
# FILE INFORMATION:
# Purpose: generate replica pseudodata for cross-section
# Created: 20260601
# Last changed: 20260601
# Notes:
# 1. 20260601:
#   There IS a SEED BEING USED!
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#################################################################################
# Scratch path
#################################################################################

# verify this is what you want
SCRATCH_PATH = 'placeholder!'

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Begin main program flow!
#################################################################################

# sampling true points with error distribution
USING_GAUSSIAN_ERROR_SAMPLING = True
_DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE = 0.1 # 90% temporary, 10% testing
_DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE = 0.1 # of the above 90% temporary, 90% training, 10% validation

#################################################################################
# Multithreading on HPC
#################################################################################

# replica number
replica_number = int(sys.argv[1])

#################################################################################
# Reading the pseudodata file:
#################################################################################

# this reads the pseudodata!
pseudodata_dataframe = pd.read_csv(
    filepath_or_buffer = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/refined_experimental_data_v{MAJOR_MINOR_NUMBER}.csv"
)

#################################################################################
# Use the pseudodata sampling technique with experimental errors:
#################################################################################

# saves a copy of the column:
pseudodata_dataframe['original_xsec'] = pseudodata_dataframe['unp_beam_unp_target_xsec']

if USING_GAUSSIAN_ERROR_SAMPLING:

    pseudodata_dataframe['unp_beam_unp_target_xsec'] = np.random.normal(
        loc = pseudodata_dataframe['original_xsec'],
        scale = pseudodata_dataframe['unp_beam_unp_target_xsec_err']
    )

x_data = pseudodata_dataframe[["t", "x_b", "q_squared", "phi"]]
y_data = pseudodata_dataframe[["unp_beam_unp_target_xsec"]]

TOTAL_DATA_SIZE = len(x_data)
print(f"[INFO]: Total data size is: {TOTAL_DATA_SIZE}")

print(f"[INFO]: This replica number is: Replica #{replica_number}")

# testing/temporary split:
x_remaining, x_testing, y_remaining, y_testing = train_test_split(
    x_data, y_data, test_size = _DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE, shuffle = True, random_state = 31415)

# training/validation split:
x_training, x_validation, y_training, y_validation = train_test_split(
    x_remaining, y_remaining, test_size = _DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE, shuffle = True, random_state = 31415)

print(f"[INFO]: Total number of training points: {len(x_training)}")
print(f"[INFO]: Total number of validaion points: {len(x_validation)}")
print(f"[INFO]: Total number of testing points: {len(x_testing)}")

#################################################################################
# Label the dataframe with training/validation/testing split:
#################################################################################

# augment rows with train/test/val
pseudodata_dataframe.loc[x_training.index, 'split'] = 'train'
pseudodata_dataframe.loc[x_validation.index, 'split'] = 'validation'
pseudodata_dataframe.loc[x_testing.index, 'split'] = 'test'

print(pseudodata_dataframe.columns)

pseudodata_dataframe.to_csv(
    path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/dnn_data_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.csv",
    index = False
)

print("[INFO]: End of script reached!")
