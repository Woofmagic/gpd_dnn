#################################################################################
# FILE INFORMATION:
# Purpose: generate replica pseudodata
# Created: 20260325
# Last changed: 20260722
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import yaml
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

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

MAJOR_NUMBER = config["versioning"]["major"]
MINOR_NUMBER = config["versioning"]["minor"]
MAJOR_MINOR_NUMBER = f"{MAJOR_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

NUMBER_OF_REPLICAS = config["dnn_config"]["replicas"]

print(f"[INFO]: We anticipate {NUMBER_OF_REPLICAS} total replicas at a single kinematic point.")

#################################################################################
# Begin main program flow!
#################################################################################

# sampling true points with error distribution
USING_GAUSSIAN_ERROR_SAMPLING = True
_DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE = 0.8 # 80% temporary, 20% testing
_DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE = 0.8 # of the above 80% temporary, 80% training, 20% validation

#################################################################################
# Multithreading on HPC
#################################################################################

with open(
    f"./slurm_logs/version_{MAJOR_MINOR_NUMBER}/valid_kinematic_sets_v{MAJOR_MINOR_NUMBER}.txt", 
    "r",
    encoding = "utf-8") as good_kinematic_sets_file:
    # This creates a list like [2, 9, 15, ...]
    good_sets = [line.strip() for line in good_kinematic_sets_file if line.strip()]

print(f"[INFO]: Good kinematic sets were: {good_sets}")

# extract kinematic set number
task_id = int(sys.argv[1]) - 1
set_index = task_id // NUMBER_OF_REPLICAS
replica_number = (task_id % NUMBER_OF_REPLICAS) + 1
kinematic_set_number = good_sets[set_index]

#################################################################################
# Reading the pseudodata file:
#################################################################################

# this reads the pseudodata!
pseudodata_dataframe = pd.read_csv(
    filepath_or_buffer = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/main_pseudodata_file_set_{kinematic_set_number}_v{MAJOR_MINOR_NUMBER}.csv"
)

assert pseudodata_dataframe["q_squared"].iloc[0] == pseudodata_dataframe["q_squared"].iloc[5], "[ASSERT]: iloc revealed kinematic sub-dataframe not invariant under index."
assert pseudodata_dataframe["k"].iloc[0] == pseudodata_dataframe["k"].iloc[10], "[ASSERT]: iloc revealed kinematic sub-dataframe not invariant under index."

#################################################################################
# Experimental errors!
#################################################################################

#about 3% total error for xsec --- realistically would be sqrt(N)/N
pseudodata_dataframe["unp_beam_unp_target_xsec_errstat"] = \
    pseudodata_dataframe["unp_beam_unp_target_xsec"] * np.random.uniform(0.02, 0.05)

pseudodata_dataframe["unp_target_bsa_errstat"] = \
    np.abs(pseudodata_dataframe["unp_target_bsa"] * 0.05) + 0.01

# 5% scale uncertainty
SYSTEMATIC_ERROR_FACTOR_XSEC = 0.05
# 3% scale uncertainty
SYSTEMATIC_ERROR_FACTOR_BSA = 0.03

pseudodata_dataframe["unp_beam_unp_target_xsec_errsyst"] = \
    pseudodata_dataframe["unp_beam_unp_target_xsec"] * SYSTEMATIC_ERROR_FACTOR_XSEC

# https://arxiv.org/pdf/2307.07874, pg. 4-5
# ...on the fit results was also studied and estimated to be around 0.005.
pseudodata_dataframe["unp_target_bsa_errsyst"] = \
    np.abs(pseudodata_dataframe["unp_target_bsa"] * SYSTEMATIC_ERROR_FACTOR_BSA) + 0.005

# this is errors in quadrature
pseudodata_dataframe["unp_beam_unp_target_xsec_err"] = np.sqrt(
    pseudodata_dataframe["unp_beam_unp_target_xsec_errstat"]**2 +
    pseudodata_dataframe["unp_beam_unp_target_xsec_errsyst"]**2
)

# this is also quadrature
pseudodata_dataframe["unp_target_bsa_err"] = np.sqrt(
    pseudodata_dataframe["unp_target_bsa_errstat"]**2 +
    pseudodata_dataframe["unp_target_bsa_errsyst"]**2
)

#################################################################################
# Finally creating the pseudodata:
#################################################################################

if USING_GAUSSIAN_ERROR_SAMPLING:

    print("[INFO]: Gaussian error sampling was true, so we are now generating new pseudodata.")

    pseudodata_dataframe["unp_beam_unp_target_xsec"] = np.random.normal(
        loc = pseudodata_dataframe["unp_beam_unp_target_xsec"],
        scale = pseudodata_dataframe["unp_beam_unp_target_xsec_err"]
    )
    
    # Sampling for the BSA
    pseudodata_dataframe["unp_target_bsa"] = np.random.normal(
        loc = pseudodata_dataframe["unp_target_bsa"],
        scale = pseudodata_dataframe["unp_target_bsa_err"]
    )

# extract the x input data to the dnn:
x_data = pseudodata_dataframe[["t", "x_b", "q_squared", "phi"]]
y_data = pseudodata_dataframe[["unp_beam_unp_target_xsec", "unp_target_bsa"]]

TOTAL_DATA_SIZE = len(x_data)
print(f"[INFO]: Total data size is: {TOTAL_DATA_SIZE}")

FIXED_BEAM_ENERGY = pseudodata_dataframe["k"].iloc[0]
FIXED_Q_SQUARED = pseudodata_dataframe["q_squared"].iloc[0]
FIXED_X_BJORKEN = pseudodata_dataframe["x_b"].iloc[0]
FIXED_T_VALUE = pseudodata_dataframe["t"].iloc[0]

print(f"[INFO]: Selected beam energy k = {FIXED_BEAM_ENERGY}")
print(f"[INFO]: Selected Q^2 = {FIXED_Q_SQUARED}")
print(f"[INFO]: Selected xB = {FIXED_X_BJORKEN}")
print(f"[INFO]: Selected t = {FIXED_T_VALUE}")

CFF_REAL_H_KM15 = pseudodata_dataframe["Re[H]"].iloc[0]
CFF_IMAG_H_KM15 = pseudodata_dataframe["Im[H]"].iloc[0]
CFF_REAL_E_KM15 = pseudodata_dataframe["Re[E]"].iloc[0]
CFF_IMAG_E_KM15 = pseudodata_dataframe["Im[E]"].iloc[0]
CFF_REAL_HT_KM15 = pseudodata_dataframe["Re[Ht]"].iloc[0]
CFF_IMAG_HT_KM15 = pseudodata_dataframe["Im[Ht]"].iloc[0]
CFF_REAL_ET_KM15 = pseudodata_dataframe["Re[Et]"].iloc[0]
CFF_IMAG_ET_KM15 = pseudodata_dataframe["Im[Et]"].iloc[0]

CFF_H_KM15 = complex(CFF_REAL_H_KM15, CFF_IMAG_H_KM15)
CFF_H_TILDE_KM15 = complex(CFF_REAL_HT_KM15, CFF_IMAG_HT_KM15)
CFF_E_KM15 = complex(CFF_REAL_E_KM15, CFF_IMAG_E_KM15)
CFF_E_TILDE_KM15 = complex(CFF_REAL_ET_KM15, CFF_IMAG_ET_KM15)

print(f"[INFO]: Selected CFF H = {CFF_H_KM15}")
print(f"[INFO]: Selected CFF E = {CFF_E_KM15}")
print(f"[INFO]: Selected CFF Ht = {CFF_H_TILDE_KM15}")
print(f"[INFO]: Selected CFF Et = {CFF_E_TILDE_KM15}")

number_of_dnn_temporary_points = int(np.ceil(TOTAL_DATA_SIZE * _DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE))
number_of_dnn_testing_points = TOTAL_DATA_SIZE - number_of_dnn_temporary_points
number_of_dnn_training_points = int(np.ceil(number_of_dnn_temporary_points * _DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE))
number_of_dnn_validation_points = TOTAL_DATA_SIZE - number_of_dnn_training_points - number_of_dnn_testing_points

print(f"[INFO]: Testing/Temporary Split is {_DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE * 100}%, giving {number_of_dnn_testing_points} testing points (with ceiling).")
print(f"[INFO]: Training/Validation Split is {_DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE * 100}%, giving {number_of_dnn_validation_points} validation points (with ceiling).")
print(f"[INFO]: Remaining training data points are: {number_of_dnn_training_points}")

print(f"[INFO]: This replica number is: Replica #{replica_number}")

# testing/temporary split:
x_testing, x_remaining, y_testing, y_remaining = train_test_split(
    x_data,
    y_data,
    test_size = _DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE,
    shuffle = True)

# training/validation split:
x_validation, x_training, y_validation, y_training = train_test_split(
    x_remaining,
    y_remaining,
    test_size = _DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE,
    shuffle = True)

print(f"[INFO]: Detected size of x training: {len(x_training)}")
assert len(x_training) == number_of_dnn_training_points, "[ASSERT]: Mismatch between expected x-training size and computed size."

print(f"[INFO]: Detected size of y training: {len(y_training)}")
assert len(y_training) == number_of_dnn_training_points, "[ASSERT]: Mismatch between expected y-training size and computed size."

print(f"[INFO]: Detected size of x validation: {len(x_validation)}")
assert len(x_validation) == number_of_dnn_validation_points, "[ASSERT]: Mismatch between expected x-validation size and computed size."

print(f"[INFO]: Detected size of y validation: {len(y_validation)}")
assert len(y_validation) == number_of_dnn_validation_points, "[ASSERT]: Mismatch between expected y-validation size and computed size."

print(f"[INFO]: Detected size of x testing: {len(x_testing)}")
assert len(x_testing) == number_of_dnn_testing_points, "[ASSERT]: Mismatch between expected x-testing size and computed size."

print(f"[INFO]: Detected size of x testing: {len(y_testing)}")
assert len(y_testing) == number_of_dnn_testing_points, "[ASSERT]: Mismatch between expected y-testing size and computed size."

# augment rows with train/test/val
pseudodata_dataframe.loc[x_training.index, 'split'] = 'train'
pseudodata_dataframe.loc[x_validation.index, 'split'] = 'validation'
pseudodata_dataframe.loc[x_testing.index, 'split'] = 'test'

pseudodata_dataframe.to_csv(
    path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/dnn_data_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.csv",
    index = False
)

####################################################################################################
# end the script
####################################################################################################

print("[INFO]: End of script reached!")
