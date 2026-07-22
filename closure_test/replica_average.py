#################################################################################
# FILE INFORMATION:
# Purpose: averages over all the replicas
# Created: 20260107
# Last changed: 20260722
#################################################################################

print("[INFO]: Script began running!")

####################################################################################################
# Importing Python Libraries
####################################################################################################

import yaml
import sys
import glob
import gc

import pandas as pd
import numpy as np
import tensorflow as tf

from simultaneous_fit_dnn_config import compute_fe
from simultaneous_fit_dnn_config import compute_fg
from simultaneous_fit_dnn_config import compute_f2
from simultaneous_fit_dnn_config import compute_f1
from simultaneous_fit_dnn_config import compute_epsilon
from simultaneous_fit_dnn_config import compute_y
from simultaneous_fit_dnn_config import compute_skewness
from simultaneous_fit_dnn_config import compute_t_min
from simultaneous_fit_dnn_config import compute_t_prime
from simultaneous_fit_dnn_config import compute_k_tilde
from simultaneous_fit_dnn_config import compute_k
from simultaneous_fit_dnn_config import compute_k_dot_delta
from simultaneous_fit_dnn_config import prop_1
from simultaneous_fit_dnn_config import prop_2

from simultaneous_fit_dnn_config import bkm10_cross_section
from simultaneous_fit_dnn_config import bkm10_bsa

print("[INFO]: Libraries imported!")

#################################################################################
# [IMPORTANT]: Static quantities parametrizing the program. Change these if you need!
#################################################################################

# verify this is what you want
SCRATCH_PATH = 'placeholder!'

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

MAJOR_NUMBER = config["versioning"]["major"]
MINOR_NUMBER = config["versioning"]["minor"]
MAJOR_MINOR_NUMBER = f"{MAJOR_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

TEST_LEPTON_HELICITY = 0.0
TEST_TARGET_POLARIZATION = 0.0

print(f"[INFO]: Detected lepton helicity to be: {'unpolarized' if TEST_LEPTON_HELICITY == 0.0 else 'polarized'}")

#################################################################################
# Tensorflow Configuration
#################################################################################

print(f"[INFO]: Physical devices available to TF are: {tf.config.list_physical_devices()}")
print(f"[INFO]: Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"[INFO]: Number of CPUs Available: {len(tf.config.list_physical_devices('CPU'))}")
print(f"[INFO]: Checking the GPU name: {tf.test.gpu_device_name()}")

#################################################################################
# Reading the Datafile:
#################################################################################

slurm_array_parameter = int(sys.argv[1])

try:
    with open(
        f"./slurm_logs/version_{MAJOR_MINOR_NUMBER}/valid_kinematic_sets_v{MAJOR_MINOR_NUMBER}.txt", 
        "r",
        encoding = "utf-8") as good_kinematic_sets_file:
        # This creates a list like [2, 9, 15, ...]
        valid_kinematic_sets = [ int(line.strip()) for line in good_kinematic_sets_file if line.strip() ]
except FileNotFoundError:
    print(f"[ERROR]: Could not find slurm_logs/version_{MAJOR_MINOR_NUMBER}!")
    sys.exit(0)
    
print(f"[INFO]: Good kinematic sets were: {valid_kinematic_sets}")

try:
    # this transforms the --array parameter into the actual kinematic set number:
    kinematic_set_number = valid_kinematic_sets[slurm_array_parameter - 1]
    print(f"[INFO]: Slurm ID {slurm_array_parameter} mapped to Kinematic Set {kinematic_set_number}")
except IndexError:
    print(f"[ERROR]: Slurm Array ID {slurm_array_parameter} is out of bounds for the valid sets list.")
    sys.exit(0)

#################################################################################
# load the replicas
#################################################################################

# find replicas:
replica_paths = sorted(
    glob.glob(
        f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/replicas/replica_*_v{MAJOR_MINOR_NUMBER}.keras"))

print(f"[INFO]: Loaded {len(replica_paths)} replica models.")

# find every replica's unique datafile:
replica_data_paths = sorted(
    glob.glob(
        f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/dnn_data_replica_*_v{MAJOR_MINOR_NUMBER}.csv"
        )
    )

if len(replica_paths) == 0:
    print("[WARN]: No replicas found for this kinematic setting. Exiting...")
    sys.exit(0)

#################################################################################
# IMPORTANT: use only *one* replica datafile to just get the constant numbers!
#################################################################################

first_replica_df = pd.read_csv(replica_data_paths[0])

FIXED_BEAM_ENERGY = first_replica_df["k"].iloc[0]
FIXED_Q_SQUARED = first_replica_df["q_squared"].iloc[0]
FIXED_X_BJORKEN = first_replica_df["x_b"].iloc[0]
FIXED_T_VALUE = first_replica_df["t"].iloc[0]

k_values = first_replica_df["k"].values
t_values = first_replica_df["t"].values
xb_values = first_replica_df["x_b"].values
q2_values = first_replica_df["q_squared"].values
phi_values = first_replica_df["phi"].values

print(f"[INFO]: Selected beam energy k = {FIXED_BEAM_ENERGY}")
print(f"[INFO]: Selected Q^2 = {FIXED_Q_SQUARED}")
print(f"[INFO]: Selected xB = {FIXED_X_BJORKEN}")
print(f"[INFO]: Selected t = {FIXED_T_VALUE}")

CFF_REAL_H_KM15 = first_replica_df["Re[H]"].values
CFF_IMAG_H_KM15 = first_replica_df["Im[H]"].values
CFF_REAL_E_KM15 = first_replica_df["Re[E]"].values
CFF_IMAG_E_KM15 = first_replica_df["Im[E]"].values
CFF_REAL_HT_KM15 = first_replica_df["Re[Ht]"].values
CFF_IMAG_HT_KM15 = first_replica_df["Im[Ht]"].values
CFF_REAL_ET_KM15 = first_replica_df["Re[Et]"].values
CFF_IMAG_ET_KM15 = first_replica_df["Im[Et]"].values

#################################################################################
# actually load the replicas:
#################################################################################

replicas = [tf.keras.models.load_model(
    path,
    compile = False,
    safe_mode = False) for path in replica_paths]

print(f"[INFO]: Loaded {len(replicas)} replica models.")

#################################################################################
# make the actual predictions:
#################################################################################

all_predictions = []
replicas_cross_predictions = []
replicas_bsa_predictions = []

for index, replica in enumerate(replicas):
    current_csv_path = replica_data_paths[index]
    replica_df = pd.read_csv(current_csv_path)
    x_data = replica_df[["t", "x_b", "q_squared", "phi"]].values.astype(np.float64)
    # predicting using x_data
    predicted_outputs = replica.predict(x_data, verbose = 0)
    all_predictions.append(predicted_outputs)

    del replica_df
    del x_data
    del predicted_outputs

all_predictions = np.array(all_predictions)

# initialize all CFF prediction arrays even though we might not use all..
replicas_h_real = []
replicas_h_imag = []
replicas_ht_real = []
replicas_ht_imag = []
replicas_e_real = []
replicas_e_imag = []
replicas_et_real = []
replicas_et_imag = []

for index, _ in enumerate(all_predictions):

    cff_h_real = all_predictions[index][:, 0] # getting Re[H]
    cff_h_imag = all_predictions[index][:, 1] # getting Im[H]
    cff_ht_real = all_predictions[index][:, 2] # getting Re[Ht]
    cff_ht_imag = all_predictions[index][:, 3] # getting Im[Ht]

    t = all_predictions[index][:, 4] # getting t
    xb = all_predictions[index][:, 5] # getting xb
    q_squared = all_predictions[index][:, 6] # getting Q^{2}
    phi = all_predictions[index][:, 7]

    fe = compute_fe(t)
    fg = compute_fg(fe)
    f2 = compute_f2(t, fe, fg)
    f1 = compute_f1(fg, f2)
    
    epsilon = compute_epsilon(xb, q_squared)
    y_lep = compute_y(FIXED_BEAM_ENERGY, q_squared, epsilon)
    xi = compute_skewness(xb, t, q_squared)
    tmin = compute_t_min(xb, q_squared, epsilon)
    tprime = compute_t_prime(t, tmin) # used in interference only
    ktilde = compute_k_tilde(xb, q_squared, t, tmin, epsilon)
    k = compute_k(q_squared, y_lep, epsilon, ktilde)
    kdd = compute_k_dot_delta(q_squared, xb, t, phi, epsilon, y_lep, k)
    p1 = prop_1(q_squared, kdd)
    p2 = prop_2(q_squared, t, kdd)

    cross_section = bkm10_cross_section(
        TEST_LEPTON_HELICITY, TEST_TARGET_POLARIZATION,
        q_squared, xb, t, epsilon, y_lep, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_h_real, cff_ht_real, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag, cff_ht_imag, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)

    predicted_bsa = bkm10_bsa(
        TEST_TARGET_POLARIZATION,
        q_squared, xb, t, epsilon, y_lep, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_h_real, cff_ht_real, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag, cff_ht_imag, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)

    replicas_h_real.append(cff_h_real)
    replicas_h_imag.append(cff_h_imag)
    replicas_ht_real.append(cff_ht_real)
    replicas_ht_imag.append(cff_ht_imag)

    replicas_cross_predictions.append(cross_section)
    replicas_bsa_predictions.append(predicted_bsa)

replicas_cross_predictions = np.array(replicas_cross_predictions)
replicas_bsa_predictions = np.array(replicas_bsa_predictions)

def crunch_statistics(data):

    # huge dictionary of statistics
    statistics_dictionary = {
        'mean': np.mean(data, axis = 0),
        'std': np.std(data, axis = 0),
        'median': np.median(data, axis = 0),
        'min': np.min(data, axis = 0),
        'max': np.max(data, axis = 0)
    }

    # adding keys to the dictionary:
    for percentile in range(10, 50, 10):
        statistics_dictionary[f'p{percentile}'] = np.percentile(data, percentile, axis = 0)
        statistics_dictionary[f'p{100 - percentile}'] = np.percentile(data, 100 - percentile, axis = 0)

    return statistics_dictionary

# observable statistics:
xs_stats = crunch_statistics(replicas_cross_predictions)
bsa_stats = crunch_statistics(replicas_bsa_predictions)

replica_statistics_dataframe = pd.DataFrame({
    'k': k_values,
    't': t_values,
    'xb': xb_values,
    'q_squared': q2_values,
    'phi': phi_values,

    # TRUE CFF VALUES:
    "Re[H]": CFF_REAL_H_KM15, "Im[H]": CFF_IMAG_H_KM15,
    "Re[E]": CFF_REAL_E_KM15, "Im[E]": CFF_IMAG_E_KM15,
    "Re[Ht]": CFF_REAL_HT_KM15, "Im[Ht]": CFF_IMAG_HT_KM15,
    "Re[Et]": CFF_REAL_ET_KM15, "Im[Et]": CFF_IMAG_ET_KM15,

    # cross-section
    'mean_xs': xs_stats['mean'],
    'std_xs': xs_stats['std'],
    'min_xs': xs_stats['min'],
    'max_xs': xs_stats['max'],
    'p10_xs': xs_stats['p10'], 'p20_xs': xs_stats['p20'], 'p30_xs': xs_stats['p30'], 'p40_xs': xs_stats['p40'],
    'p60_xs': xs_stats['p60'], 'p70_xs': xs_stats['p70'], 'p80_xs': xs_stats['p80'], 'p90_xs': xs_stats['p90'],
    
    # BSA
    'mean_bsa': bsa_stats['mean'],
    'std_bsa': bsa_stats['std'],
    'min_bsa': bsa_stats['min'],
    'max_bsa': bsa_stats['max'],
    'p10_bsa': bsa_stats['p10'], 'p20_bsa': bsa_stats['p20'], 'p30_bsa': bsa_stats['p30'], 'p40_bsa': bsa_stats['p40'],
    'p60_bsa': bsa_stats['p60'], 'p70_bsa': bsa_stats['p70'], 'p80_bsa': bsa_stats['p80'], 'p90_bsa': bsa_stats['p90']
})

replica_statistics_dataframe.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/observable_preds_v{MAJOR_MINOR_NUMBER}.csv", 
    index = False)

# CFF statistics:
cff_h_real_pred_per_replica = np.mean(all_predictions[:, :, 0], axis = 1)
cff_h_imag_pred_per_replica = np.mean(all_predictions[:, :, 1], axis = 1)
cff_ht_real_pred_per_replica = np.mean(all_predictions[:, :, 2], axis = 1)
cff_ht_imag_pred_per_replica = np.mean(all_predictions[:, :, 3], axis = 1)

replica_cffs_dataframe = pd.DataFrame({
    'ReH_pred': cff_h_real_pred_per_replica,
    'ImH_pred': cff_h_imag_pred_per_replica,
    'ReHt_pred': cff_ht_real_pred_per_replica,
    'ImHt_pred': cff_ht_imag_pred_per_replica
})

replica_cffs_dataframe.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/cff_replica_average_preds_v{MAJOR_MINOR_NUMBER}.csv", 
    index = False)

tf.keras.backend.clear_session()
gc.collect()

####################################################################################################
# end the script
####################################################################################################

print("[INFO]: End of script reached!")
