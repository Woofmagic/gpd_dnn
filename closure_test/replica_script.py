#################################################################################
# FILE INFORMATION:
# Purpose: runs a single replica on pseudodata-generated observable data
# Created: 20260903
# Last changed: 20261005
#################################################################################

from pathlib import Path
import time
import gc
import yaml
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
import gepard as g
from gepard.fits import th_KM15
from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

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

from simultaneous_fit_dnn_config import UnfoldedSimultaneousFitLoss

print("[INFO]: Libraries imported!")

####################################################################################################
# matplotlib aesthetics
####################################################################################################

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "savefig.dpi": 300,
    "axes.labelsize": 16,
    "xtick.direction": "in",
    "xtick.major.size": 8.5,
    "xtick.major.width": 1.0,
    "xtick.minor.size": 4.5,
    "xtick.minor.width": 1.0,
    "xtick.minor.visible": True,
    "xtick.top": True,
    "ytick.direction": "in",
    "ytick.major.size": 8.5,
    "ytick.major.width": 1.0,
    "ytick.minor.size": 4.5,
    "ytick.minor.width": 1.0,
    "ytick.minor.visible": True,
    "ytick.right": True,
    "xtick.labelsize": 15.0,
    "ytick.labelsize": 15.0,
})

print("[INFO]: Customized Matplotlib settings!")

####################################################################################################
# observable keys
####################################################################################################

ALL_OBSERVABLE_KEYS = (
    "unp_beam_unp_target_xsec",
    "plus_beam_unp_target_xsec",
    "minus_beam_unp_target_xsec",

    "unp_beam_plus_target_xsec",
    "plus_beam_plus_target_xsec",
    "minus_beam_plus_target_xsec",

    "unp_beam_minus_target_xsec",
    "plus_beam_minus_target_xsec",
    "minus_beam_minus_target_xsec",

    "unp_target_bsa",
    "plus_target_bsa",
    "minus_target_bsa",

    "unp_beam_tsa",
    "plus_beam_tsa",
    "minus_beam_tsa",

    "dsa",
)

####################################################################################################
# library versions
####################################################################################################

print(f"[INFO]: numpy version: {np.__version__}")
print(f"[INFO]: pandas version: {pd.__version__}")
print(f"[INFO]: tensorflow version: {tf.__version__}")
print(f"[INFO]: gepard version: {g.__version__}")

####################################################################################################
# TensorFlow configurations
####################################################################################################

cpus = tf.config.list_physical_devices('CPU')
gpus = tf.config.list_physical_devices('GPU')

print(f"[INFO]: Number of CPUs Available: {len(cpus)}")
print(f"[INFO]: Number of GPUs Available: {len(gpus)}")
print(f"[INFO]: The available devices are: {tf.config.list_physical_devices()}")
print(f"[INFO]: Is CUDA-built? {tf.test.is_built_with_cuda()}")

####################################################################################################
# reading configuration file
####################################################################################################

with open(
    "closure_test_config.yml",
    "r",
    encoding = "utf-8") as file:
    config = yaml.safe_load(file)

####################################################################################################
# finding scratch path
####################################################################################################

SCRATCH_PATH = config["scratch_path"]
print(f"[INFO]: Computed scratch path: {SCRATCH_PATH}")

####################################################################################################
# determining the version number
####################################################################################################

MAJOR_NUMBER = config["versioning"]["major"]
MINOR_NUMBER = config["versioning"]["minor"]
MAJOR_MINOR_NUMBER = f"{MAJOR_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: Recieved major version number: {MAJOR_NUMBER}")
print(f"[INFO]: Recieved minor version number: {MINOR_NUMBER}")
print(f"[INFO]: Recieved total version number: {MAJOR_MINOR_NUMBER}")

####################################################################################################
# HPC arguments
####################################################################################################

replica_number = int(sys.argv[1])
print(f"[INFO]: Received replica number = {replica_number}")

####################################################################################################
# dynamically computing the paths to relevant directories and making them
####################################################################################################

version_directory = Path(SCRATCH_PATH) / f"version_{MAJOR_MINOR_NUMBER}"

data_directory = version_directory / "data"
plots_directory = version_directory / "plots"
replica_directory = version_directory / "replicas"
learning_curves_directory = version_directory / "learning_curves"

data_directory.mkdir(parents = True, exist_ok = True)
plots_directory.mkdir(parents = True, exist_ok = True)
learning_curves_directory.mkdir(parents = True, exist_ok = True)
replica_directory.mkdir(parents = True, exist_ok = True)

####################################################################################################
# DNN configuration
####################################################################################################

NUMBER_OF_EPOCHS = config["dnn_config"]["epochs"]
NUMBER_OF_REPLICAS = config["dnn_config"]["replicas"]
BATCH_SIZE = config["dnn_config"]["batch_size"]
LEARNING_RATE = config["dnn_config"]["adam_learning_rate"]

print(f"[INFO]: Received number of epochs (per replica): {NUMBER_OF_EPOCHS}")
print(f"[INFO]: Received number of replicas: {NUMBER_OF_REPLICAS}")
print(f"[INFO]: Received batch size: {BATCH_SIZE}")
print(f"[INFO]: Received (Adam) learning rate value: {LEARNING_RATE}")

####################################################################################################
# computing the number of free CFF fitting parameters:
####################################################################################################

IS_CFF_REAL_H_FREE = config["cff_config"]["enable_cff_real_h"]
print(f"[INFO]: Is Re[H] free to fit? {IS_CFF_REAL_H_FREE}")
IS_CFF_IMAG_H_FREE = config["cff_config"]["enable_cff_imag_h"]
print(f"[INFO]: Is Im[H] free to fit? {IS_CFF_IMAG_H_FREE}")
IS_CFF_REAL_HT_FREE = config["cff_config"]["enable_cff_real_ht"]
print(f"[INFO]: Is Re[Ht] free to fit? {IS_CFF_REAL_HT_FREE}")
IS_CFF_IMAG_HT_FREE = config["cff_config"]["enable_cff_imag_ht"]
print(f"[INFO]: Is Im[Ht] free to fit? {IS_CFF_IMAG_HT_FREE}")
IS_CFF_REAL_E_FREE = config["cff_config"]["enable_cff_real_e"]
print(f"[INFO]: Is Re[E] free to fit? {IS_CFF_REAL_E_FREE}")
IS_CFF_IMAG_E_FREE = config["cff_config"]["enable_cff_imag_e"]
print(f"[INFO]: Is Im[E] free to fit? {IS_CFF_IMAG_E_FREE}")
IS_CFF_REAL_ET_FREE = config["cff_config"]["enable_cff_real_et"]
print(f"[INFO]: Is Re[Et] free to fit? {IS_CFF_REAL_ET_FREE}")
IS_CFF_IMAG_ET_FREE = config["cff_config"]["enable_cff_imag_et"]
print(f"[INFO]: Is Im[Et] free to fit? {IS_CFF_IMAG_ET_FREE}")

IS_CFF_REAL_H_QUENCHED = config["cff_quenching"]["quench_cff_real_h"]
print(f"[INFO]: Is Re[H] quenched (i.e. fixed to 0 and not fit here)? {IS_CFF_REAL_H_QUENCHED}")
IS_CFF_IMAG_H_QUENCHED = config["cff_quenching"]["quench_cff_imag_h"]
print(f"[INFO]: Is Im[H] quenched (i.e. fixed to 0 and not fit here)? {IS_CFF_IMAG_H_QUENCHED}")
IS_CFF_REAL_HT_QUENCHED = config["cff_quenching"]["quench_cff_real_ht"]
print(f"[INFO]: Is Re[Ht] quenched (i.e. fixed to 0 and not fit here)? {IS_CFF_REAL_HT_QUENCHED}")
IS_CFF_IMAG_HT_QUENCHED = config["cff_quenching"]["quench_cff_imag_ht"]
print(f"[INFO]: Is Im[Ht] quenched (i.e. fixed to 0 and not fit here)? {IS_CFF_IMAG_HT_QUENCHED}")
IS_CFF_REAL_E_QUENCHED = config["cff_quenching"]["quench_cff_real_e"]
print(f"[INFO]: Is Re[E] quenched (i.e. fixed to 0 and not fit here)? {IS_CFF_REAL_E_QUENCHED}")
IS_CFF_IMAG_E_QUENCHED = config["cff_quenching"]["quench_cff_imag_e"]
print(f"[INFO]: Is Im[E] quenched (i.e. fixed to 0 and not fit here)? {IS_CFF_IMAG_E_QUENCHED}")
IS_CFF_REAL_ET_QUENCHED = config["cff_quenching"]["quench_cff_real_et"]
print(f"[INFO]: Is Re[Et] quenched (i.e. fixed to 0 and not fit here)? {IS_CFF_REAL_ET_QUENCHED}")
IS_CFF_IMAG_ET_QUENCHED = config["cff_quenching"]["quench_cff_imag_et"]
print(f"[INFO]: Is Im[Et] quenched (i.e. fixed to 0 and not fit here)? {IS_CFF_IMAG_ET_QUENCHED}")

####################################################################################################
# constructing "master list" of CFF fitting configuration:
####################################################################################################

CFF_CONFIG = {
    "cff_h_real": (
        config["cff_config"]["enable_cff_real_h"],
        config["cff_quenching"]["quench_cff_real_h"],
    ),
    "cff_h_imag": (
        config["cff_config"]["enable_cff_imag_h"],
        config["cff_quenching"]["quench_cff_imag_h"],
    ),
    "cff_ht_real": (
        config["cff_config"]["enable_cff_real_ht"],
        config["cff_quenching"]["quench_cff_real_ht"],
    ),
    "cff_ht_imag": (
        config["cff_config"]["enable_cff_imag_ht"],
        config["cff_quenching"]["quench_cff_imag_ht"],
    ),
    "cff_e_real": (
        config["cff_config"]["enable_cff_real_e"],
        config["cff_quenching"]["quench_cff_real_e"],
    ),
    "cff_e_imag": (
        config["cff_config"]["enable_cff_imag_e"],
        config["cff_quenching"]["quench_cff_imag_e"],
    ),
    "cff_et_real": (
        config["cff_config"]["enable_cff_real_et"],
        config["cff_quenching"]["quench_cff_real_et"],
    ),
    "cff_et_imag": (
        config["cff_config"]["enable_cff_imag_et"],
        config["cff_quenching"]["quench_cff_imag_et"],
    ),
}

# validating the configuration settings (checking consistency):
for cff_name, (is_free, is_quenched) in CFF_CONFIG.items():

    if is_free and is_quenched:
        raise RuntimeError(
            "[ERROR]: I discovered that your configuration file specifies that "
            f"{cff_name} to be both free and quenched. Check it for discrepancies."
        )

CFF_ORDER = (
    "cff_h_real",
    "cff_h_imag",
    "cff_ht_real",
    "cff_ht_imag",
    "cff_e_real",
    "cff_e_imag",
    "cff_et_real",
    "cff_et_imag",
)

FREE_CFF_NAMES = [
    cff_name for cff_name in CFF_ORDER if CFF_CONFIG[cff_name][0] and not CFF_CONFIG[cff_name][1]
]

# dynamically compute the number of free CFFs (parameters):
NUMBER_OF_FREE_CFFS = len(FREE_CFF_NAMES)

if NUMBER_OF_FREE_CFFS == 0:
    raise RuntimeError(
        "[ERROR]: What the hell are you doing? Quenching all 8 CFFS? Sounds like a "
        "boring fit to me!"
    )

####################################################################################################
# determining observable settings:
####################################################################################################
    
IS_UNP_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_unp_beam_unp_target_xsec"]
print(f"[INFO]: Are we fitting sigma(0, 0): {IS_UNP_BEAM_UNP_TARGET_XSEC_INCLUDED}")
IS_PLUS_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_plus_beam_unp_target_xsec"]
print(f"[INFO]: Are we fitting sigma(+1, 0): {IS_PLUS_BEAM_UNP_TARGET_XSEC_INCLUDED}")
IS_MINUS_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_minus_beam_unp_target_xsec"]
print(f"[INFO]: Are we fitting sigma(-1, 0): {IS_MINUS_BEAM_UNP_TARGET_XSEC_INCLUDED}")

IS_UNP_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_unp_beam_lp_target_xsec"]
print(f"[INFO]: Are we fitting sigma(0, +1/2): {IS_UNP_BEAM_LP_TARGET_XSEC_INCLUDED}")
IS_PLUS_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_plus_beam_lp_target_xsec"]
print(f"[INFO]: Are we fitting sigma(-1, +1/2): {IS_PLUS_BEAM_LP_TARGET_XSEC_INCLUDED}")
IS_MINUS_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_minus_beam_lp_target_xsec"]
print(f"[INFO]: Are we fitting sigma(-1, +1/2): {IS_MINUS_BEAM_LP_TARGET_XSEC_INCLUDED}")

IS_UNP_TARGET_BSA_INCLUDED = config["observable_config"]["enable_unp_target_bsa"]
print(f"[INFO]: Are we fitting BSA(0): {IS_UNP_TARGET_BSA_INCLUDED}")
IS_PLUS_TARGET_BSA_INCLUDED = config["observable_config"]["enable_plus_target_bsa"]
print(f"[INFO]: Are we fitting BSA(+1/2): {IS_PLUS_TARGET_BSA_INCLUDED}")
IS_MINUS_TARGET_BSA_INCLUDED = config["observable_config"]["enable_minus_target_bsa"]
print(f"[INFO]: Are we fitting BSA(-1/2): {IS_MINUS_TARGET_BSA_INCLUDED}")

IS_UNP_BEAM_TSA_INCLUDED = config["observable_config"]["enable_unp_beam_tsa"]
print(f"[INFO]: Are we fitting TSA(0): {IS_UNP_BEAM_TSA_INCLUDED}")
IS_PLUS_BEAM_TSA_INCLUDED = config["observable_config"]["enable_plus_beam_tsa"]
print(f"[INFO]: Are we fitting TSA(+1): {IS_PLUS_BEAM_TSA_INCLUDED}")
IS_MINUS_BEAM_TSA_INCLUDED = config["observable_config"]["enable_minus_beam_tsa"]
print(f"[INFO]: Are we fitting TSA(-1): {IS_MINUS_BEAM_TSA_INCLUDED}")

IS_DSA_INCLUDED = config["observable_config"]["enable_dsa"]
print(f"[INFO]: Are we fitting BSA: {IS_DSA_INCLUDED}")

enabled_observables = []

if IS_UNP_BEAM_UNP_TARGET_XSEC_INCLUDED:
    enabled_observables.append("unp_beam_unp_target_xsec")

if IS_PLUS_BEAM_UNP_TARGET_XSEC_INCLUDED:
    enabled_observables.append("plus_beam_unp_target_xsec")

if IS_MINUS_BEAM_UNP_TARGET_XSEC_INCLUDED:
    enabled_observables.append("minus_beam_unp_target_xsec")

if IS_UNP_BEAM_LP_TARGET_XSEC_INCLUDED:
    enabled_observables.append("unp_beam_lp_target_xsec")

if IS_PLUS_BEAM_LP_TARGET_XSEC_INCLUDED:
    enabled_observables.append("plus_beam_lp_target_xsec")

if IS_MINUS_BEAM_LP_TARGET_XSEC_INCLUDED:
    enabled_observables.append("minus_beam_lp_target_xsec")

if IS_UNP_TARGET_BSA_INCLUDED:
    enabled_observables.append("unp_target_bsa")

if IS_PLUS_TARGET_BSA_INCLUDED:
    enabled_observables.append("plus_lp_target_bsa")

if IS_MINUS_TARGET_BSA_INCLUDED:
    enabled_observables.append("minus_lp_target_bsa")

if IS_UNP_BEAM_TSA_INCLUDED:
    enabled_observables.append("unp_beam_tsa")

if IS_PLUS_BEAM_TSA_INCLUDED:
    enabled_observables.append("plus_beam_tsa")

if IS_MINUS_BEAM_TSA_INCLUDED:
    enabled_observables.append("minus_beam_tsa")

if IS_DSA_INCLUDED:
    enabled_observables.append("dsa")

####################################################################################################
# dynamically computing the loss weights:
####################################################################################################

OBSERVABLE_WEIGHTS = []

if len(enabled_observables) == 0:
    raise ValueError("[ERROR]: I didn't see that you enabled any observables to fit...")

if config["observable_config"]["equally_weighted_observables"]:
    observable_weight = 1.0 / len(enabled_observables)
    OBSERVABLE_WEIGHTS = [observable_weight] * len(enabled_observables)
    assert np.isclose(sum(OBSERVABLE_WEIGHTS), 1.0), "[ASSERT]: Something strange happened..."

else:
    raise NotImplementedError(
        "[ERROR]: Sorry! I haven't yet implemented inequal weighting of observables in this "
        "workflow... Stay tuned!"
    )

print(f"[INFO]: Dynamically computed the weights to be {OBSERVABLE_WEIGHTS}")

####################################################################################################
# set the kinematic bin here:
####################################################################################################

FIXED_K = 5.750
FIXED_XB = 0.360
FIXED_T = -0.17
FIXED_Q_SQUARED = 2.300

print(f"[INFO]: Received k = {FIXED_K} GeV")
print(f"[INFO]: Received xB = {FIXED_XB}")
print(f"[INFO]: Received t = {FIXED_T} GeV^2")
print(f"[INFO]: Received Q^2 = {FIXED_Q_SQUARED} GeV^2")

####################################################################################################
# set the beam polarizations here
####################################################################################################

TEST_LEPTON_HELICITY = 0.0
TEST_TARGET_POLARIZATION = 0.0

print(f"[INFO]: Lepton beam helicity is = {TEST_LEPTON_HELICITY}")
print(f"[INFO]: Target polarization is = {TEST_TARGET_POLARIZATION}")

####################################################################################################
# set the azimuthal distribution here
####################################################################################################

STARTING_PHI_VALUE_IN_DEGREES = config["data_config"]["start_value_of_phi_in_degrees"]
ENDING_PHI_VALUE_IN_DEGREES = config["data_config"]["end_value_of_phi_in_degrees"]
NUMBER_OF_PHI_POINTS = config["data_config"]["number_of_phi_points"] + 1

print(
    f"[INFO]: Phi will range from {STARTING_PHI_VALUE_IN_DEGREES} degrees "
    f"to {ENDING_PHI_VALUE_IN_DEGREES} degrees."
    ) 
print(f"[INFO]: Received the total number of phi points = {NUMBER_OF_PHI_POINTS}")

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)

phi_array_in_radians = [
    np.radians(degree_value) for degree_value in phi_array_in_degrees
    ]

print(
    f"[INFO]: New list of {len(phi_array_in_radians)} of azimuthal angles "
    f"from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees")

####################################################################################################
# obtaining the KM15 values for the given kinematic settings here
####################################################################################################

try:
    # [NOTE]: We actually don't need to be super accurate here because we ONLY use
    # this class to evaluate the CFFs later!
    test_datapoints = [
        g.DataPoint(
            xB = FIXED_XB,
            t = FIXED_T,
            Q2 = FIXED_Q_SQUARED,
            phi = fixed_phi,
            process = "ep2epgamma",
            exptype = 'fixed target',
            in1energy = FIXED_K,
            in1charge = -1,
            in1polarization = +1,
            in1units = 'rad',
            observable = 'XS',
            fname = 'Trento') for fixed_phi in phi_array_in_radians]
except ZeroDivisionError:
    print(
        f"[ERROR]: Kinematic setting k = {FIXED_K}, xb = {FIXED_XB}, "
        f"t = {FIXED_T}, Q^2 = {FIXED_Q_SQUARED} unphysical according to gepard."
        )

####################################################################################################
# basic assert
####################################################################################################

assert (
    len(test_datapoints) == len(phi_array_in_radians)
), "[ASSERT]: I think your Gepard + list comprehension did something strange."

####################################################################################################
# construct ("tesselated") arrays with the ground truth KM15:
####################################################################################################

real_h_values = np.array([th_KM15.ReH(datapoint) for datapoint in test_datapoints])
imag_h_values = np.array([th_KM15.ImH(datapoint) for datapoint in test_datapoints])
real_e_values = np.array([th_KM15.ReE(datapoint) for datapoint in test_datapoints])
imag_e_values = np.array([th_KM15.ImE(datapoint) for datapoint in test_datapoints])
real_ht_values = np.array([th_KM15.ReHt(datapoint) for datapoint in test_datapoints])
imag_ht_values = np.array([th_KM15.ImHt(datapoint) for datapoint in test_datapoints])
real_et_values = np.array([th_KM15.ReEt(datapoint) for datapoint in test_datapoints])
imag_et_values = np.array([th_KM15.ImEt(datapoint) for datapoint in test_datapoints])

####################################################################################################
# basic assert
####################################################################################################

assert (
    len(real_h_values) == len(test_datapoints)
), "[ASSERT]: Your Re[H] values are not as many as the datapoints you have..."
assert (
    len(imag_h_values) == len(test_datapoints)
), "[ASSERT]: Your Im[H] values are not as many as the datapoints you have..."
assert (
    len(real_e_values) == len(test_datapoints)
), "[ASSERT]: Your Re[E] values are not as many as the datapoints you have..."
assert (
    len(imag_e_values) == len(test_datapoints)
), "[ASSERT]: Your Im[E] values are not as many as the datapoints you have..."
assert (
    len(real_ht_values) == len(test_datapoints)
), "[ASSERT]: Your Re[Ht] values are not as many as the datapoints you have..."
assert (
    len(imag_ht_values) == len(test_datapoints)
), "[ASSERT]: Your Im[Ht] values are not as many as the datapoints you have..."
assert (
    len(real_et_values) == len(test_datapoints)
), "[ASSERT]: Your Re[Et] values are not as many as the datapoints you have..."
assert (
    len(imag_et_values) == len(test_datapoints)
), "[ASSERT]: Your Im[Et] values are not as many as the datapoints you have..."

####################################################################################################
# globally define/fix the KM15 values
####################################################################################################

# here we actually set the KM15 values to 0:
CFF_REAL_H_KM15 = real_h_values[0] if IS_CFF_REAL_H_FREE else 0.0
print(f"[INFO]: Setting Re[H] = {CFF_REAL_H_KM15}")
CFF_IMAG_H_KM15 = imag_h_values[0] if IS_CFF_IMAG_H_FREE else 0.0
print(f"[INFO]: Setting Im[H] = {CFF_IMAG_H_KM15}")
CFF_REAL_HT_KM15 = real_ht_values[0] if IS_CFF_REAL_HT_FREE else 0.0
print(f"[INFO]: Setting Re[Ht] = {CFF_REAL_HT_KM15}")
CFF_IMAG_HT_KM15 = imag_ht_values[0] if IS_CFF_IMAG_HT_FREE else 0.0
print(f"[INFO]: Setting Im[Ht] = {CFF_IMAG_HT_KM15}")
CFF_REAL_E_KM15 = real_e_values[0] if IS_CFF_REAL_E_FREE else 0.0
print(f"[INFO]: Setting Re[E] = {CFF_REAL_E_KM15}")
CFF_IMAG_E_KM15 = imag_e_values[0] if IS_CFF_IMAG_E_FREE else 0.0
print(f"[INFO]: Setting Im[E] = {CFF_IMAG_E_KM15}")
CFF_REAL_ET_KM15 = real_et_values[0] if IS_CFF_REAL_ET_FREE else 0.0
print(f"[INFO]: Setting Re[Et] = {CFF_REAL_ET_KM15}")
CFF_IMAG_ET_KM15 = imag_et_values[0] if IS_CFF_IMAG_ET_FREE else 0.0
print(f"[INFO]: Setting Im[Et] = {CFF_IMAG_ET_KM15}")

CFF_H_KM15 = complex(CFF_REAL_H_KM15, CFF_IMAG_H_KM15)
CFF_H_TILDE_KM15 = complex(CFF_REAL_HT_KM15, CFF_IMAG_HT_KM15)
CFF_E_KM15 = complex(CFF_REAL_E_KM15, CFF_IMAG_E_KM15)
CFF_E_TILDE_KM15 = complex(CFF_REAL_ET_KM15, CFF_IMAG_ET_KM15)

####################################################################################################
# KM15 cff settings string for plotting
####################################################################################################

km15_cff_string = (
    rf"$\mathcal{{H}} = {CFF_H_KM15:.3f}$, "
    rf"$\mathcal{{E}} = {CFF_E_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{H}}}} = {CFF_H_TILDE_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{E}}}} = {CFF_E_TILDE_KM15:.3f}$ "
)

####################################################################################################
# make the kinematics string for plotting
####################################################################################################

this_kinematic_set_title_string = (
    rf"$k = {FIXED_K:.3f}$ GeV, "
    rf"$x_B = {FIXED_XB:.3f}$, "
    rf"$t = {FIXED_T:.3f}$ GeV$^2$, "
    rf"$Q^2 = {FIXED_Q_SQUARED:.3f}$ GeV$^2$"
)

####################################################################################################
# computing the observables with bkm10 library
####################################################################################################

km15_cross_section = DifferentialCrossSection(
    configuration = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k = FIXED_K,
            squared_Q_momentum_transfer = FIXED_Q_SQUARED,
            x_Bjorken = FIXED_XB,
            squared_hadronic_momentum_transfer_t = FIXED_T),
        "cff_inputs": CFFInputs(
            compton_form_factor_h = CFF_H_KM15,
            compton_form_factor_h_tilde = CFF_H_TILDE_KM15,
            compton_form_factor_e = CFF_E_KM15,
            compton_form_factor_e_tilde = CFF_E_TILDE_KM15),
        "using_ww": True
    },
    verbose = False, debugging = False)

####################################################################################################
# computing the observables with bkm10 library
####################################################################################################

bkm10_unp_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = 0.0,
    target_polarization = 0.0).real

print("[INFO]: Used bkm10 library to compute sigma(0, 0)")

bkm10_plus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = +1.0,
    target_polarization = 0.0).real

print("[INFO]: Used bkm10 library to compute sigma(+1, 0)")

bkm10_minus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = -1.0,
    target_polarization = 0.0).real

print("[INFO]: Used bkm10 library to compute sigma(-1, 0)")

bkm10_unp_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = 0.0,
    target_polarization = +0.5).real

print("[INFO]: Used bkm10 library to compute sigma(0, +0.5)")

bkm10_plus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = +1.0,
    target_polarization = +0.5).real

print("[INFO]: Used bkm10 library to compute sigma(+1, +0.5)")

bkm10_minus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = -1.0,
    target_polarization = 0.5).real

print("[INFO]: Used bkm10 library to compute sigma(-1, +0.5)")

bkm10_bsa_unp_target_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = 0.0).real

print("[INFO]: Used bkm10 library to compute BSA(0)")

bkm10_bsa_plus_lp_target_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = +0.5).real

print("[INFO]: Used bkm10 library to compute BSA(+0.5)")

bkm10_bsa_minus_lp_target_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = -0.5).real

print("[INFO]: Used bkm10 library to compute BSA(-0.5)")

bkm10_unp_beam_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = 0.0).real
    
print("[INFO]: Used bkm10 library to compute TSA(0)")

bkm10_plus_beam_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = +1.0).real
    
print("[INFO]: Used bkm10 library to compute TSA(+1)")

bkm10_minus_beam_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = -1.0).real
    
print("[INFO]: Used bkm10 library to compute TSA(-1)")

bkm10_dsa_km15 = km15_cross_section.compute_dsa(phi_array_in_radians).real

print("[INFO]: Used bkm10 library to compute DSA")

####################################################################################################
# plotting bkm10(KM15)-predicted observables
####################################################################################################

def plot_bkm10_observable(
    x_data,
    y_data,
    y_label,
    title,
    observable_label,
    figure_filename):
    
    figure, axis = plt.subplots(
        ncols = 1,
        nrows = 1,
        figsize = (10, 10))

    axis.scatter(
        x_data,
        y_data,
        s = 4.0,
        color = "blue",
        label = observable_label)

    axis.set_xlabel(
        r"$\phi$ [radians]",
        fontsize = 20.)
    
    axis.set_ylabel(
        y_label,
        fontsize = 20.)
    
    axis.set_title(
        title,
        fontsize = 16.)

    axis.legend(
        fontsize = 20.)
    
    axis.grid(
        visible = True,
        alpha = 0.35)

    for extension in ["png", "eps"]:
        figure.savefig(
            plots_directory /
            f"{figure_filename}.{extension}",
            facecolor = "white"
        )

    plt.close(figure)

####################################################################################################
# actually making the plots of the BKM10-predicted observables
####################################################################################################

observables_to_plot = {
    "unp_beam_unp_target_xsec": {
        "data": bkm10_unp_beam_unp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{UU}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_unp_beam_unp_target_prediction",
    },
    "plus_beam_unp_target_xsec": {
        "data": bkm10_plus_beam_unp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{+U}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_plus_beam_unp_target_prediction",
    },
    "minus_beam_unp_target_xsec": {
        "data": bkm10_minus_beam_unp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{-U}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_minus_beam_unp_target_prediction",
    },
     "unp_beam_lp_target_xsec": {
        "data": bkm10_unp_beam_lp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{L}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_unp_beam_lp_target_prediction",
    },
    "plus_beam_lp_target_xsec": {
        "data": bkm10_plus_beam_lp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{L+}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_plus_beam_lp_target_prediction",
    },
    "minus_beam_lp_target_xsec": {
        "data": bkm10_minus_beam_lp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{L-}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_minus_beam_lp_target_prediction",
    },
    "unp_target_bsa": {
        "data": bkm10_bsa_unp_target_km15,
        "label": r"BKM10 $\textrm{BSA}(\Lambda = 0)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{BSA}(\Lambda = 0)$ [unitless]",
        "filename": "bkm10_bsa_unp_target_prediction",
    },
    "plus_lp_target_bsa": {
        "data": bkm10_bsa_plus_lp_target_km15,
        "label": r"BKM10 $\textrm{BSA}(\Lambda = +1/2)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{BSA}(\Lambda = +1/2)$ [unitless]",
        "filename": "bkm10_bsa_plus_lp_target_prediction",
    },
    "minus_lp_target_bsa": {
        "data": bkm10_bsa_minus_lp_target_km15,
        "label": r"BKM10 $\textrm{BSA}(\Lambda = -1/2)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{BSA}(\Lambda = -1/2)$ [unitless]",
        "filename": "bkm10_bsa_minus_lp_target_prediction",
    },
    "unp_beam_tsa": {
        "data": bkm10_unp_beam_tsa_km15,
        "label": r"BKM10 $\textrm{TSA}(\lambda = 0)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{TSA}(\lambda = 0)$ [unitless]",
        "filename": "bkm10_tsa_unp_beam_prediction",
    },
    "plus_beam_tsa": {
        "data": bkm10_plus_beam_tsa_km15,
        "label": r"BKM10 $\textrm{TSA}(\lambda = +1)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{TSA}(\lambda = +1)$ [unitless]",
        "filename": "bkm10_tsa_plus_beam_prediction",
    },
    "minus_beam_tsa": {
        "data": bkm10_minus_beam_tsa_km15,
        "label": r"BKM10 $\textrm{TSA}(\lambda = -1)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{TSA}(\lambda = -1)$ [unitless]",
        "filename": "bkm10_tsa_minus_beam_prediction",
    },
    "dsa": {
        "data": bkm10_dsa_km15,
        "label": r"BKM10 $\textrm{DSA}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{DSA}$ [unitless]",
        "filename": "bkm10_dsa_prediction",
    },
}

for observable_name, observable in observables_to_plot.items():
    
    print(f"[INFO]: Now making bkm10-predicted plot for {observable_name}")

    plot_bkm10_observable(
        phi_array_in_radians,
        observable["data"],
        y_label = observable["ylabel"],
        title = (
            rf"{observable_name} vs. $\phi$, "
            f"{this_kinematic_set_title_string}"    
            "\n"
            f"(KM15): {km15_cff_string}"
        ),
        observable_label = observable["label"],
        figure_filename = observable['filename']
    )

####################################################################################################
# training/testing split
####################################################################################################

# https://stackoverflow.com/a/13623707 -> for the typical train/val/test data split
# (we are *not* using the same numbers as recommended here)

TOTAL_DATA_SIZE = NUMBER_OF_PHI_POINTS

_DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE = 0.1 # 90% temporary, 10% testing
_DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE = 0.1 # of the above 90% temporary, 90% training, 10% validation

number_of_dnn_testing_points = np.ceil(TOTAL_DATA_SIZE * _DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE)
number_of_dnn_temporary_points = TOTAL_DATA_SIZE - number_of_dnn_testing_points
number_of_dnn_validation_points = np.ceil(number_of_dnn_temporary_points * _DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE)
number_of_dnn_training_points = number_of_dnn_temporary_points - number_of_dnn_validation_points

print(f"[NOTE]: Testing/Temporary Split is {_DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE * 100}%, giving {number_of_dnn_testing_points} testing points")
print(f"[NOTE]: Training/Validation Split is {_DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE * 100}%, giving {number_of_dnn_validation_points} validation points")
print(f"[NOTE]: Remaining training data points are: {number_of_dnn_training_points}")

####################################################################################################
# preparing DNN input tensors
####################################################################################################

k_tf_tensor = np.full(NUMBER_OF_PHI_POINTS, FIXED_K, dtype = np.float32)
t_tf_tensor = np.full(NUMBER_OF_PHI_POINTS, FIXED_T, dtype = np.float32)
xb_tf_tensor = np.full(NUMBER_OF_PHI_POINTS, FIXED_XB, dtype = np.float32)
q2_tf_tensor = np.full(NUMBER_OF_PHI_POINTS, FIXED_Q_SQUARED, dtype = np.float32)

phi_tf_tensor = np.array(phi_array_in_radians, dtype = np.float32)

fe_tf_tensor = compute_fe(t_tf_tensor)
fg_tf_tensor = compute_fg(fe_tf_tensor)
f2_tf_tensor = compute_f2(t_tf_tensor, fe_tf_tensor, fg_tf_tensor)
f1_tf_tensor = compute_f1(fg_tf_tensor, f2_tf_tensor)

epsilon_tf_tensor = compute_epsilon(xb_tf_tensor, q2_tf_tensor)
y_tf_tensor = compute_y(k_tf_tensor, q2_tf_tensor, epsilon_tf_tensor)
xi_tf_tensor = compute_skewness(xb_tf_tensor, t_tf_tensor, q2_tf_tensor)
t_min_tf_tensor = compute_t_min(xb_tf_tensor, q2_tf_tensor, epsilon_tf_tensor)
t_prime_tf_tensor = compute_t_prime(t_tf_tensor, t_min_tf_tensor)
k_tilde_tf_tensor = compute_k_tilde(xb_tf_tensor, q2_tf_tensor, t_tf_tensor, t_min_tf_tensor, epsilon_tf_tensor)
kinematic_k_tf_tensor = compute_k(q2_tf_tensor, y_tf_tensor, epsilon_tf_tensor, k_tilde_tf_tensor)

k_dot_delta_tf_tensor = compute_k_dot_delta(
    q2_tf_tensor, xb_tf_tensor, t_tf_tensor, phi_tf_tensor,
    epsilon_tf_tensor, y_tf_tensor, kinematic_k_tf_tensor)
prop_1_tf_tensor = prop_1(q2_tf_tensor, k_dot_delta_tf_tensor)
prop_2_tf_tensor = prop_2(q2_tf_tensor, t_tf_tensor, k_dot_delta_tf_tensor)

helicity_tf_tensor = np.full(NUMBER_OF_PHI_POINTS, 0.0, dtype = np.float32)
polarization_tf_tensor = np.full(NUMBER_OF_PHI_POINTS, 0.0, dtype = np.float32)

kinematics_and_phi = np.column_stack(
    (
        t_tf_tensor,
        xb_tf_tensor,
        q2_tf_tensor,
        phi_tf_tensor)
    ).astype(np.float32)

physics_data = np.column_stack((
    # kinematics, phi:
    t_tf_tensor,
    xb_tf_tensor,
    q2_tf_tensor,
    phi_tf_tensor,

    # form factors:
    fe_tf_tensor,
    fg_tf_tensor,
    f1_tf_tensor,
    f2_tf_tensor,

    # derived kinematics:
    epsilon_tf_tensor,
    y_tf_tensor,
    xi_tf_tensor,
    t_prime_tf_tensor,
    k_tilde_tf_tensor,
    kinematic_k_tf_tensor,

    # phi-dependent stuff:
    k_dot_delta_tf_tensor,
    prop_1_tf_tensor,
    prop_2_tf_tensor,

    # polarizations:
    helicity_tf_tensor, polarization_tf_tensor
)).astype(np.float32)

OBSERVABLE_DATA = {
    "unp_beam_unp_target_xsec": bkm10_unp_beam_unp_target_km15,
    "plus_beam_unp_target_xsec": bkm10_plus_beam_unp_target_km15,
    "minus_beam_unp_target_xsec": bkm10_minus_beam_unp_target_km15,
    "unp_beam_lp_target_xsec": bkm10_unp_beam_lp_target_km15,
    "plus_beam_lp_target_xsec": bkm10_plus_beam_lp_target_km15,
    "minus_beam_lp_target_xsec": bkm10_minus_beam_lp_target_km15,
    "unp_target_bsa": bkm10_bsa_unp_target_km15,
    "plus_lp_target_bsa": bkm10_bsa_plus_lp_target_km15,
    "minus_lp_target_bsa": bkm10_bsa_minus_lp_target_km15,
    "unp_beam_tsa": bkm10_unp_beam_tsa_km15,
    "plus_beam_tsa": bkm10_plus_beam_tsa_km15,
    "minus_beam_tsa": bkm10_minus_beam_tsa_km15,
    "dsa": bkm10_dsa_km15
}

missing_observables = [
    name for name in enabled_observables
    if name not in OBSERVABLE_DATA
]

if missing_observables:
    raise KeyError(
        "[ERROR]: The configuration requested observables that were not "
        f"available in OBSERVABLE_DATA: {missing_observables}"
    )

# stack the observables to fit:
observables_to_learn = np.column_stack((
    [OBSERVABLE_DATA[name] for name in enabled_observables]
)).astype(np.float32)

for index, (name, weight) in enumerate(zip(enabled_observables, OBSERVABLE_WEIGHTS)):
    print(
        f"[INFO]: observable {name} (index = {index}) has fitting weight of"
        f"= {weight:.4f}"
    )

print(f"[INFO]: Array of observables to learn: {observables_to_learn.shape}")

####################################################################################################
# constructing the training data
####################################################################################################

indices_array = np.arange(NUMBER_OF_PHI_POINTS)

dnn_inputs = np.column_stack((t_tf_tensor, xb_tf_tensor, q2_tf_tensor)).astype(np.float32)

training_indices, testing_indices = train_test_split(
    indices_array,
    test_size = 0.10,
    random_state = 7009,)

training_indices, validation_indices = train_test_split(
    training_indices,
    test_size = 0.10,
    random_state = 7009,)

splits = {
    "train": training_indices,
    "validation": validation_indices,
    "test": testing_indices,
}

x_training = dnn_inputs[training_indices]
x_validation = dnn_inputs[validation_indices]
x_testing = dnn_inputs[testing_indices]

precomputed_physics_training = physics_data[training_indices]
precomputed_physics_validation = physics_data[validation_indices]
precomputed_physics_testing = physics_data[testing_indices]

y_training = observables_to_learn[training_indices]
y_validation = observables_to_learn[validation_indices]
y_testing = observables_to_learn[testing_indices]

####################################################################################################
# a collection of asserts
####################################################################################################

print(f"[INFO]: Size of x training: {len(x_training)}")
print(f"[INFO]: Does it match with what's expected? {len(x_training) == number_of_dnn_training_points}")

print(f"[INFO]: Size of y training: {len(y_training)}")
print(f"[INFO]: Does it match with what's expected? {len(y_training) == number_of_dnn_training_points}")

print(f"[INFO]: Size of x validation: {len(x_validation)}")
print(f"[INFO]: Does it match with what's expected? {len(x_validation) == number_of_dnn_validation_points}")

print(f"[INFO]: Size of y validation: {len(y_validation)}")
print(f"[INFO]: Does it match with what's expected? {len(y_validation) == number_of_dnn_validation_points}")

print(f"[INFO]: Size of x testing: {len(x_testing)}")
print(f"[INFO]: Does it match with what's expected? {len(x_testing) == number_of_dnn_testing_points}")

print(f"[INFO]: Size of y testing: {len(y_testing)}")
print(f"[INFO]: Does it match with what's expected? {len(y_testing) == number_of_dnn_testing_points}")

####################################################################################################
# a function to see the train-test-validation splits:
####################################################################################################

def plot_train_validation_test_observable(
    phi,
    observable,
    split_indices,
    ylabel,
    title,
    figure_filename,):
    
    phi = np.asarray(phi)
    observable = np.asarray(observable)

    figure, axis = plt.subplots(
        nrows = 1,
        ncols = 1,
        figsize = (10, 10)
    )

    number_of_training_points = len(split_indices['train'])
    number_of_validation_points = len(split_indices['validation'])
    number_of_testing_points = len(split_indices['test'])

    axis.scatter(
        phi[split_indices["train"]],
        observable[split_indices["train"]],
        s = 4.0,
        label = rf"(KM15) Training Points ($N = {number_of_training_points}$)"
    )

    axis.scatter(
        phi[split_indices["validation"]],
        observable[split_indices["validation"]],
        s = 4.0,
        label = rf"(KM15) Validation Points ($N = {number_of_validation_points}$)"
    )

    axis.scatter(
        phi[split_indices["test"]],
        observable[split_indices["test"]],
        s = 4.0,
        label = rf"(KM15) Testing Points ($N = {number_of_testing_points}$)"
    )

    axis.legend(fontsize = 20.)

    axis.set_xlabel(
        r"$\phi$ [radians]",
        fontsize = 20.)

    axis.set_ylabel(
        ylabel,
        fontsize = 20.)

    axis.set_title(
        title,
        fontsize = 16.)

    axis.grid(
        visible = True,
        alpha = 0.35)

    for extension in ["png", "eps"]:
        figure.savefig(
            plots_directory /
            f"{figure_filename}_v{MAJOR_MINOR_NUMBER}.{extension}",
            facecolor = "white")

    plt.close(figure)

####################################################################################################
# actually make the observable split plots
####################################################################################################

for observable_name in enabled_observables:

    observable = OBSERVABLE_DATA[observable_name]
    plot_info = observables_to_plot[observable_name]

    plot_train_validation_test_observable(
        phi_array_in_radians,
        observable = observable,
        split_indices = splits,
        ylabel = plot_info["ylabel"],
        title = (
            rf"{observable_name} vs. $\phi$, "
            f"{this_kinematic_set_title_string}"
            "\n"
            f"(KM15): {km15_cff_string}"
        ),
        figure_filename = f"train_test_split_{plot_info['filename']}",
    )

####################################################################################################
# custom "quench layer"
####################################################################################################

@tf.keras.utils.register_keras_serializable(package = "assemble-cffs-layer")
class AssembleCFFs(tf.keras.layers.Layer):

    def __init__(self, free_cff_names, **kwargs):
        super().__init__(**kwargs)

        self.free_cff_names = tuple(free_cff_names)

        self.free_indices = {
            cff_name: index
            for index, cff_name in enumerate(self.free_cff_names)
        }

    def call(self, free_cffs):

        columns = []

        for cff_name in CFF_ORDER:

            if cff_name in self.free_indices:

                free_index = self.free_indices[cff_name]
                column = free_cffs[:, free_index:free_index + 1]

            else:

                column = tf.zeros_like(free_cffs[:, 0:1])

            columns.append(column)

        return tf.concat(columns, axis = 1)

    def get_config(self):

        config = super().get_config()

        config.update({
            "free_cff_names": self.free_cff_names,
        })

        return config

####################################################################################################
# actual DNN model
####################################################################################################

_NUMBER_OF_NODES_LAYER_1 = 128
_NUMBER_OF_NODES_LAYER_2 = 128
_NUMBER_OF_NODES_LAYER_3 = 128
_NUMBER_OF_NODES_LAYER_4 = 128

def cff_dnn_model():

    kinematics_inputs = tf.keras.Input(
        shape = (3,),
        name = "input_values")
    
    physics_input = tf.keras.Input(
        shape = (19,),
        name = "precomputed_physics")
    
    dnn_kinematic_inputs = tf.keras.layers.Lambda(
        lambda x: x[:, :3],
        name = "input_kinematics"
    )(kinematics_inputs)
    
    hidden = tf.keras.layers.Dense(
        _NUMBER_OF_NODES_LAYER_1,
        kernel_initializer = "he_normal",
        activation = "relu"
    )(dnn_kinematic_inputs)
    
    hidden = tf.keras.layers.Dense(
        _NUMBER_OF_NODES_LAYER_2,
        kernel_initializer = "he_normal",
        activation = "relu"
    )(hidden)
    
    hidden = tf.keras.layers.Dense(
        _NUMBER_OF_NODES_LAYER_3,
        kernel_initializer = "he_normal",
        activation = "relu"
    )(hidden)
    
    hidden = tf.keras.layers.Dense(
        _NUMBER_OF_NODES_LAYER_4,
        kernel_initializer = "he_normal",
        activation = "relu"
    )(hidden)
    
    # the output dimensionalty is dependent on this variable we
    # computed at the beginning...
    cff_free_outputs = tf.keras.layers.Dense(
        NUMBER_OF_FREE_CFFS,
        activation = "linear",
        name = "cff_free_outputs"
    )(hidden)

    cff_outputs = AssembleCFFs(
        FREE_CFF_NAMES,
        name = "cff_outputs"
    )(cff_free_outputs)

    full_model_outputs = tf.keras.layers.Concatenate(
        name = "physics_and_cffs")([
        cff_outputs,
        physics_input
    ])
    
    model = tf.keras.Model(
        inputs = [
            kinematics_inputs,
            physics_input
            ],
        outputs = full_model_outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = LEARNING_RATE
            ),
        loss = UnfoldedSimultaneousFitLoss(
            enabled_observables = enabled_observables,
            observable_weights = OBSERVABLE_WEIGHTS
        ),
        jit_compile = True,
        )

    return model
    
####################################################################################################
# DNN model fitting
####################################################################################################

tf.keras.backend.clear_session()
gc.collect()

start_dnn_compile_time = time.perf_counter()
dnn_model = cff_dnn_model()
print(
    f"[TIMING]: model construction time: "
    f"{time.perf_counter() - start_dnn_compile_time:.2f} s"
)

start_dnn_fitting_time = time.perf_counter()
dnn_model_history = dnn_model.fit(
    x = {
        "input_values": x_training,
        "precomputed_physics": precomputed_physics_training
    },
    y = y_training,
    validation_data = (
        {
            "input_values": x_validation,
            "precomputed_physics": precomputed_physics_validation
        },
        y_validation
    ),
    epochs = NUMBER_OF_EPOCHS,
    batch_size = BATCH_SIZE,
    verbose = 0
)
print(
    f"[TIMING]: fitting time: "
    f"{time.perf_counter() - start_dnn_fitting_time:.2f} s"
)

number_of_epochs_run = len(dnn_model_history.epoch)
print(f"[INFO]: The model ran for {number_of_epochs_run} epochs")

####################################################################################################
# DNN model saving
####################################################################################################

start_dnn_save_time = time.perf_counter()
dnn_model.save(
    replica_directory /
    f"replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras"
)
print(
    f"[TIMING]: DNN saving time: "
    f"{time.perf_counter() - start_dnn_save_time:.2f} s"
)

####################################################################################################
# replica predictions
####################################################################################################

def make_replica_predictions(
    dnn_model,
    dnn_inputs,
    physics_data,
    t,
    xb,
    q2,
    phi,):

    predicted_outputs = dnn_model.predict(
        {
            "input_values": dnn_inputs,
            "precomputed_physics": physics_data,
        },
        verbose = 0
    )

    # First eight outputs are the CFF predictions.
    cff_predictions = predicted_outputs[:, :8]

    cff_h_real = cff_predictions[:, 0]
    cff_h_imag = cff_predictions[:, 1]
    cff_ht_real = cff_predictions[:, 2]
    cff_ht_imag = cff_predictions[:, 3]
    cff_e_real = cff_predictions[:, 4]
    cff_e_imag = cff_predictions[:, 5]
    cff_et_real = cff_predictions[:, 6]
    cff_et_imag = cff_predictions[:, 7]

    fe = compute_fe(t)
    fg = compute_fg(fe)
    f2 = compute_f2(t, fe, fg)
    f1 = compute_f1(fg, f2)
    epsilon = compute_epsilon(xb, q2)
    y_lep = compute_y(FIXED_K, q2, epsilon)
    xi = compute_skewness(xb, t, q2)
    tmin = compute_t_min(xb, q2, epsilon)
    tprime = compute_t_prime(t, tmin)
    k_tilde = compute_k_tilde(xb, q2, t, tmin, epsilon)
    kinematic_k = compute_k(q2, y_lep, epsilon, k_tilde)
    kdd = compute_k_dot_delta(q2, xb, t, phi, epsilon, y_lep, kinematic_k)
    p1 = prop_1(q2, kdd)
    p2 = prop_2(q2, t, kdd)

    sigma_plus_plus = bkm10_cross_section(
        +1.0, +0.5,
        q2, xb, t, epsilon, y_lep, xi, kinematic_k, f1, f2, k_tilde, tprime, phi_tf_tensor, p1, p2,
        cff_h_real, cff_ht_real, cff_e_real, cff_et_real, cff_h_imag, cff_ht_imag, cff_e_imag, cff_et_imag)
    sigma_minus_plus = bkm10_cross_section(
        -1.0, +0.5,
        q2, xb, t, epsilon, y_lep, xi, kinematic_k, f1, f2, k_tilde, tprime, phi_tf_tensor, p1, p2,
        cff_h_real, cff_ht_real, cff_e_real, cff_et_real, cff_h_imag, cff_ht_imag, cff_e_imag, cff_et_imag)
    sigma_plus_minus = bkm10_cross_section(
        +1.0, -0.5,
        q2, xb, t, epsilon, y_lep, xi, kinematic_k, f1, f2, k_tilde, tprime, phi_tf_tensor, p1, p2,
        cff_h_real, cff_ht_real, cff_e_real, cff_et_real, cff_h_imag, cff_ht_imag, cff_e_imag, cff_et_imag)
    sigma_minus_minus = bkm10_cross_section(
        -1.0, -0.5,
        q2, xb, t, epsilon, y_lep, xi, kinematic_k, f1, f2, k_tilde, tprime, phi_tf_tensor, p1, p2,
        cff_h_real, cff_ht_real, cff_e_real, cff_et_real, cff_h_imag, cff_ht_imag, cff_e_imag, cff_et_imag)

    # cross-section | sigma(0, 0):
    sigma_unp_beam_unp_target = (0.25 * (
        sigma_plus_plus + sigma_plus_minus + sigma_minus_plus + sigma_minus_minus
    ))

    # cross-section | sigma(+1, 0):
    sigma_plus_beam_unp_target = (0.5 * (sigma_plus_plus + sigma_plus_minus))
    # cross-section | sigma(-1, 0):
    sigma_minus_beam_unp_target = (0.5 * (sigma_minus_plus + sigma_minus_minus))

    # cross-section | sigma(0, +1/2):
    sigma_unp_beam_plus_target = (0.5 * (sigma_plus_plus + sigma_minus_plus))
    # cross-section | sigma(0, -1/2):
    sigma_unp_beam_minus_target = (0.5 * (sigma_plus_minus + sigma_minus_minus))

    # cross-section | sigma(+1, +1/2):
    sigma_plus_beam_plus_target = sigma_plus_plus
    # cross-section | sigma(-1, +1/2):
    sigma_minus_beam_plus_target = sigma_minus_plus
    # cross-section | sigma(+1, -1/2):
    sigma_plus_beam_minus_target = sigma_plus_minus
    # cross-section | sigma(-1, +1/2):
    sigma_minus_beam_minus_target = sigma_minus_minus

    # sum of all sigma contributions:
    total_sigma = (
        sigma_plus_plus + sigma_plus_minus + sigma_minus_plus + sigma_minus_minus
    )

    # BSA(0)
    bsa_unp_target = ((
        sigma_plus_plus + sigma_plus_minus - sigma_minus_plus - sigma_minus_minus
    ) / total_sigma)

    # BSA(+1/2)
    bsa_plus_target = (sigma_plus_plus - sigma_minus_plus) / (sigma_plus_plus + sigma_minus_plus)

    # BSA(-1/2)
    bsa_minus_target = (sigma_plus_minus - sigma_minus_minus) / (sigma_plus_minus + sigma_minus_minus)

    # TSA(0)
    tsa_unp_beam = (
        (sigma_plus_plus + sigma_minus_plus - sigma_plus_minus - sigma_minus_minus) / 
        total_sigma)

    # TSA(+1)
    tsa_plus_beam = (
        (sigma_plus_plus - sigma_plus_minus) /
        (sigma_plus_plus + sigma_plus_minus))

    # TSA(-1):
    tsa_minus_beam = (
        (sigma_minus_plus - sigma_minus_minus) /
        (sigma_minus_plus + sigma_minus_minus))

    # DSA
    dsa = ((
        sigma_plus_plus - sigma_plus_minus- sigma_minus_plus + sigma_minus_minus
    ) / total_sigma)

    # here, we compute ALL observables possible:
    observable_predictions = {
        "unp_beam_unp_target_xsec": sigma_unp_beam_unp_target,
        "plus_beam_unp_target_xsec": sigma_plus_beam_unp_target,
        "minus_beam_unp_target_xsec": sigma_minus_beam_unp_target,
        "unp_beam_plus_target_xsec": sigma_unp_beam_plus_target,
        "plus_beam_plus_target_xsec": sigma_plus_beam_plus_target,
        "minus_beam_plus_target_xsec": sigma_minus_beam_plus_target,
        "unp_beam_minus_target_xsec": sigma_unp_beam_minus_target,
        "plus_beam_minus_target_xsec": sigma_plus_beam_minus_target,
        "minus_beam_minus_target_xsec": sigma_minus_beam_minus_target,
        "unp_target_bsa": bsa_unp_target,
        "plus_target_bsa": bsa_plus_target,
        "minus_target_bsa": bsa_minus_target,
        "unp_beam_tsa": tsa_unp_beam,
        "plus_beam_tsa": tsa_plus_beam,
        "minus_beam_tsa": tsa_minus_beam,
        "dsa": dsa,
    }
        
    if set(observable_predictions.keys()) != set(ALL_OBSERVABLE_KEYS):
        raise RuntimeError(
            "[ERROR]: The observable prediction dictionary does not "
            "contain exactly the expected canonical observable keys."
        )

    if not set(enabled_observables).issubset(
        observable_predictions.keys()
    ):
        raise RuntimeError(
            "[ERROR]: At least one enabled observable has no "
            "corresponding prediction."
        )
        
    np.savez(
        # filename:
        file = 
            (
                replica_directory / 
                f"replica_{replica_number}_predictions_v{MAJOR_MINOR_NUMBER}.npz"
            ),

        # kinematics:
        t = t_tf_tensor,
        xb = xb_tf_tensor,
        q_squared = q2_tf_tensor,
        phi = phi_tf_tensor,

        # save CFF data:
        cff_h_real = cff_h_real,
        cff_h_imag = cff_h_imag,
        cff_ht_real = cff_ht_real,
        cff_ht_imag = cff_ht_imag,
        cff_e_real = cff_e_real,
        cff_e_imag = cff_e_imag,
        cff_et_real = cff_et_real,
        cff_et_imag = cff_et_imag,

        # four "fundamental" cross-sections:
        sigma_plus_plus = sigma_plus_plus,
        sigma_minus_plus = sigma_minus_plus,
        sigma_plus_minus = sigma_plus_minus,
        sigma_minus_minus = sigma_minus_minus,

        # observables that the DNN can in principle predict
        **observable_predictions,

        enabled_observables = np.asarray(
            enabled_observables,
            dtype = "U",
        ),

        observable_weights = np.asarray(
            OBSERVABLE_WEIGHTS,
            dtype = np.float64,
        ),
    )

    return {
        "cff_predictions": cff_predictions,
        "observable_predictions": observable_predictions,
    }
    
start_replica_prediction_time = time.perf_counter()
replica_predictions = make_replica_predictions(
    dnn_model = dnn_model,
    dnn_inputs = dnn_inputs,
    physics_data = physics_data,
    t = t_tf_tensor,
    xb = xb_tf_tensor,
    q2 = q2_tf_tensor,
    phi = phi_tf_tensor,
)
print(
    f"[TIMING]: replica prediction time: "
    f"{time.perf_counter() - start_replica_prediction_time:.2f} s"
)

cff_predictions = replica_predictions["cff_predictions"]
observable_predictions = replica_predictions["observable_predictions"]

####################################################################################################
# DNN model history data
####################################################################################################

training_loss_data = np.asarray(dnn_model_history.history["loss"])
validation_loss_data = np.asarray(dnn_model_history.history["val_loss"])

####################################################################################################
# DNN model evaluation
####################################################################################################

start_dnn_evaluation_time = time.perf_counter()

dnn_evaluation_statistics = dnn_model.evaluate(
    x = {
        "input_values": x_testing,
        "precomputed_physics": precomputed_physics_testing,
    },
    y = y_testing,
    verbose = 0
)

print(
    f"[TIMING]: DNN evaluation time: "
    f"{time.perf_counter() - start_dnn_evaluation_time:.2f} s"
)

print(
    f"[INFO]: Test Loss for Replica "
    f"{replica_number}: {dnn_evaluation_statistics}"
)

####################################################################################################
# DNN model loss data
####################################################################################################

testing_loss_information = pd.DataFrame({
    "testing_loss": [dnn_evaluation_statistics],
})

testing_loss_information.to_csv(
    replica_directory /
    f"replica_{replica_number}_loss_data_v{MAJOR_MINOR_NUMBER}.csv",
    index = False
)

####################################################################################################
# DNN losses versus epoch
####################################################################################################

# just do this for now hahahaha:
_REGULARIZER = 1e-21

np.savez(
    file = (
        replica_directory /
        f"replica_{replica_number}_losses_vs_epochs_v{MAJOR_MINOR_NUMBER}.npz"
    ),
    training_loss = training_loss_data,
    validation_loss = validation_loss_data
)

start_learning_curves_time = time.perf_counter()

epoch_array = np.arange(number_of_epochs_run)

initial_loss_value = training_loss_data[0]
testing_loss = testing_loss_information["testing_loss"].iloc[0]

curves_fig, curves_ax = plt.subplots(
    nrows = 1,
    ncols = 1,
    figsize = (10, 10))

curves_ax.axhline(
    initial_loss_value,
    color = "red",
    linestyle = "--",
    label = "Initial Loss Value"
)

curves_ax.axhline(
    0.0,
    color = "green",
    linestyle = "--",
    label = r"Loss $= 0$"
)

curves_ax.plot(
    epoch_array,
    training_loss_data,
    color = "blue",
    label = "Training Loss")

curves_ax.plot(
    epoch_array,
    validation_loss_data,
    color = "purple",
    label = "Validation Loss")

curves_ax.legend(fontsize = 16.)

curves_ax.set_xlabel(
    "Epoch",
    fontsize = 20.)

curves_ax.set_ylabel(
    "MSE",
    fontsize = 20.)

curves_ax.set_title(
    f"Replica {replica_number} Learning Curves\n(Test Loss $= {testing_loss:.3g}$)",
    fontsize = 20.)

curves_fig.tight_layout()

curves_png_path = (
    learning_curves_directory
    / f"lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.png"
)

curves_eps_path = (
    learning_curves_directory
    / f"lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.eps"
)

curves_fig.savefig(curves_png_path)
curves_fig.savefig(curves_eps_path)

log_curves_fig, log_curves_ax = plt.subplots(
    nrows = 1,
    ncols = 1,
    figsize = (10, 10))

log_curves_ax.plot(
    epoch_array,
    np.log(training_loss_data + _REGULARIZER),
    color = "blue",
    label = "Log Training Loss")

log_curves_ax.plot(
    epoch_array,
    np.log(validation_loss_data + _REGULARIZER),
    color = "purple",
    label = "Log Validation Loss")

log_curves_ax.legend(fontsize = 16.)

log_curves_ax.set_xlabel(
    "Epoch", 
    fontsize = 20.)

log_curves_ax.set_ylabel(
    "Log MSE Loss", 
    fontsize = 20.)

log_curves_ax.set_title(
    f"Replica {replica_number} Learning Curves\n(Test Loss $= {testing_loss:.3g}$)",
    fontsize = 20.)
        
log_curves_fig.tight_layout()

log_curves_png_path = (
    learning_curves_directory
    / f"log_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.png"
)

log_curves_eps_path = (
    learning_curves_directory
    / f"log_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.eps"
)

log_curves_fig.savefig(log_curves_png_path)
log_curves_fig.savefig(log_curves_eps_path)

plt.close(curves_fig)
plt.close(log_curves_fig)

print(
    f"[TIMING]: learning curves saving time: "
    f"{time.perf_counter() - start_learning_curves_time:.2f} s"
)

####################################################################################################
# Cleanup!
####################################################################################################

del dnn_model
del dnn_model_history
del curves_fig
del log_curves_fig

tf.keras.backend.clear_session()
gc.collect()

print(f"[INFO]: Script finished!")
