
import glob
import gc
import datetime
import yaml
import re
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import corner
from scipy.stats import norm
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

from unfolded_loss import UnfoldedSimultaneousFitLoss
from simultaneous_fit_dnn_config import bkm10_cross_section
from simultaneous_fit_dnn_config import bkm10_bsa

print(f"[INFO]: numpy version: {np.__version__}")
print(f"[INFO]: pandas version: {pd.__version__}")
print(f"[INFO]: tensorflow version: {tf.__version__}")
print(f"[INFO]: gepard version: {g.__version__}")
print(f"[INFO]: corner version: {corner.__version__}")

print(f"[INFO]: Libraries imported!")

plt.rcParams.update({
    "text.usetex": True,
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

with open("closure_test_config.yml", "r") as file:
    config = yaml.safe_load(file)

MAJOR_NUMBER = config["versioning"]["major"]
MINOR_NUMBER = config["versioning"]["minor"]
MAJOR_MINOR_NUMBER = f"{MAJOR_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: Recieved major version number: {MAJOR_NUMBER}")
print(f"[INFO]: Recieved minor version number: {MINOR_NUMBER}")
print(f"[INFO]: Recieved total version number: {MAJOR_MINOR_NUMBER}")

NUMBER_OF_EPOCHS = config["dnn_config"]["epochs"]
NUMBER_OF_REPLICAS = config["dnn_config"]["replicas"]
BATCH_SIZE = config["dnn_config"]["batch_size"]
LEARNING_RATE = config["dnn_config"]["adam_learning_rate"]

print(f"[INFO]: Received number of epochs (per replica): {NUMBER_OF_EPOCHS}")
print(f"[INFO]: Received number of replicas: {NUMBER_OF_REPLICAS}")
print(f"[INFO]: Received batch size: {BATCH_SIZE}")
print(f"[INFO]: Received (Adam) learning rate value: {LEARNING_RATE}")

IS_CFF_REAL_H_FREE = config["cff_config"]["enable_cff_real_h"]
IS_CFF_IMAG_H_FREE = config["cff_config"]["enable_cff_imag_h"]
IS_CFF_REAL_HT_FREE = config["cff_config"]["enable_cff_real_ht"]
IS_CFF_IMAG_HT_FREE = config["cff_config"]["enable_cff_imag_ht"]
IS_CFF_REAL_E_FREE = config["cff_config"]["enable_cff_real_e"]
IS_CFF_IMAG_E_FREE = config["cff_config"]["enable_cff_imag_e"]
IS_CFF_REAL_ET_FREE = config["cff_config"]["enable_cff_real_et"]
IS_CFF_IMAG_ET_FREE = config["cff_config"]["enable_cff_imag_et"]

# cross-section observables:
IS_UNP_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_unp_beam_unp_target_xsec"]
IS_PLUS_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_plus_beam_unp_target_xsec"]
IS_MINUS_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_minus_beam_unp_target_xsec"]
IS_UNP_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_unp_beam_lp_target_xsec"]
IS_PLUS_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_plus_beam_lp_target_xsec"]
IS_MINUS_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_minus_beam_lp_target_xsec"]

IS_UNP_TARGET_BSA_INCLUDED = config["observable_config"]["enable_unp_target_bsa"]
IS_PLUS_TARGET_BSA_INCLUDED = config["observable_config"]["enable_plus_target_bsa"]
IS_MINUS_TARGET_BSA_INCLUDED = config["observable_config"]["enable_minus_target_bsa"]

IS_UNP_BEAM_TSA_INCLUDED = config["observable_config"]["enable_unp_beam_tsa"]
IS_PLUS_BEAM_TSA_INCLUDED = config["observable_config"]["enable_plus_beam_tsa"]
IS_MINUS_BEAM_TSA_INCLUDED = config["observable_config"]["enable_minus_beam_tsa"]

IS_DSA_INCLUDED = config["observable_config"]["enable_dsa"]

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

OBSERVABLE_WEIGHTS = [
    0.5,
    0.5,
]

base_directory = Path('./local') / f"version_{MAJOR_MINOR_NUMBER}"

data_directory = base_directory / "data"
plots_directory = base_directory / "plots"
replica_directory = base_directory / "replica"
learning_curves_directory = base_directory / "learning_curves"

data_directory.mkdir(parents = True, exist_ok = True)
plots_directory.mkdir(parents = True, exist_ok = True)
replica_directory.mkdir(parents = True, exist_ok = True)
learning_curves_directory.mkdir(parents = True, exist_ok = True)

STARTING_PHI_VALUE_IN_DEGREES = config["data_config"]["start_value_of_phi_in_degrees"]
ENDING_PHI_VALUE_IN_DEGREES = config["data_config"]["end_value_of_phi_in_degrees"]
NUMBER_OF_PHI_POINTS = config["data_config"]["number_of_phi_points"] + 1

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)

phi_array_in_radians = [np.radians(degree_value) for degree_value in phi_array_in_degrees]

print(
    f"[INFO]: New list of {len(phi_array_in_radians)} of azimuthal angles "
    f"from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees")

FIXED_K = 5.750
FIXED_XB = 0.360
FIXED_T = -0.17
FIXED_Q_SQUARED = 2.300
TEST_LEPTON_HELICITY = 0.0
TEST_TARGET_POLARIZATION = 0.0

print(f"[INFO]: Received k = {FIXED_K} GeV")
print(f"[INFO]: Received xB = {FIXED_XB}")
print(f"[INFO]: Received t = {FIXED_T} GeV^2")
print(f"[INFO]: Received Q^2 = {FIXED_Q_SQUARED} GeV^2")

try:
    # [NOTE]: We actually don't need to be super accurate here because we ONLY use
    # this class to evaluate the CFFs later!
    test_datapoints = [g.DataPoint(
        xB = FIXED_XB, t = FIXED_T, Q2 = FIXED_Q_SQUARED, phi = fixed_phi,
        process = "ep2epgamma", exptype = 'fixed target',
        in1energy = FIXED_K, in1charge = -1, in1polarization = +1, in1units = 'rad',
        observable = 'XS',
        fname = 'Trento') for fixed_phi in phi_array_in_radians]
except ZeroDivisionError:
    print(f"[ERROR]: Kinematic setting k = {FIXED_K}, xb = {FIXED_XB}, t = {FIXED_T}, Q^2 = {FIXED_Q_SQUARED} unphysical according to gepard.")

# here we compute the actual CFFs!
real_h_values = np.array([th_KM15.ReH(datapoint) for datapoint in test_datapoints])
imag_h_values = np.array([th_KM15.ImH(datapoint) for datapoint in test_datapoints])
real_e_values = np.array([th_KM15.ReE(datapoint) for datapoint in test_datapoints])
imag_e_values = np.array([th_KM15.ImE(datapoint) for datapoint in test_datapoints])
real_ht_values = np.array([th_KM15.ReHt(datapoint) for datapoint in test_datapoints])
imag_ht_values = np.array([th_KM15.ImHt(datapoint) for datapoint in test_datapoints])
real_et_values = np.array([th_KM15.ReEt(datapoint) for datapoint in test_datapoints])
imag_et_values = np.array([th_KM15.ImEt(datapoint) for datapoint in test_datapoints])

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

km15_cff_string = (
    rf"$\mathcal{{H}} = {CFF_H_KM15:.3f}$, "
    rf"$\mathcal{{E}} = {CFF_E_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{H}}}} = {CFF_H_TILDE_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{E}}}} = {CFF_E_TILDE_KM15:.3f}$ "
)

this_kinematic_set_title_string = (
    rf"$k = {FIXED_K:.3f}$ GeV, "
    rf"$x_B = {FIXED_XB:.3f}$, "
    rf"$t = {FIXED_T:.3f}$ GeV$^2$, "
    rf"$Q^2 = {FIXED_Q_SQUARED:.3f}$ GeV$^2$"
)

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
 
bkm10_unp_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = 0.0,
    target_polarization = 0.0).real

bkm10_plus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = +1.0,
    target_polarization = 0.0).real

bkm10_minus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = -1.0,
    target_polarization = 0.0).real

bkm10_unp_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = 0.0,
    target_polarization = +0.5).real

bkm10_plus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = +1.0,
    target_polarization = +0.5).real

bkm10_minus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = -1.0,
    target_polarization = +0.5).real

bkm10_bsa_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = 0.0).real

bkm10_bsa_plus_lp_target_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = +0.5).real

bkm10_bsa_minus_lp_target_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = -0.5).real

def plot_bkm10_observable(
    x_data, y_data,
    y_label, title,
    observable_label,
    figure_filename):
    
    figure, axis = plt.subplots(1, figsize = (10, 10))

    axis.scatter(
        x_data,
        y_data,
        s = 4.0,
        color = "blue",
        label = observable_label
    )

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
            plots_directory / f"{figure_filename}.{extension}",
            facecolor = "white"
        )

    plt.close(figure)

observables_to_plot = {
    "xsec_UU": {
        "data": bkm10_unp_beam_unp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{UU}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_unp_beam_unp_target_prediction",
    },
    "xsec_+U": {
        "data": bkm10_plus_beam_unp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{+U}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_plus_beam_unp_target_prediction",
    },
    "xsec_-U": {
        "data": bkm10_minus_beam_unp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{-U}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_minus_beam_unp_target_prediction",
    },
     "xsec_UL": {
        "data": bkm10_unp_beam_lp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{L}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_unp_beam_lp_target_prediction",
    },
    "xsec_+L": {
        "data": bkm10_plus_beam_lp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{L+}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_plus_beam_lp_target_prediction",
    },
    "xsec_-L": {
        "data": bkm10_minus_beam_lp_target_km15,
        "label": r"BKM10 $d^{4}\sigma^{L-}(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$d^{4}\sigma$ [nb / GeV$^{4}$]",
        "filename": "bkm10_xsec_minus_beam_lp_target_prediction",
    },
    "bsa_LU": {
        "data": bkm10_bsa_km15,
        "label": r"BKM10 $\textrm{BSA}(\Lambda = 0)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{BSA}(\Lambda = 0)$ [unitless]",
        "filename": "bkm10_bsa_lp_target_prediction",
    },
    "bsa_U+": {
        "data": bkm10_bsa_plus_lp_target_km15,
        "label": r"BKM10 $\textrm{BSA}(\Lambda = +1/2)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{BSA}(\Lambda = +1/2)$ [unitless]",
        "filename": "bkm10_bsa_plus_lp_target_prediction",
    },
    "bsa_U-": {
        "data": bkm10_bsa_minus_lp_target_km15,
        "label": r"BKM10 $\textrm{BSA}(\Lambda = -1/2)(\mathcal{F}_{\textrm{KM15}})$",
        "ylabel": r"$\textrm{BSA}(\Lambda = -1/2)$ [unitless]",
        "filename": "bkm10_bsa_plus_lp_target_prediction",
    },
}

for observable_name, observable in observables_to_plot.items():

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

k = np.full(NUMBER_OF_PHI_POINTS, FIXED_K, dtype = np.float32)
t = np.full(NUMBER_OF_PHI_POINTS, FIXED_T, dtype = np.float32)
xb = np.full(NUMBER_OF_PHI_POINTS, FIXED_XB, dtype = np.float32)
q2 = np.full(NUMBER_OF_PHI_POINTS, FIXED_Q_SQUARED, dtype = np.float32)

phi = np.array(phi_array_in_radians, dtype = np.float32)

fe = compute_fe(t)
fg = compute_fg(fe)
f2 = compute_f2(t, fe, fg)
f1 = compute_f1(fg, f2)

epsilon = compute_epsilon(xb,q2)
y = compute_y(k, q2, epsilon)
xi = compute_skewness(xb, t, q2)
t_min = compute_t_min(xb, q2, epsilon)
t_prime = compute_t_prime(t, t_min)
k_tilde = compute_k_tilde(xb, q2, t, t_min, epsilon)
kinematic_k = compute_k(q2, y, epsilon, k_tilde)

k_dot_delta = compute_k_dot_delta(
    q2, xb, t, phi,
    epsilon, y, kinematic_k)
prop_1_values = prop_1(q2, k_dot_delta)
prop_2_values = prop_2(q2, t, k_dot_delta)

helicity = np.full(NUMBER_OF_PHI_POINTS, 0.0, dtype = np.float32)
polarization = np.full(NUMBER_OF_PHI_POINTS, 0.0, dtype = np.float32)

kinematics_and_phi = np.column_stack((t, xb, q2, phi)).astype(np.float32)

physics_data = np.column_stack((
    # kinematics, phi:
    t, xb, q2, phi,

    # form factors:
    fe, fg, f1, f2,

    # derived kinematics:
    epsilon, y, xi, t_prime, k_tilde, kinematic_k,

    # phi-dependent stuff:
    k_dot_delta,  prop_1_values, prop_2_values,

    # polarizations:
    helicity, polarization
)).astype(np.float32)

observable_data = {
    "unp_beam_unp_target_xsec": bkm10_unp_beam_unp_target_km15,
    "unp_target_bsa": bkm10_bsa_km15,
}

# stack the observables to fit:
observables = np.column_stack((
    [observable_data[name] for name in enabled_observables]
)).astype(np.float32)

####################################################################################################
# Constructing the training data
####################################################################################################

indices = np.arange(NUMBER_OF_PHI_POINTS)

dnn_inputs = np.column_stack((t, xb, q2)).astype(np.float32)

training_indices, testing_indices = train_test_split(
    indices,
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

y_training = observables[training_indices]
y_validation = observables[validation_indices]
y_testing = observables[testing_indices]

# npz nowadays:
np.savez(
    'testing_npz_rightnow',
    x_training = x_training,
    x_validation= x_validation,
    x_testing = x_testing,

    physics_training = precomputed_physics_training,
    physics_validation = precomputed_physics_validation,
    physics_testing = precomputed_physics_testing,

    y_training = y_training,
    y_validation = y_validation,
    y_testing = y_testing,

    training_indices = training_indices,
    validation_indices = validation_indices,
    testing_indices = testing_indices,
)

def cff_h_model():

    kinematics_inputs = tf.keras.Input(
        shape = (3,),
        name = "input_values")
    
    physics_input = tf.keras.Input(
        shape = (19,),
        name = "precomputed_physics")
    
    dnn_kinematic_inputs = tf.keras.layers.Lambda(
        lambda x: x[:, :3],
        name = "input_kinematics")(kinematics_inputs)
    
    hidden = tf.keras.layers.Dense(
        10,
        kernel_initializer = "he_normal",
        activation = "relu")(dnn_kinematic_inputs)
    
    hidden = tf.keras.layers.Dense(
        10,
        kernel_initializer = "he_normal",
        activation = "relu")(hidden)
    
    hidden = tf.keras.layers.Dense(
        10,
        kernel_initializer = "he_normal",
        activation = "relu")(hidden)
    
    hidden = tf.keras.layers.Dense(
        10,
        kernel_initializer = "he_normal",
        activation = "relu")(hidden)
    
    # Re[H], Im[H], Re[Ht], Im[Ht], Re[E], Im[E], Re[Et], Im[Et]
    cff_outputs = tf.keras.layers.Dense(
        8,
        activation = "linear",
        name = "cff_outputs")(hidden)
    
    full_model_outputs = tf.keras.layers.Concatenate(
        name = "physics_and_cffs")([
            cff_outputs,
            physics_input])
    
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
    
    # return model:
    return model

for replica in range(NUMBER_OF_REPLICAS):
    
    replica_number = replica + 1

    start_dnn_compile_time = time.perf_counter()
    dnn_model = cff_h_model()
    print(f"[TIMING]: model construction time: {time.perf_counter() - start_dnn_compile_time:.2f} s")

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
    print(f"[TIMING]: fitting time: {time.perf_counter() - start_dnn_fitting_time:.2f} s")

    number_of_epochs_run = len(dnn_model_history.epoch)
    print(f"[INFO]: The model ran for {number_of_epochs_run} epochs before early stopping.")

    start_dnn_save_time = time.perf_counter()
    dnn_model.save(
        replica_directory /
        f"replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras"
        )
    print(f"[TIMING]: DNN saving time: {time.perf_counter() - start_dnn_save_time:.2f} s")

    training_loss_data = dnn_model_history.history["loss"]
    validation_loss_data = dnn_model_history.history["val_loss"]

    start_dnn_evaluation_time = time.perf_counter()
    dnn_evaluation_statistics = dnn_model.evaluate(
        x = {
            "input_values": x_testing,
            "precomputed_physics": precomputed_physics_testing
            },
        y = y_testing,
        verbose = 0)
    print(f"[TIMING]: DNN evaluation time: {time.perf_counter() - start_dnn_evaluation_time:.2f} s")
    print(f"[INFO]: Test Loss for Replica {replica_number}: {dnn_evaluation_statistics}")

    # make DF with testing metrics:
    pd.DataFrame({
        'testing_loss': [dnn_evaluation_statistics], # https://stackoverflow.com/a/17840195 -> for why we need to cast it into a list!
    }).to_csv(
        replica_directory /
        f"replica_{replica_number}_loss_data.csv",
        index = False)

    # save npz with DNN training information:
    np.savez(
        file = replica_directory / f"replica_{replica_number}_losses_vs_epochs.npz",
        training_loss = training_loss_data,
        validation_loss = validation_loss_data
        )

    # make DF with DNN training information
    pd.DataFrame(dnn_model_history.history).to_csv(
        replica_directory / f"replica_{replica_number}_losses_vs_epochs.csv",
        index = False)

    predicted_outputs = dnn_model.predict(
        {
            "input_values": dnn_inputs,
            "precomputed_physics": physics_data,
        },
        verbose = 0
    )

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
    k_tilde = compute_k_tilde(xb, q2, t, t_min, epsilon)
    kinematic_k = compute_k(q2, y, epsilon, k_tilde)
    kdd = compute_k_dot_delta(q2, xb, t, phi, epsilon, y, kinematic_k)
    p1 = prop_1(q2, kdd)
    p2 = prop_2(q2, t, kdd)
    cross_section_prediction = bkm10_cross_section(
        TEST_LEPTON_HELICITY, TEST_TARGET_POLARIZATION,
        q2, xb, t, epsilon, y_lep, xi, k, f1, f2, k_tilde, tprime, phi, p1, p2,
        cff_h_real, cff_ht_real, cff_e_real, cff_et_real, cff_h_imag, cff_ht_imag, cff_e_imag, cff_et_imag)

    bsa_prediction = bkm10_bsa(
        TEST_TARGET_POLARIZATION,
        q2, xb, t, epsilon, y_lep, xi, k, f1, f2, k_tilde, tprime, phi, p1, p2,
        cff_h_real, cff_ht_real, cff_e_real, cff_et_real, cff_h_imag, cff_ht_imag, cff_e_imag, cff_et_imag)

    np.savez(
        file = replica_directory / f"replica_{replica_number}_predictions.npz",
        t = t,
        xb = xb,
        q_squared = q2,
        phi = phi,

        cff_h_real = cff_predictions[:, 0],
        cff_h_imag = cff_predictions[:, 1],
        cff_ht_real = cff_predictions[:, 2],
        cff_ht_imag = cff_predictions[:, 3],
        cff_e_real = cff_predictions[:, 4],
        cff_e_imag = cff_predictions[:, 5],
        cff_et_real = cff_predictions[:, 6],
        cff_et_imag = cff_predictions[:, 7],

        cross_section = cross_section_prediction,
        bsa = bsa_prediction,
    )


    # [INFO]: this is called an "aggressive" memory cleanup:
    del dnn_model
    del dnn_model_history
    del training_loss_data
    del validation_loss_data
    del dnn_evaluation_statistics

    gc.collect()
    tf.keras.backend.clear_session()

loss_files = sorted(glob.glob(f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_*_losses_vs_epochs.npz"))

for replica_index, loss_file in enumerate(loss_files, start = 1):

    loss_information = np.load(loss_file)

    training_loss_data = loss_information["training_loss"]
    validation_loss_data = loss_information["validation_loss"]

    number_of_epochs_run = len(training_loss_data)
    epochs = np.arange(number_of_epochs_run)

    testing_information = pd.read_csv(
        f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_index}_loss_data.csv"
    )

    testing_loss = testing_information["testing_loss"].iloc[0]

    curves_fig, curves_ax = plt.subplots(1, figsize = (8, 8))
    log_curves_fig, log_curves_ax = plt.subplots(1, figsize = (8, 8))

    initial_loss_value = training_loss_data[0]

    curves_ax.axhline(initial_loss_value, color = "red", linestyle = "--", label = "Initial Loss Value")
    curves_ax.axhline(0.0, color = "green", linestyle="--", label = r"Loss$ = 0$")

    curves_ax.plot(np.arange(0, number_of_epochs_run, 1), training_loss_data, color = "blue", label = "Training Loss")
    curves_ax.plot(np.arange(0, number_of_epochs_run, 1), validation_loss_data, color = "purple", label = "Validation Loss")

    log_curves_ax.axhline(initial_loss_value, color = "red", linestyle = "--", label = "Initial Loss Value")
    log_curves_ax.axhline(0.0, color = "green", linestyle = "--", label = r"Loss$ = 0$")

    # just do this for now hahahaha:
    _REGULARIZER = 1e-21

    log_curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.log(training_loss_data + _REGULARIZER), color = "blue", label = "Log Training Loss")
    log_curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.log(validation_loss_data + _REGULARIZER), color = "purple", label = "Log Validation Loss")

    curves_ax.legend(fontsize = 15)
    log_curves_ax.legend(fontsize = 15)

    curves_ax.set_xlabel("Epoch", fontsize = 15)
    curves_ax.set_ylabel("MSE", fontsize = 15)
    curves_ax.set_title(f"Replica {replica_index} Learning Curves\n(Eval. Loss $= {testing_loss:.3g}$", fontsize = 16.)

    log_curves_ax.set_xlabel("Epoch", fontsize = 15)
    log_curves_ax.set_ylabel("Log MSE Loss", fontsize = 15)
    log_curves_ax.set_title(f"Replica {replica_index} Learning Curves\n(Eval. Loss $= {testing_loss:.3g}$", fontsize = 15.)

    for extension in ['png', 'eps']:
        curves_fig.savefig(
            learning_curves_directory /
            f"lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.{extension}",
            facecolor = 'white')

        log_curves_fig.savefig(
            learning_curves_directory /
            f"log_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.{extension}",
            facecolor = 'white')
    plt.close(curves_fig)
    plt.close(log_curves_fig)

    del curves_fig
    del log_curves_fig

range_of_t = np.linspace(x_training.min(), x_training.max())
range_of_x_b = np.linspace(x_training.min(), x_training.max())
range_of_q_squared = np.linspace(x_training.min(), x_training.max())

replica_paths = sorted(
    replica_directory.glob(f"replica_*_v{MAJOR_MINOR_NUMBER}.keras")
)

replicas = [tf.keras.models.load_model(
    path,
    compile = False,
    safe_mode = False) for path in replica_paths]

print(f"[INFO]: Loaded {len(replicas)} replica models.")

assert len(replicas) == NUMBER_OF_REPLICAS, "[ASSERT]: Number of loaded replicas does not equal expected number"

replicas_cross_predictions = []
replicas_bsa_predictions = []

replicas_h_real = []
replicas_h_imag = []
replicas_ht_real = []
replicas_ht_imag = []
replicas_e_real = []
replicas_e_imag = []
replicas_et_real = []
replicas_et_imag = []

for replica_index in range(1, NUMBER_OF_REPLICAS + 1):

    prediction_path = (
        replica_directory
        / f"replica_{replica_index}_predictions.npz"
    )

    prediction_information = np.load(prediction_path)
    replicas_h_real.append(prediction_information["cff_h_real"])
    replicas_h_imag.append(prediction_information["cff_h_imag"])
    replicas_ht_real.append(prediction_information["cff_ht_real"])
    replicas_ht_imag.append(prediction_information["cff_ht_imag"])
    replicas_e_real.append(prediction_information["cff_e_real"])
    replicas_e_imag.append(prediction_information["cff_e_imag"])
    replicas_et_real.append(prediction_information["cff_et_real"])
    replicas_et_imag.append(prediction_information["cff_et_imag"])
    replicas_cross_predictions.append(prediction_information["cross_section"])
    replicas_bsa_predictions.append(prediction_information["bsa"])

    prediction_information.close()

def crunch_statistics(data):

    # huge dictionary of statistics
    statistics_dictionary = {
        'mean': np.mean(data, axis = 0),
        'std': np.std(data, axis = 0),
        'median': np.median(data, axis = 0),
        'min': np.min(data, axis = 0),
        'max': np.max(data, axis = 0)
    }

    for percentile in range(10, 50, 10):
        statistics_dictionary[f'p{percentile}'] = np.percentile(data, percentile, axis = 0)
        statistics_dictionary[f'p{100 - percentile}'] = np.percentile(data, 100 - percentile, axis = 0)

    return statistics_dictionary

# observable statistics:
xs_stats = crunch_statistics(replicas_cross_predictions)
bsa_stats = crunch_statistics(replicas_bsa_predictions)

replica_statistics_dataframe = pd.DataFrame({
    'k': FIXED_K,
    't': FIXED_T,
    'xb': FIXED_XB,
    'q_squared': FIXED_Q_SQUARED,
    'phi': phi_array_in_degrees,

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
    data_directory / f"observable_preds_v{MAJOR_MINOR_NUMBER}.csv", 
    index = False)

cff_h_real_pred_per_replica = np.mean(
    replicas_h_real,
    axis = 1
)

cff_h_imag_pred_per_replica = np.mean(
    replicas_h_imag,
    axis = 1
)

cff_ht_real_pred_per_replica = np.mean(
    replicas_ht_real,
    axis = 1
)

cff_ht_imag_pred_per_replica = np.mean(
    replicas_ht_imag,
    axis = 1
)

cff_e_real_pred_per_replica = np.mean(
    replicas_e_real,
    axis = 1
)

cff_e_imag_pred_per_replica = np.mean(
    replicas_e_imag,
    axis = 1
)

cff_et_real_pred_per_replica = np.mean(
    replicas_et_real,
    axis = 1
)

cff_et_imag_pred_per_replica = np.mean(
    replicas_et_imag,
    axis = 1
)

replica_cffs_dataframe = pd.DataFrame({
    "ReH_pred": cff_h_real_pred_per_replica,
    "ImH_pred": cff_h_imag_pred_per_replica,

    "ReHt_pred": cff_ht_real_pred_per_replica,
    "ImHt_pred": cff_ht_imag_pred_per_replica,

    "ReE_pred": cff_e_real_pred_per_replica,
    "ImE_pred": cff_e_imag_pred_per_replica,

    "ReEt_pred": cff_et_real_pred_per_replica,
    "ImEt_pred": cff_et_imag_pred_per_replica,
})

replica_cffs_dataframe.to_csv(
    data_directory /
    f"cff_replica_average_preds_v{MAJOR_MINOR_NUMBER}.csv", 
    index = False)

fig2, ax2 = plt.subplots(1, figsize = (10, 7))

ax2.scatter(
    phi_array_in_radians, bkm10_unp_beam_unp_target_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

ax2.plot(
    phi_array_in_radians,
    replica_statistics_dataframe['mean_xs'],
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['max_xs'],
    y2 = replica_statistics_dataframe['min_xs'],
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['p90_xs'],
    y2 = replica_statistics_dataframe['p10_xs'],
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['p80_xs'],
    y2 = replica_statistics_dataframe['p20_xs'],
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['p70_xs'],
    y2 = replica_statistics_dataframe['p30_xs'],
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['p60_xs'],
    y2 = replica_statistics_dataframe['p40_xs'],
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

ax2.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
ax2.set_ylabel(r"$d^{4}\sigma$ [nb / GeV$^{4}$]", fontsize = 16)
ax2.set_title(
    rf"$d^{{4}}\sigma^{{UU}}$ vs. $\phi$, {this_kinematic_set_title_string}"
    "\n"
    f"(KM15): {km15_cff_string}", fontsize = 16
)

ax2.legend()
plt.tight_layout()

replica_cross_section_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/dnn_xsec_vs_phi_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig2.savefig(
        f"{replica_cross_section_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig2)

fig3, ax3 = plt.subplots(1, figsize = (10, 7))

ax3.scatter(
    phi_array_in_radians, bkm10_bsa_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

ax3.plot(
    phi_array_in_radians,
    replica_statistics_dataframe['mean_bsa'],
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['max_bsa'],
    y2 = replica_statistics_dataframe['min_bsa'],
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['p90_bsa'],
    y2 = replica_statistics_dataframe['p10_bsa'],
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['p80_bsa'],
    y2 = replica_statistics_dataframe['p20_bsa'],
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['p70_bsa'],
    y2 = replica_statistics_dataframe['p30_bsa'],
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = replica_statistics_dataframe['p60_bsa'],
    y2 = replica_statistics_dataframe['p40_bsa'],
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

ax3.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
ax3.set_ylabel(r"BSA [unitless]", fontsize = 16)
ax3.set_title(
    rf"BSA vs. $\phi$, {this_kinematic_set_title_string}"
    "\n"
    f"(KM15): {km15_cff_string}", fontsize = 16)

ax3.legend()
fig3.tight_layout()

replica_bsa_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/dnn_bsa_vs_phi_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig3.savefig(
        f"{replica_bsa_plotname}.{extension}",
        facecolor = 'white')

plt.close(fig3)

def make_cff_plot_label(
        cff_label: str
    ):
    """
    I am going to customize LaTeX representation of a given CFF
    based on the string version that I received from a datafile.
    """

    # why does this work? because the variable is a STRING
    real_or_imag = cff_label[:2]
    cff_name = cff_label[2:]

    component = { "Re": "Re", "Im": "Im" }[real_or_imag]

    cff_symbol = {
        "H": r"\mathcal{H}",
        "Ht": r"\widetilde{\mathcal{H}}",
        "E": r"\mathcal{E}",
        "Et": r"\widetilde{\mathcal{E}}",
    }[cff_name]

    return rf"{component}$[{cff_symbol}]$"

_NUMBER_OF_STDDEVS = 4.
_NUMBER_OF_HISTOGRAM_BINS = 30
_NUMBER_OF_GAUSSIAN_POINTS = 200

km15_values = {
    "ReH": CFF_REAL_H_KM15,
    "ImH": CFF_IMAG_H_KM15,
    "ReE": CFF_REAL_E_KM15,
    "ImE": CFF_IMAG_E_KM15,
    "ReHt": CFF_REAL_HT_KM15,
    "ImHt": CFF_IMAG_HT_KM15,
    "ReEt": CFF_REAL_ET_KM15,
    "ImEt": CFF_IMAG_ET_KM15,
}

cff_h_km15 = complex(CFF_REAL_H_KM15, CFF_IMAG_H_KM15)
cff_e_km15 = complex(CFF_REAL_E_KM15, CFF_IMAG_E_KM15)
cff_ht_km15 = complex(CFF_REAL_HT_KM15, CFF_IMAG_HT_KM15)
cff_et_km15 = complex(CFF_REAL_ET_KM15, CFF_IMAG_ET_KM15)

km15_cff_string = (
    rf"$\mathcal{{H}} = {cff_h_km15:.3f}$, "
    rf"$\mathcal{{E}} = {cff_e_km15:.3f}$, "
    rf"$\widetilde{{\mathcal{{H}}}} = {cff_ht_km15:.3f}$, "
    rf"$\widetilde{{\mathcal{{E}}}} = {cff_et_km15:.3f}$"
)

for cff_label in km15_values:
        
    corresponding_key = f"{cff_label}_pred"

    if corresponding_key not in replica_cffs_dataframe.columns:
        continue

    cff_prediction_per_replica = replica_cffs_dataframe[corresponding_key]
    
    cff_mean, cff_stddev = norm.fit(cff_prediction_per_replica)
    
    cff_km15_value = km15_values[cff_label]
    
    gaussian_x_values = np.linspace(
        cff_mean - _NUMBER_OF_STDDEVS * cff_stddev,
        cff_mean + _NUMBER_OF_STDDEVS * cff_stddev,
        _NUMBER_OF_GAUSSIAN_POINTS
    )
    
    kinematic_title = (
        rf"$k = {FIXED_K:.3f}$ GeV, "
        rf"$x_B = {FIXED_XB:.3f}$, "
        rf"$t = {FIXED_T:.3f}$ GeV$^2$, "
        rf"$Q^2 = {FIXED_Q_SQUARED:.3f}$ GeV$^2$"
        )
    
    cff_figure, cff_axis = plt.subplots(1, 1, figsize = (10, 8))
    
    cff_axis.hist(
        cff_prediction_per_replica,
        bins = _NUMBER_OF_HISTOGRAM_BINS,
        alpha = 0.6,
        color = "skyblue",
        edgecolor = "black")
    
    cff_axis.plot(
        gaussian_x_values,
        norm.pdf(
            gaussian_x_values,
            cff_mean,
            cff_stddev
        ),
        color = "red",
        linestyle = "--",
        label = (
            fr"Gaussian Fit: $\mu = {cff_mean:.3f}$, $\sigma = {cff_stddev:.3f}$"
        )
    )
    
    cff_axis.axvline(
        cff_km15_value,
        color = "green",
        linestyle = "-",
        linewidth = 2.0,
        label = f"KM15: {cff_km15_value:.3f}")
    
    cff_axis.set_ylabel(
        "Frequency",
        rotation = 90.,
        fontsize = 16.0)
    
    cff_axis.set_xlabel(
        make_cff_plot_label(cff_label),
        fontsize = 16.0)
    
    cff_axis.set_title(
        rf"{make_cff_plot_label(cff_label)} Distribution, "
        rf"{kinematic_title}"
        "\n"
        rf"(KM15): {km15_cff_string}",
        fontsize = 16.0)
    
    cff_axis.legend(fontsize = 16.0)
    
    cff_axis.text(
        0.00, -0.05,
        f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        transform = cff_axis.transAxes)
    
    cff_figure.tight_layout()
    
    for extension in ["png", "eps"]:
        cff_figure.savefig(
            plots_directory /
            f"cff_{cff_label}_fits_v{MAJOR_MINOR_NUMBER}.{extension}",
            facecolor = "white",
            transparent = False
        )
    
    plt.close(cff_figure)
