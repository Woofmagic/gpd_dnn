"""
A single script that runs a simultaneous local fit with
N < 10 replicas.
Created: 20260823
Last changed: 20260901
Notes:
    1.  2026/09/01:
        Successfully runs.
"""

from pathlib import Path
import glob
import time
import gc
import yaml

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
from simultaneous_fit_dnn_config import bkm10_cross_section
from simultaneous_fit_dnn_config import bkm10_bsa

print("[INFO]: Libraries imported!")

with open(
    "closure_test_config.yml",
    "r",
    encoding = "utf-8") as file:
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

observable_weights = [
    0.5,
    0.5,
]

FIXED_K = 5.750
FIXED_XB = 0.360
FIXED_T = -0.17
FIXED_Q_SQUARED = 2.300

print(f"[INFO]: Received k = {FIXED_K} GeV")
print(f"[INFO]: Received xB = {FIXED_XB}")
print(f"[INFO]: Received t = {FIXED_T} GeV^2")
print(f"[INFO]: Received Q^2 = {FIXED_Q_SQUARED} GeV^2")

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

phi_array_in_radians = [np.radians(degree_value) for degree_value in phi_array_in_degrees]

print(
    f"[INFO]: New list of {len(phi_array_in_radians)} of azimuthal angles "
    f"from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees")

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
    target_polarization = 0+0.5).real

bkm10_bsa_unp_target_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = 0.0).real

bkm10_bsa_unp_plus_lp_target_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = +0.5).real

bkm10_bsa_unp_minus_lp_target_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = -0.5).real

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
    "unp_target_bsa": bkm10_bsa_unp_target_km15,
}

# stack the observables to fit:
observables = np.column_stack((
    [observable_data[name] for name in enabled_observables]
)).astype(np.float32)

indices = np.arange(NUMBER_OF_PHI_POINTS)

dnn_inputs = np.column_stack((t, xb, q2)).astype(np.float32)

training_indices, testing_indices = train_test_split(
    indices, test_size = 0.10, random_state = 7009,)

training_indices, validation_indices = train_test_split(
    training_indices, test_size = 0.10, random_state = 7009,)

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

class UnfoldedSimultaneousFitLoss(tf.keras.losses.Loss):
    def __init__(
            self,
            enabled_observables,
            observable_weights = None
        ): 
        super().__init__(name = "simultaneous_loss")

        # debugging parameter:
        self.debugging = False

        # the set of observables we want to fit:
        self.enabled_observables = enabled_observables

        # the set of *weights* per observable:
        self.observable_weights = tf.constant(
            observable_weights,
            dtype = tf.float32,
        )

        # WW relations:
        self.use_ww = True

        # numerical constants:
        self.gev6_to_gev4_per_nb = tf.constant(.389379 * 1000000.)
        self.mp = tf.constant(0.93827208816)
        self.qed_alpha = tf.constant(1./137.035999177)
        self.fe_constant = tf.constant(0.710649)
        self.mu_proton = tf.constant(2.79284734463)

    def debug_print(self, label, value):
        # need this for huge comparisons...
        if self.debugging:
            tf.print(label, value)

    def compute_cross_section(
        self,
        q_sq_tf, xb_tf, t_tf, ep_tf, y_lep_tf, xi_tf, k_tf, f1_tf, f2_tf, ktilde_tf, tprime_tf, phi_tf, p1_tf, p2_tf,
        cff_h_real_tf, cff_ht_real_tf, cff_e_real_tf, cff_et_real_tf, cff_h_imag_tf, cff_ht_imag_tf, cff_e_imag_tf, cff_et_imag_tf,
        cff_h_real_eff_tf, cff_ht_real_eff_tf, cff_e_real_eff_tf, cff_et_real_eff_tf, cff_h_imag_eff_tf, cff_ht_imag_eff_tf, cff_e_imag_eff_tf, cff_et_imag_eff_tf,
        lep_lambda, tgt_lambda
        ):

        # BH: c0
        first_line = 8. * k_tf**2 * (((2. + 3. * ep_tf**2) * (f1_tf**2 - (t_tf * f2_tf**2 / (4. * self.mp**2))) / (t_tf / q_sq_tf)) + (2. * xb_tf**2 * (f1_tf + f2_tf)**2))
        second_line_first_part = (2. + ep_tf**2) * ((4. * xb_tf**2 * self.mp**2 / t_tf) * (1. + (t_tf / q_sq_tf))**2 + 4. * (1 - xb_tf) * (1. + (xb_tf * (t_tf / q_sq_tf)))) * (f1_tf**2 - (t_tf * f2_tf**2 / (4. * self.mp**2)))
        second_line_second_part = 4. * xb_tf**2 * (xb_tf + (1. - xb_tf + (ep_tf**2 / 2.)) * (1 - (t_tf / q_sq_tf))**2 - xb_tf * (1. - 2. * xb_tf) * (t_tf / q_sq_tf)**2) * (f1_tf + f2_tf)**2
        second_line = (2. - y_lep_tf)**2 * (second_line_first_part + second_line_second_part)
        third_line = 8. * (1. + ep_tf**2) * (1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)) * (2. * ep_tf**2 * (1 - (t_tf / (4. * self.mp**2))) * (f1_tf**2 - (t_tf * f2_tf**2 / (4. * self.mp**2))) - xb_tf**2 * (1 - (t_tf / q_sq_tf))**2 * (f1_tf + f2_tf)**2)
        bh_c0 = first_line + second_line + third_line

        # correct, 2026/08/25
        self.debug_print("[DEBUG]: BH c0 unp: ", bh_c0)

        # BH: c1
        addition_of_form_factors_squared = (f1_tf + f2_tf)**2
        weighted_combination_of_form_factors = f1_tf**2 - ((t_tf / (4. * self.mp**2)) * f2_tf**2)
        first_line_first_part = ((4. * xb_tf**2 * self.mp**2 / t_tf) - 2. * xb_tf - ep_tf**2) * weighted_combination_of_form_factors
        first_line_second_part = 2. * xb_tf**2 * (1. - (1. - 2. * xb_tf) * (t_tf / q_sq_tf)) * addition_of_form_factors_squared
        bh_c1 = 8. * k_tf * (2. - y_lep_tf) * (first_line_first_part + first_line_second_part)

        # correct, 2026/08/25
        self.debug_print("[DEBUG]: BH c1 unp", bh_c1)

        # BH: c2
        addition_of_form_factors_squared = (f1_tf + f2_tf)**2
        weighted_combination_of_form_factors = f1_tf**2 - ((t_tf/ (4. * self.mp**2)) * f2_tf**2)
        first_part_of_contribution = (4. * self.mp**2 / t_tf) * weighted_combination_of_form_factors
        bh_c2 = 8. * xb_tf**2 * k_tf**2 * (first_part_of_contribution + 2. * addition_of_form_factors_squared)

        # correct, 2026/08/25
        self.debug_print("[DEBUG]: BH c2 unp: ", bh_c2)

        # BH LP: c0
        sum_of_form_factors = (f1_tf + f2_tf)
        t_over_four_mp_squared = t_tf / (4. * self.mp**2)
        weighted_sum_of_form_factors = f1_tf + t_over_four_mp_squared * f2_tf
        one_minus_xb = 1. - xb_tf
        t_over_Q_squared = t_tf / q_sq_tf
        one_minus_t_over_Q_squared = 1. - t_over_Q_squared
        first_term_first_bracket = 0.5 * xb_tf * (one_minus_t_over_Q_squared) - t_over_four_mp_squared
        first_term_second_bracket = 2. - xb_tf - (2. * (one_minus_xb)**2 * t_over_Q_squared) + (ep_tf**2 * one_minus_t_over_Q_squared) - (xb_tf * (1. - 2. * xb_tf) * t_over_Q_squared**2)
        first_term = 0.5 * sum_of_form_factors * first_term_first_bracket * first_term_second_bracket
        second_term_first_bracket = xb_tf**2 * (1. + t_over_Q_squared)**2 / (4. * t_over_four_mp_squared) + ((1. - xb_tf) * (1. + xb_tf * t_over_Q_squared))
        second_term = (1. - (1. - xb_tf) * t_over_Q_squared) * weighted_sum_of_form_factors * second_term_first_bracket
        prefactor = 8. * lep_lambda * tgt_lambda * xb_tf * (2. - y_lep_tf) * y_lep_tf * tf.sqrt(1. + ep_tf**2) * sum_of_form_factors / (1. - t_over_four_mp_squared)
        bh_lp_c0 = prefactor * (first_term + second_term)

        # correct, 2026/08/25
        self.debug_print("[DEBUG]: BH c0 LP: ", bh_lp_c0)

        # BH LP: c1
        sum_of_form_factors = (f1_tf + f2_tf)
        t_over_four_mp_squared = t_tf / (4. * self.mp**2)
        weighted_sum_of_form_factors = f1_tf + t_over_four_mp_squared * f2_tf
        t_over_Q_squared = t_tf / q_sq_tf
        first_term = ((2. * t_over_four_mp_squared) - (xb_tf * (1. - t_over_Q_squared))) * ((1. - xb_tf + (xb_tf * t_over_Q_squared))) * sum_of_form_factors
        second_term_bracket_term = 1. + xb_tf - ((3. - 2. * xb_tf) * (1. + xb_tf * t_over_Q_squared)) - (xb_tf**2 * (1. + t_over_Q_squared**2) / t_over_four_mp_squared)
        second_term = weighted_sum_of_form_factors * second_term_bracket_term
        prefactor = -8. * lep_lambda * tgt_lambda * xb_tf * y_lep_tf * k_tf * tf.sqrt(1. + ep_tf**2) * sum_of_form_factors / (1. - t_over_four_mp_squared)
        bh_lp_c1 = prefactor * (first_term + second_term)

        # correct, 2026/08/25
        self.debug_print("[DEBUG]: BH c1 LP: ", bh_lp_c1)

        # BH LP: c2
        bh_lp_c2 = 0.0

        # correct, 2026/08/25
        self.debug_print("[DEBUG]: BH c2 LP: ", bh_lp_c2)

        # DVCS Re[CurlyC](F| F*):
        first_line = (4.*(1.-xb_tf)*(cff_h_real_tf*cff_h_real_tf - cff_h_imag_tf*(-cff_h_imag_tf))) + (4.*(1.-xb_tf + 0.25*((2.*q_sq_tf + t_tf)*ep_tf**2)/(q_sq_tf + xb_tf*t_tf))*(cff_ht_real_tf*cff_ht_real_tf - cff_ht_imag_tf*(-cff_ht_imag_tf)))
        next_line = -xb_tf**2*(q_sq_tf+t_tf)**2*(cff_h_real_tf*cff_e_real_tf - cff_e_imag_tf*(-cff_h_imag_tf) + cff_e_real_tf*cff_h_real_tf - cff_h_imag_tf*(-cff_e_imag_tf))/(q_sq_tf*(q_sq_tf+xb_tf*t_tf)) - (xb_tf**2*q_sq_tf*(cff_ht_real_tf*cff_et_real_tf - cff_et_imag_tf*(-cff_ht_imag_tf) + cff_et_real_tf*cff_ht_real_tf - cff_ht_imag_tf*(-cff_et_imag_tf))/(q_sq_tf+xb_tf*t_tf))
        final_line = -1.*(xb_tf**2*(q_sq_tf+t_tf)**2/(q_sq_tf*(q_sq_tf+xb_tf*t_tf)) + 0.25*((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2*t_tf/(q_sq_tf*self.mp**2*(q_sq_tf+xb_tf*t_tf)))*(cff_e_real_tf*cff_e_real_tf - cff_e_imag_tf*(-cff_e_imag_tf)) -0.25*xb_tf**2*q_sq_tf*t_tf*(cff_et_real_tf*cff_et_real_tf - cff_et_imag_tf*(-cff_et_imag_tf))/((q_sq_tf+xb_tf*t_tf)*self.mp**2)
        dvcs_real_curlyc = ((first_line + next_line + final_line)*q_sq_tf*(q_sq_tf+xb_tf*t_tf)/((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2)

        self.debug_print("[DEBUG]: DVCS Re[CurlyC](F| F*): ", dvcs_real_curlyc)

        # DVCS Re[CurlyC](Feff| Feff*):
        first_line = (4.*(1.-xb_tf)*(cff_h_real_eff_tf*cff_h_real_eff_tf - cff_h_imag_eff_tf*(-cff_h_imag_eff_tf))) + (4.*(1.-xb_tf + 0.25*((2.*q_sq_tf + t_tf)*ep_tf**2)/(q_sq_tf + xb_tf*t_tf))*(cff_ht_real_eff_tf*cff_ht_real_eff_tf - cff_ht_imag_eff_tf*(-cff_ht_imag_eff_tf)))
        next_line = -xb_tf**2*(q_sq_tf+t_tf)**2*(cff_h_real_eff_tf*cff_e_real_eff_tf - cff_e_imag_eff_tf*(-cff_h_imag_eff_tf) + cff_e_real_eff_tf*cff_h_real_eff_tf - cff_h_imag_eff_tf*(-cff_e_imag_eff_tf))/(q_sq_tf*(q_sq_tf+xb_tf*t_tf)) - (xb_tf**2*q_sq_tf*(cff_ht_real_eff_tf*cff_et_real_eff_tf - cff_et_imag_eff_tf*(-cff_ht_imag_eff_tf) + cff_et_real_eff_tf*cff_ht_real_eff_tf - cff_ht_imag_eff_tf*(-cff_et_imag_eff_tf))/(q_sq_tf+xb_tf*t_tf))
        final_line = -1.*(xb_tf**2*(q_sq_tf+t_tf)**2/(q_sq_tf*(q_sq_tf+xb_tf*t_tf)) + 0.25*((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2*t_tf/(q_sq_tf*self.mp**2*(q_sq_tf+xb_tf*t_tf)))*(cff_e_real_eff_tf*cff_e_real_eff_tf - cff_e_imag_eff_tf*(-cff_e_imag_eff_tf)) -0.25*xb_tf**2*q_sq_tf*t_tf*(cff_et_real_eff_tf*cff_ht_real_eff_tf - cff_et_imag_eff_tf*(-cff_et_imag_eff_tf))/((q_sq_tf+xb_tf*t_tf)*self.mp**2)
        dvcs_real_curlyc_feff = ((first_line + next_line + final_line)*q_sq_tf*(q_sq_tf+xb_tf*t_tf)/((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2)

        self.debug_print("[DEBUG]: DVCS Re[CurlyC](Feff| Feff*): ", dvcs_real_curlyc_feff)

        # DVCS Re[CurlyC](Feff | F*):
        first_line = 4.*(1.-xb_tf)*(cff_h_real_eff_tf*cff_h_real_tf - cff_h_imag_eff_tf*(-cff_h_imag_tf))+4.*(1.-xb_tf+ 0.25*((2.*q_sq_tf + t_tf)*ep_tf**2)/(q_sq_tf + xb_tf*t_tf))*(cff_ht_real_eff_tf*cff_ht_real_tf- cff_ht_imag_eff_tf*(-cff_ht_imag_tf))
        next_line = -xb_tf**2*(q_sq_tf+t_tf)**2*(cff_h_real_eff_tf*cff_e_real_tf- cff_e_imag_eff_tf*(-cff_h_imag_tf)+ cff_e_real_eff_tf*cff_h_real_tf - cff_h_imag_eff_tf*(-cff_e_imag_tf))/(q_sq_tf*(q_sq_tf+xb_tf*t_tf))-xb_tf**2*q_sq_tf*(cff_ht_real_eff_tf*cff_et_real_tf - cff_et_imag_eff_tf*(-cff_ht_imag_tf)+ cff_et_real_eff_tf*cff_ht_real_tf- cff_ht_imag_eff_tf*(-cff_et_imag_tf))/(q_sq_tf+xb_tf*t_tf)
        final_line = -1.*(xb_tf**2*(q_sq_tf+t_tf)**2/(q_sq_tf*(q_sq_tf+xb_tf*t_tf))+0.25*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2*t_tf/(q_sq_tf*self.mp**2*(q_sq_tf+xb_tf*t_tf)))*(cff_e_real_eff_tf*cff_e_real_tf- cff_e_imag_eff_tf*(-cff_e_imag_tf))-0.25*xb_tf**2*q_sq_tf*t_tf*(cff_et_real_eff_tf*cff_et_real_tf- cff_et_imag_eff_tf*(-cff_et_imag_tf))/((q_sq_tf+xb_tf*t_tf)*self.mp**2)
        dvcs_real_curlyc_f_eff = ((first_line + next_line + final_line)* q_sq_tf*(q_sq_tf+xb_tf*t_tf)/ ((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2)

        self.debug_print("[DEBUG]: DVCS Re[CurlyC](Feff | F*): ", dvcs_real_curlyc_f_eff)

        # DVCS Im[CurlyC](Feff| F*):
        first_line = (4.*(1.-xb_tf)*(cff_h_imag_eff_tf*cff_h_real_tf- cff_h_real_eff_tf*cff_h_imag_tf)+4.*(1.-xb_tf+ 0.25*(2.*q_sq_tf + t_tf)*ep_tf**2/(q_sq_tf + xb_tf*t_tf))*(cff_ht_imag_eff_tf*cff_ht_real_tf - cff_ht_real_eff_tf*cff_ht_imag_tf))
        next_line = (-xb_tf**2*(q_sq_tf+t_tf)**2*(cff_h_imag_eff_tf*cff_e_real_tf- cff_e_real_eff_tf*cff_h_imag_tf+ cff_e_imag_eff_tf*cff_h_real_tf- cff_h_real_eff_tf*cff_e_imag_tf)/(q_sq_tf*(q_sq_tf+xb_tf*t_tf))-xb_tf**2*q_sq_tf*(cff_ht_imag_eff_tf*cff_et_real_tf- cff_et_real_eff_tf*cff_ht_imag_tf+ cff_et_imag_eff_tf*cff_ht_real_tf- cff_ht_real_eff_tf*cff_et_imag_tf)/(q_sq_tf+xb_tf*t_tf))
        final_line = (-1.*(xb_tf**2*(q_sq_tf+t_tf)**2 /(q_sq_tf*(q_sq_tf+xb_tf*t_tf))+0.25*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2*t_tf/(q_sq_tf*self.mp**2*(q_sq_tf+xb_tf*t_tf)))*(cff_e_imag_eff_tf*cff_e_real_tf- cff_e_real_eff_tf*cff_e_imag_tf)-0.25*xb_tf**2*q_sq_tf*t_tf*(cff_et_imag_eff_tf*cff_et_real_tf- cff_et_real_eff_tf*cff_et_imag_tf)/((q_sq_tf+xb_tf*t_tf)*self.mp**2))
        dvcs_imag_curlyc_f_feff = ((first_line + next_line + final_line)* q_sq_tf*(q_sq_tf + xb_tf*t_tf)/((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2)

        # DVCS Re[CurlyC_LP](F | F*)
        # [NOTE]: this one is for c0 LP:
        first_line = 4.*(1.-xb_tf+ ((3.-2.*xb_tf)*q_sq_tf + t_tf)*ep_tf**2/(4.*(q_sq_tf + xb_tf*t_tf)))*(cff_h_real_tf*cff_ht_real_tf- cff_ht_imag_tf*(-cff_h_imag_tf)+ cff_ht_real_tf*cff_h_real_tf- cff_h_imag_tf*(-cff_ht_imag_tf))
        second_line = (-xb_tf**2*(q_sq_tf - xb_tf*t_tf*(1.-2.*xb_tf))*(cff_h_real_tf*cff_et_real_tf- cff_et_imag_tf*(-cff_h_imag_tf) + cff_et_real_tf*cff_h_real_tf- cff_h_imag_tf*(-cff_et_imag_tf)+ cff_ht_real_tf*cff_e_real_tf- cff_e_imag_tf*(-cff_ht_imag_tf)+ cff_e_real_tf*cff_ht_real_tf- cff_ht_imag_tf*(-cff_e_imag_tf))/(q_sq_tf + xb_tf*t_tf))
        third_line = (-xb_tf*(4.*(1.-xb_tf)*(q_sq_tf + xb_tf*t_tf)*t_tf+ ep_tf**2*(q_sq_tf+t_tf)**2)*(cff_ht_real_tf*cff_e_real_tf- cff_e_imag_tf*(-cff_ht_imag_tf)+ cff_e_real_tf*cff_ht_real_tf- cff_ht_imag_tf*(-cff_e_imag_tf))/(2.*q_sq_tf*(q_sq_tf + xb_tf*t_tf)))
        fourth_line = (-xb_tf*((q_sq_tf*(2.-xb_tf) + xb_tf*t_tf)/(q_sq_tf + xb_tf*t_tf))*(xb_tf**2*(q_sq_tf+t_tf)**2/(2.*q_sq_tf*(q_sq_tf*(2.-xb_tf) + xb_tf*t_tf)) + t_tf/(4.*self.mp**2))*(cff_e_real_tf*cff_et_real_tf- cff_e_imag_tf*(-cff_et_imag_tf) + cff_et_real_tf*cff_e_real_tf- cff_et_imag_tf*(-cff_e_imag_tf)))
        dvcs_real_curlyc_lp = ((first_line + second_line + third_line + fourth_line)* q_sq_tf*(q_sq_tf + xb_tf*t_tf)/(tf.sqrt(1.+ep_tf**2)*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2))

        self.debug_print("[DEBUG]: DVCS Re[CurlyC_LP](F | F*): ", dvcs_real_curlyc_lp)

        # DVCS LP Re[CurlyC_LP](F_eff | F*)
        # [NOTE]: Need this for c1 LP:
        first_line = (4.*(1.-xb_tf+ ((3.-2.*xb_tf)*q_sq_tf + t_tf)*ep_tf**2/(4.*(q_sq_tf + xb_tf*t_tf)))*(cff_h_real_eff_tf*cff_ht_real_tf- cff_ht_imag_tf*(-cff_h_imag_tf)+ cff_ht_real_eff_tf*cff_h_real_tf- cff_h_imag_eff_tf*(-cff_ht_imag_tf)))
        second_line = (-xb_tf**2*(q_sq_tf - xb_tf*t_tf*(1.-2.*xb_tf))*(cff_h_real_eff_tf*cff_et_real_tf- cff_et_imag_tf*(-cff_h_imag_tf)+ cff_et_real_eff_tf*cff_h_real_tf- cff_h_imag_eff_tf*(-cff_et_imag_tf)+ cff_ht_real_eff_tf*cff_e_real_tf- cff_e_imag_tf*(-cff_ht_imag_tf)+ cff_e_real_eff_tf*cff_ht_real_tf- cff_ht_imag_eff_tf*(-cff_e_imag_tf))/(q_sq_tf + xb_tf*t_tf))
        third_line = (-xb_tf*(4.*(1.-xb_tf)*(q_sq_tf + xb_tf*t_tf)*t_tf+ ep_tf**2*(q_sq_tf+t_tf)**2)*(cff_ht_real_eff_tf*cff_e_real_tf- cff_e_imag_tf*(-cff_ht_imag_tf)+ cff_e_real_eff_tf*cff_ht_real_tf- cff_ht_imag_eff_tf*(-cff_e_imag_tf))/(2.*q_sq_tf*(q_sq_tf + xb_tf*t_tf)))
        fourth_line = (-xb_tf*((q_sq_tf*(2.-xb_tf) + xb_tf*t_tf)/(q_sq_tf + xb_tf*t_tf))*(xb_tf**2*(q_sq_tf+t_tf)**2/(2.*q_sq_tf*(q_sq_tf*(2.-xb_tf) + xb_tf*t_tf))+ t_tf/(4.*self.mp**2))*(cff_e_real_eff_tf*cff_et_real_tf- cff_et_imag_tf*(-cff_e_imag_tf)+ cff_et_real_eff_tf*cff_e_real_tf- cff_e_imag_eff_tf*(-cff_et_imag_tf)))
        dvcs_real_curlyc_lp_feff_f = ((first_line + second_line + third_line + fourth_line)* q_sq_tf*(q_sq_tf + xb_tf*t_tf)/(tf.sqrt(1.+ep_tf**2)*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2))

        self.debug_print("[DEBUG]: DVCS Re[CurlyC_LP](F_eff | F*): ", dvcs_real_curlyc_lp_feff_f)

        # DVCS LP Im[CurlyC_LP](F_eff | F*)
        # [NOTE]: Need this for s1 LP:
        first_line = (4.*(1.-xb_tf+ ((3.-2.*xb_tf)*q_sq_tf + t_tf)*ep_tf**2/(4.*(q_sq_tf + xb_tf*t_tf)))*(cff_h_imag_eff_tf*cff_ht_real_tf - cff_ht_real_tf*cff_h_imag_tf + cff_ht_imag_eff_tf*cff_h_real_tf- cff_h_real_tf*cff_ht_imag_tf))
        second_line = (-xb_tf**2*(q_sq_tf - xb_tf*t_tf*(1.-2.*xb_tf))*(cff_h_imag_eff_tf*cff_et_real_tf- cff_et_real_tf*cff_h_imag_tf+ cff_et_imag_eff_tf*cff_h_real_tf- cff_h_real_tf*cff_et_imag_tf+ cff_ht_imag_eff_tf*cff_e_real_tf- cff_e_real_tf*cff_ht_imag_tf+ cff_e_imag_eff_tf*cff_ht_real_tf- cff_ht_real_tf*cff_e_imag_tf)/(q_sq_tf + xb_tf*t_tf))
        third_line = (-xb_tf*(4.*(1.-xb_tf)*(q_sq_tf + xb_tf*t_tf)*t_tf+ ep_tf**2*(q_sq_tf+t_tf)**2)*(cff_h_imag_eff_tf*cff_et_real_tf- cff_et_real_tf*cff_h_imag_tf+ cff_et_imag_eff_tf*cff_h_real_tf- cff_h_real_tf*cff_et_imag_tf)/(2.*q_sq_tf*(q_sq_tf + xb_tf*t_tf)))
        fourth_line = (-xb_tf*((q_sq_tf*(2.-xb_tf) + xb_tf*t_tf)/(q_sq_tf + xb_tf*t_tf))*(xb_tf**2*(q_sq_tf+t_tf)**2/(2.*q_sq_tf*(q_sq_tf*(2.-xb_tf) + xb_tf*t_tf))+ t_tf/(4.*self.mp**2))*(cff_e_imag_eff_tf*cff_et_real_tf- cff_et_real_tf*cff_e_imag_tf+ cff_et_imag_eff_tf*cff_e_real_tf- cff_e_real_tf*cff_et_imag_tf))
        dvcs_imag_curlyc_lp_feff_f = ((first_line + second_line + third_line + fourth_line)* q_sq_tf*(q_sq_tf + xb_tf*t_tf)/(tf.sqrt(1.+ep_tf**2)*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2))

        self.debug_print("[DEBUG]: DVCS Im[CurlyC_LP](F_eff | F*): ", dvcs_imag_curlyc_lp_feff_f)
        
        # DVCS: c0(F | F*):
        first_term_prefactor = 2. * ( 2. - 2. * y_lep_tf + y_lep_tf**2 + (ep_tf**2 * y_lep_tf**2 / 2.)) / (1. + ep_tf**2)
        second_term_prefactor = 16. * k_tf**2 / ((2. - xb_tf)**2 * (1. + ep_tf**2))
        dvcs_unp_c0 = first_term_prefactor * dvcs_real_curlyc + second_term_prefactor * dvcs_real_curlyc_feff

        # correct, 2026/08/25
        self.debug_print("[DEBUG]: DVCS c0 unp: ", dvcs_unp_c0)

        # DVCS: c1(Feff | F*):
        prefactor = 8. * k_tf * (2. - y_lep_tf) / ((2. - xb_tf) * (1. + ep_tf**2))
        dvcs_unp_c1 = prefactor * dvcs_real_curlyc_f_eff

        # correct, 2026/08/25
        self.debug_print("[DEBUG]: DVCS c1 unp: ", dvcs_unp_c1)

        # DVCS: s1(Feff | F*):
        prefactor = -8. * k_tf * lep_lambda * y_lep_tf * tf.sqrt(1. + ep_tf**2) / ((2. - xb_tf) * (1. + ep_tf**2))
        dvcs_unp_s1 = prefactor * dvcs_imag_curlyc_f_feff

        # correct, 2026/08/25
        # [NOTE]: This cancellation is analytically zero under the WW prescription
        # The direct float32 evaluation leaves a 1e-7 numerical residue due to
        # subtractive cancellation.
        self.debug_print("[DEBUG]: DVCS s1 unp: ", dvcs_unp_s1)

        # DVCS LP: c0:
        prefactor = 2.*lep_lambda*tgt_lambda*y_lep_tf*(2.-y_lep_tf)/tf.sqrt(1.+ep_tf*ep_tf)
        dvcs_lp_c0 = prefactor * dvcs_real_curlyc_lp

        self.debug_print("[DEBUG]: DVCS c0 LP: ", dvcs_lp_c0)

        # DVCS LP: c1:
        prefactor = 8.*tgt_lambda*k_tf*lep_lambda*y_lep_tf*tf.sqrt(1+ep_tf*ep_tf)/((2.-xb_tf)*(1.+ep_tf*ep_tf))
        dvcs_lp_c1 = prefactor * dvcs_real_curlyc_lp_feff_f

        self.debug_print("[DEBUG]: DVCS c1 LP: ", dvcs_lp_c1)
    
        # DVCS LP: s1:
        prefactor = -8.*tgt_lambda*k_tf*(2.-y_lep_tf)/((2.-xb_tf)*(1.+ep_tf*ep_tf))
        dvcs_lp_s1 = prefactor * dvcs_imag_curlyc_lp_feff_f

        self.debug_print("[DEBUG]: DVCS s1 LP: ", dvcs_lp_s1)

        # Interference: C(n = 0)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        two_minus_xb = 2. - xb_tf
        two_minus_y = 2. - y_lep_tf
        first_term_in_brackets = ktilde_tf**2 * two_minus_y**2 / (q_sq_tf * root_one_plus_epsilon_squared)
        second_term_in_brackets_first_part = t_over_Q_squared * two_minus_xb * (1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.))
        second_term_in_brackets_second_part_numerator = 2. * xb_tf * t_over_Q_squared * (two_minus_xb + 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep_tf**2 / xb_tf) + ep_tf**2
        second_term_in_brackets_second_part =  1. + second_term_in_brackets_second_part_numerator / (two_minus_xb * one_plus_root_epsilon_stuff)
        prefactor = -4. * two_minus_y * one_plus_root_epsilon_stuff / tf.pow(root_one_plus_epsilon_squared, 4)
        c_0_plus_plus_unp = prefactor * (first_term_in_brackets + second_term_in_brackets_first_part * second_term_in_brackets_second_part)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 0)++ unp: ", c_0_plus_plus_unp)

        # Interference: CV(n = 0)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_term_in_brackets = (2. - y_lep_tf)**2 * ktilde_tf**2 / (root_one_plus_epsilon_squared * q_sq_tf)
        second_term_first_multiplicative_term = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        second_term_second_multiplicative_term = one_plus_root_epsilon_stuff / 2.
        second_term_third_multiplicative_term = 1. + t_over_Q_squared
        second_term_fourth_multiplicative_term = 1. + (root_one_plus_epsilon_squared - 1. + (2. * xb_tf)) * t_over_Q_squared / one_plus_root_epsilon_stuff
        second_term_in_brackets = second_term_first_multiplicative_term * second_term_second_multiplicative_term * second_term_third_multiplicative_term * second_term_fourth_multiplicative_term
        coefficient_prefactor = 8. * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_0_plus_plus_V_unp = coefficient_prefactor * (first_term_in_brackets + second_term_in_brackets)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 0)++ unp: ", c_0_plus_plus_V_unp)

        # Interference: CA(n = 0):
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        two_minus_y = 2. - y_lep_tf
        ktilde_over_Q_squared = ktilde_tf**2 / q_sq_tf
        curly_bracket_first_term = two_minus_y**2 * ktilde_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf) / (2. * root_one_plus_epsilon_squared)
        deepest_parentheses_term = (xb_tf * (2. + one_plus_root_epsilon_stuff - 2. * xb_tf) / one_plus_root_epsilon_stuff + (one_plus_root_epsilon_stuff - 2.)) * t_over_Q_squared
        square_bracket_term = one_plus_root_epsilon_stuff * (one_plus_root_epsilon_stuff - xb_tf + deepest_parentheses_term) / 2. - (2. * ktilde_over_Q_squared)
        curly_bracket_second_term = (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * square_bracket_term
        coefficient_prefactor = 8. * two_minus_y * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_0_plus_plus_A_unp = coefficient_prefactor * (curly_bracket_first_term + curly_bracket_second_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 0)++ unp: ", c_0_plus_plus_A_unp)

        # Interference: C(n = 0)0+:
        bracket_quantity = ep_tf**2 + t_tf * (2. - 6.* xb_tf - ep_tf**2) / (3. * q_sq_tf)
        prefactor = 12. * tf.sqrt(2.) * k_tf * (2. - y_lep_tf) * tf.sqrt(1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4)) / tf.pow(1. + ep_tf**2, 2.5)
        c_0_zero_plus_unp = prefactor * bracket_quantity

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 0)0+ unp: ", c_0_zero_plus_unp)

        # Interference: CV(n = 0)0+:
        t_over_Q_squared = t_tf / q_sq_tf
        main_part = xb_tf * t_over_Q_squared * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)
        prefactor = 24. * tf.sqrt(2.) * k_tf * (2. - y_lep_tf) * tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / (1. + ep_tf**2)**2.5
        c_0_zero_plus_V_unp = prefactor * main_part

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 0)0+ unp: ", c_0_zero_plus_V_unp)

        # Interference: CA(n = 0)0+:
        t_over_Q_squared = t_tf / q_sq_tf
        fancy_xb_epsilon_term = 8. - 6. * xb_tf + 5. * ep_tf**2
        brackets_term = 1. - t_over_Q_squared * (2. - 12. * xb_tf * (1. - xb_tf) - ep_tf**2) / fancy_xb_epsilon_term
        prefactor = 4. * tf.sqrt(2.) * k_tf * (2. - y_lep_tf) * tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / tf.pow(1. + ep_tf**2, 2.5)
        c_0_zero_plus_A_unp = prefactor * t_over_Q_squared * fancy_xb_epsilon_term * brackets_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 0)0+ unp: ", c_0_zero_plus_A_unp)

        # Interference: C(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_bracket_first_term = (1. + (1. - xb_tf) * (root_one_plus_epsilon_squared - 1.) / (2. * xb_tf) + ep_tf**2 / (4. * xb_tf)) * xb_tf * t_over_Q_squared
        first_bracket_term = first_bracket_first_term - 3. * ep_tf**2 / 4.
        second_bracket_term = 1. - (1. - 3. * xb_tf) * t_over_Q_squared + (1. - root_one_plus_epsilon_squared + 3. * ep_tf**2) * xb_tf * t_over_Q_squared / (one_plus_root_epsilon_stuff - ep_tf**2)
        fancy_y_coefficient = 2. - 2. * y_lep_tf + y_lep_tf**2 + ep_tf**2 * y_lep_tf**2 / 2.
        second_term = -4. * k_tf * fancy_y_coefficient * (one_plus_root_epsilon_stuff - ep_tf**2) * second_bracket_term / root_one_plus_epsilon_squared**5
        first_term = -16. * k_tf * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * first_bracket_term / root_one_plus_epsilon_squared**5
        c_1_plus_plus_unp = first_term + second_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 1)++ unp: ", c_1_plus_plus_unp)

        # Interference: CV(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        first_bracket_term = (2. - y_lep_tf)**2 * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)
        second_bracket_term_first_part = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        second_bracket_term_second_part = 0.5 * (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * tprime_tf / q_sq_tf
        coefficient_prefactor = 16. * k_tf * xb_tf * t_over_Q_squared / tf.pow(root_one_plus_epsilon_squared, 5)
        c_1_plus_plus_V_unp = coefficient_prefactor * (first_bracket_term + second_bracket_term_first_part * second_bracket_term_second_part)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 1)++ unp: ", c_1_plus_plus_V_unp)

        # Interference: CA(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        one_minus_xb = 1. - xb_tf
        one_minus_2xb = 1. - 2. * xb_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        first_bracket_term_second_part = 1. - one_minus_2xb * t_over_Q_squared + (4. * xb_tf * one_minus_xb + ep_tf**2) * t_prime_over_Q_squared / (4. * root_one_plus_epsilon_squared)
        second_bracket_term = 1. - 0.5 * xb_tf + 0.25 * (one_minus_2xb + root_one_plus_epsilon_squared) * (1. - t_over_Q_squared) + (4. * xb_tf * one_minus_xb + ep_tf**2) * t_prime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
        prefactor = -16. * k_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_1_plus_plus_A_unp = prefactor * (fancy_y_stuff * first_bracket_term_second_part - (2. - y_lep_tf)**2 * second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 1)++ unp: ", c_1_plus_plus_A_unp)

        # Interference: C(n = 1)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        one_minus_xb = 1. - xb_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        first_bracket_term = (2. - y_lep_tf)**2 * t_prime_over_Q_squared * (one_minus_xb + (one_minus_xb * xb_tf + (ep_tf**2 / 4.)) * t_prime_over_Q_squared / root_one_plus_epsilon_squared)
        second_bracket_term = y_quantity * (1. - (1. - 2. * xb_tf) * t_over_Q_squared) * (ep_tf**2 - 2. * (1. + (ep_tf**2 / (2. * xb_tf))) * xb_tf * t_over_Q_squared) / root_one_plus_epsilon_squared
        prefactor = 8. * tf.sqrt(2. * y_quantity) / root_one_plus_epsilon_squared**4
        c_1_zero_plus_unp = prefactor * (first_bracket_term + second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 1)0+ unp: ", c_1_zero_plus_unp)

        # Interference: CV(n = 1)0+:
        t_over_Q_squared = t_tf / q_sq_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        major_part = (2 - y_lep_tf)**2 * ktilde_tf**2 / q_sq_tf + (1. - (1. - 2. * xb_tf) * t_over_Q_squared)**2 * y_quantity
        prefactor = 16. * tf.sqrt(2. * y_quantity) * xb_tf * t_over_Q_squared / (1. + ep_tf**2)**2.5
        c_1_zero_plus_V_unp = prefactor * major_part

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 1)0+ unp: ", c_1_zero_plus_V_unp)

        # Interference: CA(n = 1)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_minus_2xb = 1. - 2. * xb_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        second_term_first_part = (1. - one_minus_2xb * t_over_Q_squared) * y_quantity
        second_term_second_part = 4. - 2. * xb_tf + 3. * ep_tf**2 + t_over_Q_squared * (4. * xb_tf * (1. - xb_tf) + ep_tf**2)
        first_term = ktilde_tf**2 * one_minus_2xb * (2. - y_lep_tf)**2 / q_sq_tf
        prefactor = 8. * tf.sqrt(2. * y_quantity) * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_1_zero_plus_A_unp = prefactor * (first_term + second_term_first_part * second_term_second_part)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 1)0+ unp: ", c_1_zero_plus_A_unp)

        # Interference: C(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        first_bracket_term = 2. * ep_tf**2 * ktilde_tf**2 / (root_one_plus_epsilon_squared * (1. + root_one_plus_epsilon_squared) * q_sq_tf)
        second_bracket_term = xb_tf * tprime_tf * t_over_Q_squared * (1. - xb_tf - 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep_tf**2 / xb_tf) / q_sq_tf
        prefactor = 8. * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) / root_one_plus_epsilon_squared**4
        c_2_plus_plus_unp = prefactor * (first_bracket_term + second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 2)++ unp: ", c_2_plus_plus_unp)

        # Interference: CV(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        major_term = (4. * ktilde_tf**2 / (root_one_plus_epsilon_squared * q_sq_tf)) + 0.5 * (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * (1. + t_over_Q_squared) * t_prime_over_Q_squared
        prefactor = 8. * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_2_plus_plus_V_unp = prefactor * major_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 2)++ unp: ", c_2_plus_plus_V_unp)

        # Interference: CA(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        first_bracket_term = 4. * (1. - 2. * xb_tf) * ktilde_tf**2 / (root_one_plus_epsilon_squared * q_sq_tf)
        second_bracket_term = (3.  - root_one_plus_epsilon_squared - 2. * xb_tf + ep_tf**2 / xb_tf ) * xb_tf * t_prime_over_Q_squared
        prefactor = 4. * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_2_plus_plus_A_unp = prefactor * (first_bracket_term - second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 2)++ unp: ", c_2_plus_plus_A_unp)

        # Interference: C(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        epsilon_squared_over_2 = ep_tf**2 / 2.
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        bracket_term = 1. + ((1. + epsilon_squared_over_2 / xb_tf) / (1. + epsilon_squared_over_2)) * xb_tf * t_tf / q_sq_tf
        prefactor = -8. * tf.sqrt(2. * y_quantity) * k_tf * (2. - y_lep_tf) / root_one_plus_epsilon_squared**5
        c_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 2)0+ unp: ", c_2_zero_plus_unp)

        # Interference: CV(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        y_quantity = tf.sqrt(1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * y_quantity * k_tf * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_2_zero_plus_V_unp = prefactor * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 2)0+ unp: ", c_2_zero_plus_V_unp)

        # Interference: CA(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        one_minus_xb = 1. - xb_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        bracket_term = one_minus_xb + 0.5 * t_prime_over_Q_squared * (4. * xb_tf * one_minus_xb + ep_tf**2) / root_one_plus_epsilon_squared
        prefactor = 8. * tf.sqrt(2. * y_quantity) * k_tf * (2. - y_lep_tf) * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_2_zero_plus_A_unp = prefactor * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 2)0+ unp: ", c_2_zero_plus_A_unp)

        # Interference: C(n = 3)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        major_term = (1. - xb_tf) * t_over_Q_squared + 0.5 * (root_one_plus_epsilon_squared - 1.) * (1. + t_over_Q_squared)
        intermediate_term = (root_one_plus_epsilon_squared - 1.) / root_one_plus_epsilon_squared**5
        prefactor = -8. * k_tf * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.)
        c_3_plus_plus_unp = prefactor * intermediate_term * major_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 3)++ unp: ", c_3_plus_plus_unp)

        # Interference: CV(n = 3)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        major_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * t_over_Q_squared
        prefactor = -8. * k_tf * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_3_plus_plus_V_unp = prefactor * major_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 3)++ unp: ", c_3_plus_plus_V_unp)

        # Interference: CA(n = 3)++:
        main_term = t_tf * tprime_tf * (xb_tf * (1. - xb_tf) + ep_tf**2 / 4.) / q_sq_tf**2
        prefactor = 16. * k_tf * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) / (1. + ep_tf**2)**2.5
        c_3_plus_plus_A_unp = prefactor * main_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 3)++ unp: ", c_3_plus_plus_A_unp)

        # Interference: C(n = 3)0+:
        c_3_zero_plus_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 3)0+ unp: ", c_3_zero_plus_unp)

        # Interference: CV(n = 3)0+:
        c_3_zero_plus_V_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 3)0+ unp: ", c_3_zero_plus_V_unp)

        # Interference: CA(n = 3)0+:
        c_3_zero_plus_A_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 3)0+ unp: ", c_3_zero_plus_A_unp)

        # Interference: S(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        tPrime_over_Q_squared = tprime_tf / q_sq_tf
        bracket_term = 1. + ((1. - xb_tf + 0.5 * (root_one_plus_epsilon_squared - 1.)) / root_one_plus_epsilon_squared**2) * tPrime_over_Q_squared
        prefactor = 8. * lep_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) / root_one_plus_epsilon_squared**2
        s_1_plus_plus_unp = prefactor * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 1)++ unp: ", s_1_plus_plus_unp)

        # Interference: SV(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        bracket_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * t_over_Q_squared
        prefactor = -8. * lep_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        s_1_plus_plus_V_unp = prefactor * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 1)++ unp: ", s_1_plus_plus_V_unp)

        # Interference: SA(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        tPrime_over_Q_squared = tprime_tf / q_sq_tf
        one_minus_2xb = 1. - 2. * xb_tf
        bracket_term = 1. - one_minus_2xb * (one_minus_2xb + root_one_plus_epsilon_squared) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
        prefactor = 8. * lep_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) * t_over_Q_squared / root_one_plus_epsilon_squared**2
        s_1_plus_plus_A_unp = prefactor * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 1)++ unp: ", s_1_plus_plus_A_unp)

        # Interference: S(n = 1)0+:
        root_one_plus_epsilon_squared = (1. + ep_tf**2)**2
        y_quantity = tf.sqrt(1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.))
        s_1_zero_plus_unp = 8. * tf.sqrt(2.) * lep_lambda * (2. - y_lep_tf) * y_lep_tf * y_quantity * ktilde_tf**2 / (root_one_plus_epsilon_squared * q_sq_tf)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 1)0+ unp: ", s_1_zero_plus_unp)

        # Interference: SV(n = 1)0+:
        one_plus_epsilon_squared_squared = (1. + ep_tf**2)**2
        t_over_Q_squared = t_tf / q_sq_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        bracket_term = 4. * (1. - 2. * xb_tf) * t_over_Q_squared * (1. + xb_tf * t_over_Q_squared) + ep_tf**2 * (1. + t_over_Q_squared)**2
        prefactor = 4. * tf.sqrt(2. * fancy_y_stuff) * lep_lambda * y_lep_tf * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / one_plus_epsilon_squared_squared
        s_1_zero_plus_V_unp = prefactor * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 1)0+ unp: ", s_1_zero_plus_V_unp)

        # Interference: SA(n = 1)0+:
        one_plus_epsilon_squared_squared = (1. + ep_tf**2)**2
        fancy_y_stuff = tf.sqrt(1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.)
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * y_lep_tf * (2. - y_lep_tf) * (1. - 2. * xb_tf) / one_plus_epsilon_squared_squared
        s_1_zero_plus_A_unp = prefactor * fancy_y_stuff * t_tf * k_tf**2 / q_sq_tf

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 1)0+ unp: ", s_1_zero_plus_A_unp)

        # Interference: S(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        tPrime_over_Q_squared = tprime_tf / q_sq_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        first_bracket_term = (ep_tf**2 - xb_tf * (root_one_plus_epsilon_squared - 1.)) / (1. + root_one_plus_epsilon_squared - 2. * xb_tf)
        second_bracket_term = (2. * xb_tf + ep_tf**2) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
        prefactor = -4. * lep_lambda * fancy_y_stuff * y_lep_tf * (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * tPrime_over_Q_squared / root_one_plus_epsilon_squared**3
        s_2_plus_plus_unp = prefactor * (first_bracket_term - second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 2)++ unp: ", s_2_plus_plus_unp)

        # Interference: SV(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        one_minus_2xb = 1. - 2. * xb_tf
        bracket_term = root_one_plus_epsilon_squared - 1. + (one_minus_2xb + root_one_plus_epsilon_squared) * t_over_Q_squared
        parentheses_term = 1. - one_minus_2xb * t_over_Q_squared
        prefactor = -4. * lep_lambda * fancy_y_stuff * y_lep_tf * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        s_2_plus_plus_V_unp = prefactor * parentheses_term * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 2)++ unp: ", s_2_plus_plus_V_unp)

        # Interference: SA(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        tPrime_over_Q_squared = tprime_tf / q_sq_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        last_term = 1. + (4. * (1. - xb_tf) * xb_tf + ep_tf**2) * t_over_Q_squared / (4. - 2. * xb_tf + 3. * ep_tf**2)
        middle_term = 1. + root_one_plus_epsilon_squared - 2. * xb_tf
        prefactor = -8. * lep_lambda * fancy_y_stuff * y_lep_tf * t_over_Q_squared * tPrime_over_Q_squared / root_one_plus_epsilon_squared**4
        s_2_plus_plus_A_unp = prefactor * middle_term * last_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 2)++ unp: ", s_2_plus_plus_A_unp)

        # Interference: S(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        epsilon_squared_over_2 = ep_tf**2 / 2.
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        bracket_term = 1. + ((1. + epsilon_squared_over_2 / xb_tf) / (1. + epsilon_squared_over_2)) * xb_tf * t_tf / q_sq_tf
        prefactor = 8. * lep_lambda * tf.sqrt(2. * y_quantity) * k_tf * y_lep_tf / root_one_plus_epsilon_squared**4
        s_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 2)0+ unp: ", s_2_zero_plus_unp)

        # Interference: SV(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        y_quantity = tf.sqrt(1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * y_quantity * k_tf * y_lep_tf * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        s_2_zero_plus_V_unp = prefactor * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 2)0+ unp: ", s_2_zero_plus_V_unp)

        # Interference: SA(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_minus_xb = 1. - xb_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        main_term = 4. * one_minus_xb + 2. * ep_tf**2 + 4. * t_over_Q_squared * (4. * xb_tf * one_minus_xb + ep_tf**2)
        prefactor = -2. * tf.sqrt(2. * y_quantity) * lep_lambda * k_tf * y_lep_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        s_2_zero_plus_A_unp = prefactor * main_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 2)0+ unp: ", s_2_zero_plus_A_unp)

        # Interference: S(n = 3)++:
        s_3_plus_plus_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 3)++ unp: ", s_3_plus_plus_unp)

        # Interference: SV(n = 3)++:
        s_3_plus_plus_V_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 3)++ unp: ", s_3_plus_plus_V_unp)

        # Interference: SA(n = 3)++:
        s_3_plus_plus_A_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 3)++ unp: ", s_3_plus_plus_A_unp)

        # Interference: S(n = 3)0+:
        s_3_zero_plus_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 3)0+ unp: ", s_3_zero_plus_unp)

        # Interference: SV(n = 3)0+:
        s_3_zero_plus_V_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 3)0+ unp: ", s_3_zero_plus_V_unp)

        # Interference: SA(n = 3)0+:
        s_3_zero_plus_A_unp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 3)0+ unp: ", s_3_zero_plus_A_unp)

        # Interference: C(n = 0)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf 
        first_bracket_term = (2. - y_lep_tf)**2 * ktilde_tf**2 / q_sq_tf
        second_bracket_term_first_part = 1. - y_lep_tf + (ep_tf**2 * y_lep_tf**2 / 4.)
        second_bracket_term_second_part = xb_tf * t_over_Q_squared - (ep_tf**2 * (1. - t_over_Q_squared) / 2.)
        second_bracket_term_third_part = 1. + t_over_Q_squared * ((root_one_plus_epsilon_squared - 1. + 2. * xb_tf) / (1. + root_one_plus_epsilon_squared))
        second_bracket_term = second_bracket_term_first_part * second_bracket_term_second_part * second_bracket_term_third_part
        prefactor = -4. * lep_lambda * tgt_lambda * y_lep_tf * (1. + root_one_plus_epsilon_squared) / root_one_plus_epsilon_squared**5
        c_0_plus_plus_lp = prefactor * (first_bracket_term + second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 0)++ LP: ", c_0_plus_plus_lp)

        # Interference: CV(n = 0)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_bracket_term = (2. - y_lep_tf)**2 * (one_plus_root_epsilon_stuff - 2. * xb_tf) * ktilde_tf**2 / (q_sq_tf * one_plus_root_epsilon_stuff)
        second_bracket_term_first_part = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        second_bracket_term_second_part = 2. - xb_tf + 3. * ep_tf**2 / 2
        second_bracket_term_third_part = 1. + (t_over_Q_squared * (4. * (1. - xb_tf) * xb_tf + ep_tf**2) / (4. - 2. * xb_tf + 3. * ep_tf**2))
        second_bracket_term_fourth_part = 1. + (t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. + 2. * xb_tf) / one_plus_root_epsilon_stuff)
        second_bracket_term = second_bracket_term_first_part * second_bracket_term_second_part * second_bracket_term_third_part * second_bracket_term_fourth_part
        prefactor = 4. * lep_lambda * tgt_lambda * y_lep_tf * one_plus_root_epsilon_stuff * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_0_plus_plus_V_lp = prefactor * (first_bracket_term + second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 0)++ LP: ", c_0_plus_plus_V_lp)

        # Interference: CA(n = 0)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_bracket_term = 2. * (2. - y_lep_tf)**2 * ktilde_tf**2 / q_sq_tf
        second_bracket_term_first_part = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        second_bracket_term_second_part = 1. - (1. - 2. * xb_tf) * t_over_Q_squared
        second_bracket_term_third_part = 1. + (t_over_Q_squared * (root_one_plus_epsilon_squared - 1. + 2. * xb_tf) / one_plus_root_epsilon_stuff)
        second_bracket_term = second_bracket_term_first_part * one_plus_root_epsilon_stuff * second_bracket_term_second_part * second_bracket_term_third_part
        prefactor = 4. * lep_lambda * tgt_lambda * y_lep_tf * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_0_plus_plus_A_lp = prefactor * (first_bracket_term + second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 0)++ LP: ", c_0_plus_plus_A_lp)

        # Interference: C(n = 0)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * (1. - xb_tf) * y_lep_tf / (1. + ep_tf**2)**2
        c_0_zero_plus_lp = prefactor * root_combination_of_y_and_epsilon * t_tf / q_sq_tf

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 0)0+ LP: ", c_0_zero_plus_lp)

        # Interference: CV(n = 0)0+ LP:
        modulating_factor = (xb_tf - (t_tf * (1. - 2. * xb_tf) / q_sq_tf)) / (1. - xb_tf)
        c_0_zero_plus_V_lp = c_0_zero_plus_lp * modulating_factor

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 0)0+ LP: ", c_0_zero_plus_V_lp)

        # Interference: CA(n = 0)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * y_lep_tf / (1. + ep_tf**2)**2
        t_over_Q_squared = t_tf / q_sq_tf
        c_0_zero_plus_A_lp = prefactor * root_combination_of_y_and_epsilon * xb_tf * t_over_Q_squared * (1. + t_over_Q_squared)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 0)0+ LP: ", c_0_zero_plus_A_lp)

        # Interference: C(n = 1)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        one_plus_root_epsilon_minus_epsilon_squared = one_plus_root_epsilon_stuff - ep_tf**2
        major_factor = 1. - ((t_tf / q_sq_tf) * (1. - 2. * xb_tf * (one_plus_root_epsilon_stuff + 1.) / one_plus_root_epsilon_minus_epsilon_squared))
        prefactor = -4. * lep_lambda * tgt_lambda * y_lep_tf * k_tf * (2. - y_lep_tf) / root_one_plus_epsilon_squared**5
        c_1_plus_plus_lp = prefactor * one_plus_root_epsilon_minus_epsilon_squared * major_factor

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 1)++ LP: ", c_1_plus_plus_lp)

        # Interference: CV(n = 1)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_minus_xb = 1. - xb_tf
        root_epsilon_and_xb_quantity = root_one_plus_epsilon_squared + 2. * one_minus_xb
        bracket_factor_numerator = 1. + ((1. - ep_tf**2) / root_one_plus_epsilon_squared) - (2. * xb_tf * (1. + (4. * one_minus_xb / root_one_plus_epsilon_squared)))
        bracket_factor_denominator = 2. * root_epsilon_and_xb_quantity
        bracket_factor = 1. - (tprime_tf * bracket_factor_numerator / (q_sq_tf * bracket_factor_denominator))
        prefactor = 8. * lep_lambda * tgt_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) / root_one_plus_epsilon_squared**4
        c_1_plus_plus_V_lp = prefactor * root_epsilon_and_xb_quantity * t_tf * bracket_factor / q_sq_tf

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 1)++ LP: ", c_1_plus_plus_V_lp)

        # Interference: CA(n = 1)++ LP:
        t_over_Q_squared = t_tf / q_sq_tf
        major_factor = xb_tf * t_over_Q_squared * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)
        prefactor = 16. * lep_lambda * tgt_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) / tf.sqrt(1. + ep_tf**2)**5
        c_1_plus_plus_A_lp = prefactor * major_factor

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 1)++ LP: ", c_1_plus_plus_A_lp)

        # Interference: C(n = 1)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * (1. - y_lep_tf) * y_lep_tf / (1. + ep_tf**2)**2
        c_1_zero_plus_lp = prefactor * root_combination_of_y_and_epsilon * ktilde_tf**2 / q_sq_tf

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 1)0+ LP: ", c_1_zero_plus_lp)

        # Interference: CV(n = 1)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * lep_lambda * tgt_lambda  * (2. - y_lep_tf) * y_lep_tf / (1. + ep_tf**2)**2
        c_1_zero_plus_V_lp = prefactor * root_combination_of_y_and_epsilon * t_tf * ktilde_tf**2 / q_sq_tf**2

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 1)0+ LP: ", c_1_zero_plus_V_lp)

        # Interference: CA(n = 1)0+ LP:
        c_1_zero_plus_A_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 1)0+ LP: ", c_1_zero_plus_A_lp)

        # Interference: C(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_multiplicative_factor = (-1. * one_plus_root_epsilon_stuff + 2.) - t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf)
        second_multiplicative_factor = xb_tf * t_over_Q_squared - (ep_tf**2 * (1. - t_over_Q_squared) / 2.)
        prefactor = -4. * lep_lambda * tgt_lambda * y_lep_tf * (1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / root_one_plus_epsilon_squared**5
        c_2_plus_plus_lp = prefactor * first_multiplicative_factor * second_multiplicative_factor

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 2)++ LP: ", c_2_plus_plus_lp)

        # Interference: CV(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_multiplicative_factor = (one_plus_root_epsilon_stuff - 2.) + t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf)
        second_multiplicative_factor = 1. + (t_over_Q_squared * (4. * (1. - xb_tf) * xb_tf + ep_tf**2 ) / (4. - 2. * xb_tf + 3. * ep_tf**2))
        third_multiplicative_factor = t_over_Q_squared * (4. - 2. * xb_tf + 3. * ep_tf**2)
        prefactor = -2.*lep_lambda*tgt_lambda*y_lep_tf*(1.-y_lep_tf-(y_lep_tf**2 * ep_tf**2 / 4.)) / root_one_plus_epsilon_squared**5
        c_2_plus_plus_V_lp = prefactor * first_multiplicative_factor * second_multiplicative_factor * third_multiplicative_factor

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 2)++ LP: ", c_2_plus_plus_V_lp)

        # Interference: CA(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_multiplicative_factor = (1. - root_one_plus_epsilon_squared) - t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf)
        second_multiplicative_factor = xb_tf * t_over_Q_squared * (1. - t_over_Q_squared * (1. - 2. * xb_tf))
        prefactor = 4. * lep_lambda * tgt_lambda * y_lep_tf * (1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / root_one_plus_epsilon_squared**5
        c_2_plus_plus_A_lp = prefactor * first_multiplicative_factor * second_multiplicative_factor

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 2)++ LP: ", c_2_plus_plus_A_lp)

        # Interference: C(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * y_lep_tf / (1. + ep_tf**2)**2
        c_2_zero_plus_lp = prefactor * root_combination_of_y_and_epsilon * (1. + (xb_tf * t_tf / q_sq_tf))

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 2)0+ LP: ", c_2_zero_plus_lp)
    
        # Interference: CV(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * y_lep_tf / (1. + ep_tf**2)**2
        c_2_zero_plus_V_lp = prefactor * root_combination_of_y_and_epsilon * (1. - xb_tf ) * t_tf / q_sq_tf

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 2)0+ LP: ", c_2_zero_plus_V_lp)

        # Interference: CA(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * y_lep_tf / (1. + ep_tf**2)**2
        t_over_Q_squared = t_tf / q_sq_tf
        c_2_zero_plus_A_lp = prefactor * root_combination_of_y_and_epsilon * xb_tf * t_over_Q_squared * (1. + t_tf / q_sq_tf)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 2)0+ LP: ", c_2_zero_plus_A_lp)

        # Interference: C(n = 3)++ LP:
        c_3_plus_plus_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 3)++ LP: ", c_3_plus_plus_lp)

        # Interference: CV(n = 3)++ LP:
        c_3_plus_plus_V_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 3)++ LP: ", c_3_plus_plus_V_lp)

        # Interference: CA(n = 3)++ LP:
        c_3_plus_plus_A_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 3)++ LP: ", c_3_plus_plus_A_lp)

        # Interference: C(n = 3)0+ LP:
        c_3_zero_plus_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: C(n = 3)0+ LP: ", c_3_zero_plus_lp)

        # Interference: CV(n = 3)0+ LP:
        c_3_zero_plus_V_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CV(n = 3)0+ LP: ", c_3_zero_plus_V_lp)

        # Interference: CA(n = 3)0+ LP:
        c_3_zero_plus_A_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: CA(n = 3)0+ LP: ", c_3_zero_plus_A_lp)

        # Interference: S(n = 1)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        t_over_Q_squared = t_tf / q_sq_tf
        epsilon_y_over_2_squared = (ep_tf * y_lep_tf / 2.) ** 2
        first_bracket_term = 2. * root_one_plus_epsilon_squared - 1. + (t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf) / one_plus_root_epsilon_stuff)
        second_bracket_term = (3. * ep_tf**2 / 2.) + (t_over_Q_squared * (1. - root_one_plus_epsilon_squared - ep_tf**2 / 2. - xb_tf * (3.  - root_one_plus_epsilon_squared)))
        almost_prefactor = 4. * tgt_lambda * k_tf / root_one_plus_epsilon_squared**6
        prefactor_one = almost_prefactor * (2. - 2. * y_lep_tf + y_lep_tf**2 + 2. * epsilon_y_over_2_squared) * one_plus_root_epsilon_stuff
        prefactor_two = 2. * almost_prefactor * (1. - y_lep_tf - epsilon_y_over_2_squared)
        s_1_plus_plus_lp = prefactor_one * first_bracket_term + prefactor_two * second_bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 1)++ LP: ", s_1_plus_plus_lp)

        # Interference: SV(n = 1)++ LP:
        ep_squared = ep_tf**2
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_squared)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        epsilon_y_over_2_squared = ep_squared * y_lep_tf**2 / 4.
        first_bracket_term = 1. - (t_prime_over_Q_squared * ((1. - 2. * xb_tf) * (1. - 2. * xb_tf + root_one_plus_epsilon_squared)) / (2. * root_one_plus_epsilon_squared**2))
        second_term_parentheses_term = t_over_Q_squared * (1. - (xb_tf * ((3. + root_one_plus_epsilon_squared) / 4.)) + (5. * ep_squared / 8.))
        second_bracket_term_numerator = 1. - root_one_plus_epsilon_squared + (ep_squared / 2.) - (2. * xb_tf * (3. * (1. - xb_tf) - root_one_plus_epsilon_squared))
        second_bracket_term_denominator = 4. - (xb_tf * (root_one_plus_epsilon_squared + 3.)) + (5. * ep_squared / 2.)
        second_bracket_term = 1. - (t_over_Q_squared * second_bracket_term_numerator / second_bracket_term_denominator)
        almost_prefactor = 8. * tgt_lambda * k_tf / root_one_plus_epsilon_squared**4
        prefactor_one = almost_prefactor * (2. - 2. * y_lep_tf + y_lep_tf**2 + 2. * epsilon_y_over_2_squared) * t_over_Q_squared
        prefactor_two = 4. * almost_prefactor * (1. - y_lep_tf - epsilon_y_over_2_squared) / root_one_plus_epsilon_squared**2
        s_1_plus_plus_V_lp = prefactor_one * first_bracket_term + prefactor_two * second_term_parentheses_term * second_bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 1)++ LP: ", s_1_plus_plus_V_lp)

        # Interference: SA(n = 1)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        xB_t_over_Q_squared = xb_tf * t_over_Q_squared
        three_plus_root_epsilon_stuff = 3 + root_one_plus_epsilon_squared
        epsilon_y_over_2_squared = (ep_tf * y_lep_tf / 2.) ** 2
        almost_prefactor = 8. * tgt_lambda * k_tf / root_one_plus_epsilon_squared**6
        first_bracket_term = root_one_plus_epsilon_squared - 1. + (t_over_Q_squared * (1. + root_one_plus_epsilon_squared - 2. * xb_tf))
        second_bracket_term = 1. - (t_over_Q_squared * (3.  - root_one_plus_epsilon_squared - 6. * xb_tf) / three_plus_root_epsilon_stuff)
        prefactor_one = -1. * almost_prefactor * (2. - 2. * y_lep_tf + y_lep_tf**2 + 2. * epsilon_y_over_2_squared) * xB_t_over_Q_squared
        prefactor_two = almost_prefactor * (1. - y_lep_tf - epsilon_y_over_2_squared) * three_plus_root_epsilon_stuff * xB_t_over_Q_squared
        s_1_plus_plus_A_lp = prefactor_one * first_bracket_term + prefactor_two * second_bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 1)++ LP: ", s_1_plus_plus_A_lp)

        # Interference: S(n = 1)0+ LP:
        combination_of_y_and_epsilon = 1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)
        t_over_Q_squared = t_tf / q_sq_tf
        first_bracket_term = ktilde_tf**2 * (2. - y_lep_tf)**2 / q_sq_tf
        second_bracket_term = (1. + t_over_Q_squared) * combination_of_y_and_epsilon * (2. * xb_tf * t_over_Q_squared - (ep_tf**2 * (1. - t_over_Q_squared)))
        prefactor = 8. * tf.sqrt(2.) * tgt_lambda  * tf.sqrt(combination_of_y_and_epsilon) / tf.sqrt((1. + ep_tf**2)**5)
        s_1_zero_plus_lp = prefactor * (first_bracket_term + second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 1)0+ LP: ", s_1_zero_plus_lp)

        # Interference: SV(n = 1)0+ LP:
        combination_of_y_and_epsilon = 1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)
        t_over_Q_squared = t_tf / q_sq_tf
        first_bracket_term = ktilde_tf**2 * (2. - y_lep_tf)**2 / q_sq_tf
        second_bracket_term_long = 4. - 2. * xb_tf + 3. * ep_tf**2 + t_over_Q_squared * (4. * xb_tf * (1. - xb_tf) + ep_tf**2)
        second_bracket_term = (1. + t_over_Q_squared) * combination_of_y_and_epsilon * second_bracket_term_long
        prefactor = -8. * tf.sqrt(2.) * tgt_lambda  * tf.sqrt(combination_of_y_and_epsilon) * t_over_Q_squared / tf.sqrt((1. + ep_tf**2)**5)
        s_1_zero_plus_V_lp = prefactor * (first_bracket_term + second_bracket_term)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 1)0+ LP: ", s_1_zero_plus_V_lp)

        # Interference: SA(n = 1)0+ LP:
        combination_of_y_and_epsilon_to_3_halves = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))**3
        t_over_Q_squared = t_tf / q_sq_tf
        prefactor = -16. * tf.sqrt(2.) * tgt_lambda * xb_tf * t_over_Q_squared * (1. + t_over_Q_squared) / tf.sqrt((1. + ep_tf**2)**5)
        s_1_zero_plus_A_lp = prefactor * combination_of_y_and_epsilon_to_3_halves * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 1)0+ LP: ", s_1_zero_plus_A_lp)

        # Interference: S(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        bracket_term = 4. * ktilde_tf**2 * (one_plus_root_epsilon_stuff - 2. * xb_tf) * (one_plus_root_epsilon_stuff + xb_tf * t_tf / q_sq_tf) * tprime_tf / (root_one_plus_epsilon_squared * q_sq_tf**2)
        prefactor = -4. * tgt_lambda * (2. - y_lep_tf) * (1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)) / root_one_plus_epsilon_squared**5
        s_2_plus_plus_lp = prefactor * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 2)++ LP: ", s_2_plus_plus_lp)

        # Interference: SV(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        bracket_term_second_term = (3.  - root_one_plus_epsilon_squared - (2. * xb_tf) + (ep_tf**2 / xb_tf)) * xb_tf * tprime_tf / q_sq_tf
        bracket_term_first_term = 4. * ktilde_tf**2 * (1. - 2. * xb_tf) / (root_one_plus_epsilon_squared * q_sq_tf)
        bracket_term = t_tf * (bracket_term_first_term - bracket_term_second_term) / q_sq_tf
        prefactor = 4. * tgt_lambda * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) / root_one_plus_epsilon_squared**5
        s_2_plus_plus_V_lp = prefactor * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 2)++ LP: ", s_2_plus_plus_V_lp)

        # Interference: SA(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        bracket_term_first_term = (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * (1. - ((1. - 2. * xb_tf) * t_tf / q_sq_tf)) * tprime_tf / q_sq_tf
        bracket_term_second_term = 4. * ktilde_tf**2 / q_sq_tf
        bracket_term = xb_tf * t_tf * (bracket_term_second_term - bracket_term_first_term) / q_sq_tf
        prefactor = 4. * tgt_lambda * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) / root_one_plus_epsilon_squared**5
        s_2_plus_plus_A_lp = prefactor * bracket_term

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 2)++ LP: ", s_2_plus_plus_A_lp)

        # Interference: S(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * tgt_lambda * k_tf * (2. - y_lep_tf )/ tf.sqrt((1. + ep_tf**2)**5)
        s_2_zero_plus_lp = prefactor * root_combination_of_y_and_epsilon * (1. + (xb_tf * t_tf / q_sq_tf))

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 2)0+ LP: ", s_2_zero_plus_lp)

        # Interference: SV(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * tgt_lambda * k_tf * (2. - y_lep_tf) * t_tf / (tf.sqrt((1. + ep_tf**2)**5) * q_sq_tf)
        s_2_zero_plus_V_lp = prefactor * (1. - xb_tf) * root_combination_of_y_and_epsilon

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 2)0+ LP: ", s_2_zero_plus_V_lp)

        # Interference: SA(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        t_over_Q_squared = t_tf / q_sq_tf
        prefactor = -8. * tf.sqrt(2.) * tgt_lambda  * k_tf * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / tf.sqrt((1. + ep_tf**2)**5)
        s_2_zero_plus_A_lp = prefactor * root_combination_of_y_and_epsilon * (1. + t_over_Q_squared)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 2)0+ LP: ", s_2_zero_plus_A_lp)

        # Interference: S(n = 3)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        prefactor = -4. * tgt_lambda * k_tf * (1. - y_lep_tf - y_lep_tf**2 * ep_tf**2 / 4.) / root_one_plus_epsilon_squared**6
        s_3_plus_plus_lp = prefactor * (one_plus_root_epsilon_stuff - 2. * xb_tf) * ep_tf**2 * tprime_tf / (q_sq_tf * one_plus_root_epsilon_stuff)

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 3)++ LP: ", s_3_plus_plus_lp)

        # Interference: SV(n = 3)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        multiplicative_contribution = t_tf * tprime_tf * (4. * (1. - xb_tf) * xb_tf + ep_tf**2) / q_sq_tf**2
        prefactor = 4. * tgt_lambda * k_tf * (1. - y_lep_tf - y_lep_tf**2 * ep_tf**2 / 4.) / root_one_plus_epsilon_squared**6
        s_3_plus_plus_V_lp = prefactor * multiplicative_contribution

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 3)++ LP: ", s_3_plus_plus_V_lp)

        # Interference: SA(n = 3)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        multiplicative_contribution = xb_tf * t_tf * tprime_tf * (1. + root_one_plus_epsilon_squared - 2. * xb_tf) / q_sq_tf**2
        prefactor = -8. * tgt_lambda * k_tf * (1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / root_one_plus_epsilon_squared**6
        s_3_plus_plus_A_lp = prefactor * multiplicative_contribution

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 3)++ LP: ", s_3_plus_plus_A_lp)

        # Interference: S(n = 3)0+ LP:
        s_3_zero_plus_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: S(n = 3)0+ LP: ", s_3_zero_plus_lp)

        # Interference: SV(n = 3)0+ LP:
        s_3_zero_plus_V_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SV(n = 3)0+ LP: ", s_3_zero_plus_V_lp)

        # Interference: SA(n = 3)0+ LP:
        s_3_zero_plus_A_lp = 0.0

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: SA(n = 3)0+ LP: ", s_3_zero_plus_A_lp)

        # Interference: Re[CurlyC(F)]
        i_curly_c_unp_real = (
            (f1_tf*cff_h_real_tf) - t_tf * f2_tf * cff_e_real_tf / (4.*self.mp**2) +
            xb_tf * (f1_tf+f2_tf)*cff_ht_real_tf / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
            )

        # Interference: Re[CurlyC(F_eff)]
        i_curly_c_unp_feff = (ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf))* (
            (f1_tf*cff_h_real_eff_tf)- t_tf * f2_tf * cff_e_real_eff_tf / (4.*self.mp**2) +
            xb_tf * (f1_tf + f2_tf)*cff_ht_real_eff_tf / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        ))

        # Interference: Im[CurlyC(F)]
        i_curly_c_unp_imag = (
            (f1_tf * cff_h_imag_tf) - t_tf * f2_tf * cff_e_imag_tf / (4.*self.mp**2) +
            xb_tf * (f1_tf + f2_tf) * cff_ht_imag_tf / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyC(F_eff)]
        i_curly_c_unp_imag_feff = (ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * (
            (f1_tf * cff_h_imag_eff_tf) - t_tf * f2_tf * cff_e_imag_eff_tf / (4.*self.mp**2) +
            xb_tf * (f1_tf + f2_tf) * cff_ht_imag_eff_tf / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        ))

        # Interference: Re[CurlyCV(F)]
        i_curly_c_v_unp_real = (
            (cff_h_real_tf + cff_e_real_tf) * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyCV(F_eff)]
        i_curly_c_v_unp_real_feff = (
            ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf))*(cff_h_real_eff_tf + cff_e_real_eff_tf) * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyCV(F)]
        i_curly_c_v_unp_imag = (
            (cff_h_imag_tf + cff_e_imag_tf) * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyCV(F_eff)]
        i_curly_c_v_unp_imag_feff = (
            ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * (cff_h_imag_eff_tf + cff_e_imag_eff_tf) * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyCA(F)]
        i_curly_c_a_unp_real = (
            cff_ht_real_tf * xb_tf * (f1_tf + f2_tf)/ (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyCA(F_eff)]
        i_curly_c_a_unp_real_feff = (
            ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * cff_ht_real_eff_tf * xb_tf * (f1_tf + f2_tf)/ (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyCA(F)]
        i_curly_c_a_unp_imag = (
            cff_ht_imag_tf * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyCA(F_eff)]
        i_curly_c_a_unp_imag_feff = (
            ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * cff_ht_imag_eff_tf * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyC(F)] LP:
        t_over_q_sq_tf = t_tf / q_sq_tf
        ratio_xb_tf = xb_tf / (2. - xb_tf + xb_tf*t_over_q_sq_tf)
        xb_correction_tf = (xb_tf * (1. - t_over_q_sq_tf) / 2.)
        i_curly_c_lp_real = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_real_tf+xb_correction_tf*cff_e_real_tf)+
            (
                1.+(self.mp**2*xb_tf*ratio_xb_tf*(3. + t_over_q_sq_tf)/q_sq_tf)
            )*f1_tf*cff_ht_real_tf-t_over_q_sq_tf*2.*(1.-2.*xb_tf)*ratio_xb_tf*f2_tf*cff_ht_real_tf
            - ratio_xb_tf*(xb_correction_tf*f1_tf+ t_tf*f2_tf/(4.*self.mp**2))*cff_et_real_tf
        )

        # Interference: Re[CurlyC(F_eff)] LP:
        t_over_q_sq_tf = t_tf / q_sq_tf
        ratio_xb_tf = xb_tf / (2. - xb_tf + xb_tf*t_over_q_sq_tf)
        xb_correction_tf = (xb_tf * (1. - t_over_q_sq_tf) / 2.)
        i_curly_c_lp_real_feff = (ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_real_eff_tf+xb_correction_tf*cff_e_real_eff_tf)+
            (
                1.+(self.mp**2*xb_tf*ratio_xb_tf*(3. + t_over_q_sq_tf)/q_sq_tf)
            )*f1_tf*cff_ht_real_eff_tf-t_over_q_sq_tf*2.*(1.-2.*xb_tf)*ratio_xb_tf*f2_tf*cff_ht_real_eff_tf
            - ratio_xb_tf*(xb_correction_tf*f1_tf+ t_tf*f2_tf/(4.*self.mp**2))*cff_et_real_eff_tf
        ))

        # Interference: Im[CurlyC(F)] LP:
        t_over_q_sq_tf = t_tf / q_sq_tf
        ratio_xb_tf = xb_tf / (2. - xb_tf + xb_tf*t_over_q_sq_tf)
        xb_correction_tf = (xb_tf * (1. - t_over_q_sq_tf) / 2.)
        i_curly_c_lp_imag = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_imag_tf+xb_correction_tf*cff_e_imag_tf)+
            (
                1.+(self.mp**2*xb_tf*ratio_xb_tf*(3. + t_over_q_sq_tf)/q_sq_tf)
            )*f1_tf*cff_ht_imag_tf-t_over_q_sq_tf*2.*(1.-2.*xb_tf)*ratio_xb_tf*f2_tf*cff_ht_imag_tf
            - ratio_xb_tf*(xb_correction_tf*f1_tf+ t_tf*f2_tf/(4.*self.mp**2))*cff_et_imag_tf
        )

        # Interference: Im[CurlyC(F_eff)] LP:
        t_over_q_sq_tf = t_tf / q_sq_tf
        ratio_xb_tf = xb_tf / (2. - xb_tf + xb_tf*t_over_q_sq_tf)
        xb_correction_tf = (xb_tf * (1. - t_over_q_sq_tf) / 2.)
        i_curly_c_lp_imag_feff = (ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_imag_eff_tf+xb_correction_tf*cff_e_imag_eff_tf)+
            (
                1.+(self.mp**2*xb_tf*ratio_xb_tf*(3. + t_over_q_sq_tf)/q_sq_tf)
            )*f1_tf*cff_ht_imag_eff_tf-t_over_q_sq_tf*2.*(1.-2.*xb_tf)*ratio_xb_tf*f2_tf*cff_ht_imag_eff_tf
            - ratio_xb_tf*(xb_correction_tf*f1_tf+ t_tf*f2_tf/(4.*self.mp**2))*cff_et_imag_eff_tf
        ))

        # Interference: Re[CurlyCV(F)] LP
        i_curly_c_v_lp_real = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_real_tf+ xb_correction_tf*cff_e_real_tf)
        )

        # Interference: Re[CurlyCV(F_eff)] LP
        i_curly_c_v_lp_real_feff = (
            ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_real_eff_tf+ xb_correction_tf*cff_e_real_eff_tf)
        )

        # Interference: Im[CurlyCV(F)] LP
        i_curly_c_v_lp_imag = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_imag_tf + xb_correction_tf*cff_e_imag_tf)
        )

        # Interference: Im[CurlyCV(F_eff)] LP
        i_curly_c_v_lp_imag_feff = (
            ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_imag_eff_tf+ xb_correction_tf*cff_e_imag_eff_tf)
        )

        # Interference Re[CurlyCA(F)] LP
        i_curly_c_a_lp_real = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_ht_real_tf*(1. + 2.*xb_tf*self.mp**2/q_sq_tf)+xb_tf*cff_et_real_tf/2.)
        )

        # Interference Re[CurlyCA(F_eff)] LP
        i_curly_c_a_lp_real_feff = (
            ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * ratio_xb_tf*(f1_tf + f2_tf)*(cff_ht_real_eff_tf*(1. + 2.*xb_tf*self.mp**2/q_sq_tf)+xb_tf*cff_et_real_eff_tf/2.)
        )

        # Interference Im[CurlyCA(F)] LP
        i_curly_c_a_lp_imag = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_ht_imag_tf*(1. + 2.*xb_tf*self.mp**2/q_sq_tf)+xb_tf*cff_et_imag_tf/2.)
        )

        # Interference Im[CurlyCA(F_eff)] LP
        i_curly_c_a_lp_imag_feff = (
            ktilde_tf * tf.sqrt(2.) / ((2. - xb_tf) * tf.sqrt(q_sq_tf)) * ratio_xb_tf*(f1_tf + f2_tf)*(cff_ht_imag_eff_tf*(1. + 2.*xb_tf*self.mp**2/q_sq_tf)+xb_tf*cff_et_imag_eff_tf/2.)
        )

        # Interference: c_{0}:
        c0_unp = (
            c_0_plus_plus_unp * i_curly_c_unp_real +
            c_0_plus_plus_V_unp * i_curly_c_v_unp_real +
            c_0_plus_plus_A_unp * i_curly_c_a_unp_real +
            c_0_zero_plus_unp * i_curly_c_unp_feff +
            c_0_zero_plus_V_unp * i_curly_c_v_unp_real_feff +
            c_0_zero_plus_A_unp * i_curly_c_a_unp_real_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: c0 unp: ", c0_unp)

        c0_lp = (
            c_0_plus_plus_lp * i_curly_c_lp_real +
            c_0_plus_plus_V_lp * i_curly_c_v_lp_real +
            c_0_plus_plus_A_lp * i_curly_c_a_lp_real +
            c_0_zero_plus_lp * i_curly_c_lp_real_feff +
            c_0_zero_plus_V_lp * i_curly_c_v_lp_real_feff +
            c_0_zero_plus_A_lp * i_curly_c_a_lp_real_feff
            )


        self.debug_print("[DEBUG]: c0 LP: ", c0_lp)

        # Interference: c_{1}:
        c1_unp = (
            c_1_plus_plus_unp * i_curly_c_unp_real +
            c_1_plus_plus_V_unp * i_curly_c_v_unp_real +
            c_1_plus_plus_A_unp * i_curly_c_a_unp_real +
            c_1_zero_plus_unp * i_curly_c_unp_feff +
            c_1_zero_plus_V_unp * i_curly_c_v_unp_real_feff +
            c_1_zero_plus_A_unp * i_curly_c_a_unp_real_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: c1 unp: ", c1_unp)

        c1_lp = (
            c_1_plus_plus_lp * i_curly_c_lp_real +
            c_1_plus_plus_V_lp * i_curly_c_v_lp_real +
            c_1_plus_plus_A_lp * i_curly_c_a_lp_real +
            c_1_zero_plus_lp * i_curly_c_lp_real_feff +
            c_1_zero_plus_V_lp * i_curly_c_v_lp_real_feff +
            c_1_zero_plus_A_lp * i_curly_c_a_lp_real_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: c1 LP: ", c1_lp)

        # Interference: c_{2}:
        c2_unp = (
            c_2_plus_plus_unp * i_curly_c_unp_real +
            c_2_plus_plus_V_unp * i_curly_c_v_unp_real +
            c_2_plus_plus_A_unp * i_curly_c_a_unp_real +
            c_2_zero_plus_unp * i_curly_c_unp_feff +
            c_2_zero_plus_V_unp * i_curly_c_v_unp_real_feff +
            c_2_zero_plus_A_unp * i_curly_c_a_unp_real_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: c2 unp: ", c2_unp)

        c2_lp = (
            c_2_plus_plus_lp * i_curly_c_lp_real +
            c_2_plus_plus_V_lp * i_curly_c_v_lp_real +
            c_2_plus_plus_A_lp * i_curly_c_a_lp_real +
            c_2_zero_plus_lp * i_curly_c_lp_real_feff +
            c_2_zero_plus_V_lp * i_curly_c_v_lp_real_feff +
            c_2_zero_plus_A_lp * i_curly_c_a_lp_real_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: c2 LP: ", c2_lp)

        # Interference: c_{3}:
        c3_unp = (
            c_3_plus_plus_unp * i_curly_c_unp_real +
            c_3_plus_plus_V_unp * i_curly_c_v_unp_real +
            c_3_plus_plus_A_unp * i_curly_c_a_unp_real +
            c_3_zero_plus_unp * i_curly_c_unp_feff +
            c_3_zero_plus_V_unp * i_curly_c_v_unp_real_feff +
            c_3_zero_plus_A_unp * i_curly_c_a_unp_real_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: c3 unp: ", c3_unp)

        c3_lp = (
            c_3_plus_plus_lp * i_curly_c_lp_real +
            c_3_plus_plus_V_lp * i_curly_c_v_lp_real +
            c_3_plus_plus_A_lp * i_curly_c_a_lp_real +
            c_3_zero_plus_lp * i_curly_c_lp_real_feff +
            c_3_zero_plus_V_lp * i_curly_c_v_lp_real_feff +
            c_3_zero_plus_A_lp * i_curly_c_a_lp_real_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: c3 LP: ", c3_lp)

        # Interference: s_{1}:
        s1_unp = (
            s_1_plus_plus_unp * i_curly_c_unp_imag +
            s_1_plus_plus_V_unp * i_curly_c_v_unp_imag +
            s_1_plus_plus_A_unp * i_curly_c_a_unp_imag+
            s_1_zero_plus_unp * i_curly_c_unp_imag_feff +
            s_1_zero_plus_V_unp * i_curly_c_v_unp_imag_feff +
            s_1_zero_plus_A_unp * i_curly_c_a_unp_imag_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: s1 unp: ", s1_unp)

        s1_lp = (
            s_1_plus_plus_lp * i_curly_c_lp_imag +
            s_1_plus_plus_V_lp * i_curly_c_v_lp_imag +
            s_1_plus_plus_A_lp * i_curly_c_a_lp_imag+
            s_1_zero_plus_lp * i_curly_c_lp_imag_feff +
            s_1_zero_plus_V_lp * i_curly_c_v_lp_imag_feff +
            s_1_zero_plus_A_lp * i_curly_c_a_lp_imag_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: s1 LP: ", s1_lp)

        # Interference: s_{2}:
        s2_unp = (
            s_2_plus_plus_unp * i_curly_c_unp_imag +
            s_2_plus_plus_V_unp * i_curly_c_v_unp_imag +
            s_2_plus_plus_A_unp * i_curly_c_a_unp_imag +
            s_2_zero_plus_unp * i_curly_c_unp_imag_feff +
            s_2_zero_plus_V_unp * i_curly_c_v_unp_imag_feff +
            s_2_zero_plus_A_unp * i_curly_c_a_unp_imag_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: s2 unp: ", s2_unp)

        s2_lp = (
            s_2_plus_plus_lp * i_curly_c_lp_imag +
            s_2_plus_plus_V_lp * i_curly_c_v_lp_imag +
            s_2_plus_plus_A_lp * i_curly_c_a_lp_imag+
            s_2_zero_plus_lp * i_curly_c_lp_imag_feff +
            s_2_zero_plus_V_lp * i_curly_c_v_lp_imag_feff +
            s_2_zero_plus_A_lp * i_curly_c_a_lp_imag_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: s2 LP: ", s2_lp)

        # Interference: s_{3}:
        s3_unp = (
            s_3_plus_plus_unp * i_curly_c_unp_imag +
            s_3_plus_plus_V_unp * i_curly_c_v_unp_imag +
            s_3_plus_plus_A_unp * i_curly_c_a_unp_imag +
            s_3_zero_plus_unp * i_curly_c_unp_imag_feff +
            s_3_zero_plus_V_unp * i_curly_c_v_unp_imag_feff +
            s_3_zero_plus_A_unp * i_curly_c_a_unp_imag_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: s3 unp: ", s3_unp)

        s3_lp = (
            s_3_plus_plus_lp * i_curly_c_lp_imag +
            s_3_plus_plus_V_lp * i_curly_c_v_lp_imag +
            s_3_plus_plus_A_lp * i_curly_c_a_lp_imag+
            s_3_zero_plus_lp * i_curly_c_lp_imag_feff +
            s_3_zero_plus_V_lp * i_curly_c_v_lp_imag_feff +
            s_3_zero_plus_A_lp * i_curly_c_a_lp_imag_feff
            )

        # correct, 2026/08/31
        self.debug_print("[DEBUG]: s3 LP: ", s3_lp)

        # Trigomometry:
        cosine_0 = tf.cos(0. * (tf.constant(np.pi) - phi_tf))
        cosine_1 = tf.cos(1. * (tf.constant(np.pi) - phi_tf))
        cosine_2 = tf.cos(2. * (tf.constant(np.pi) - phi_tf))
        cosine_3 = tf.cos(3. * (tf.constant(np.pi) - phi_tf))
        sine_1 = tf.sin(1. * (tf.constant(np.pi) - phi_tf))
        sine_2 = tf.sin(2. * (tf.constant(np.pi) - phi_tf))
        sine_3 = tf.sin(2. * (tf.constant(np.pi) - phi_tf))

        bh_squared = (
            (
                (bh_c0 + bh_lp_c0) * cosine_0 +
                (bh_c1 + bh_lp_c1) * cosine_1 +
                (bh_c2 + bh_lp_c2) * cosine_2
            ) / 
            (xb_tf * xb_tf * y_lep_tf * y_lep_tf * (1. + ep_tf**2)**2 * t_tf * p1_tf * p2_tf)
            )

        dvcs_squared = (
            (
                (dvcs_unp_c0 + dvcs_lp_c0) * cosine_0 +
                (dvcs_unp_c1 + dvcs_lp_c1) * cosine_1 +
                (dvcs_unp_s1 + dvcs_lp_s1) * sine_1
            ) / (y_lep_tf * y_lep_tf * q_sq_tf)
        )

        interference = (
            (
                (c0_unp + c0_lp) * cosine_0 +
                (c1_unp + c1_lp) * cosine_1 +
                (c2_unp + c2_lp) * cosine_2 +
                (c3_unp + c3_lp) * cosine_3 +
                (s1_unp + s1_lp) * sine_1 +
                (s2_unp + s2_lp) * sine_2 +
                (s3_unp + s3_lp) * sine_3
            ) /(xb_tf * y_lep_tf * y_lep_tf * y_lep_tf * t_tf * p1_tf * p2_tf)
        )

        cross_section = (self.gev6_to_gev4_per_nb*self.qed_alpha**3*xb_tf*y_lep_tf*y_lep_tf*(
            bh_squared +
            dvcs_squared +
            interference
            ) / (8.*tf.constant(np.pi)*q_sq_tf*q_sq_tf*tf.sqrt(1. + ep_tf**2)))

        return cross_section

    def call(self, true_values, predicted_values):
    
        # the CFFs:
        cff_h_real_tf = predicted_values[:, 0]
        cff_h_imag_tf = predicted_values[:, 1]
        cff_ht_real_tf = predicted_values[:, 2]
        cff_ht_imag_tf = predicted_values[:, 3]
        cff_e_real_tf = predicted_values[:, 4]
        cff_e_imag_tf = predicted_values[:, 5]
        cff_et_real_tf = predicted_values[:, 6]
        cff_et_imag_tf = predicted_values[:, 7]

        # kinematics:
        t_tf = predicted_values[:, 8]
        xb_tf = predicted_values[:, 9]
        q_sq_tf = predicted_values[:, 10]
        phi_tf = predicted_values[:, 11]

        # derived quantities -> form factors:
        fe_tf = predicted_values[:, 12]
        fg_tf = predicted_values[:, 13]
        f1_tf = predicted_values[:, 14]
        f2_tf = predicted_values[:, 15]

        # derived quantities -> kinematics
        ep_tf = predicted_values[:, 16]
        y_lep_tf = predicted_values[:, 17]
        xi_tf = predicted_values[:, 18]
        tprime_tf = predicted_values[:, 19]
        ktilde_tf = predicted_values[:, 20]
        k_tf = predicted_values[:, 21]

        # derived quantities -> phi-depdendent stuff:
        kdd_tf = predicted_values[:, 22]
        p1_tf = predicted_values[:, 23]
        p2_tf = predicted_values[:, 24]

        # polarizations:
        lep_lambda = predicted_values[:, 25]
        tgt_lambda = predicted_values[:, 26]

        # the effective CFFs (using WW!):
        cff_h_real_eff_tf = 2. * cff_h_real_tf / (1. + xi_tf)
        cff_h_imag_eff_tf = 2. * cff_h_imag_tf / (1. + xi_tf)
        cff_ht_real_eff_tf = 2. * cff_ht_real_tf / (1. + xi_tf)
        cff_ht_imag_eff_tf = 2. * cff_ht_imag_tf / (1. + xi_tf)
        cff_e_real_eff_tf = 2. * cff_e_real_tf / (1. + xi_tf)
        cff_e_imag_eff_tf = 2. * cff_e_imag_tf / (1. + xi_tf)
        cff_et_real_eff_tf = 2. * cff_et_real_tf / (1. + xi_tf)
        cff_et_imag_eff_tf = 2. * cff_et_imag_tf / (1. + xi_tf)

        self.debug_print("[DEBUG]: effective Re[H]: ", cff_h_real_eff_tf)
        self.debug_print("[DEBUG]: effective Im[H]: ", cff_h_imag_eff_tf)
        self.debug_print("[DEBUG]: effective Re[Ht]: ", cff_ht_real_eff_tf)
        self.debug_print("[DEBUG]: effective Im[Ht]: ", cff_ht_imag_eff_tf)
        self.debug_print("[DEBUG]: effective Re[E]: ", cff_e_real_eff_tf)
        self.debug_print("[DEBUG]: effective Im[E]: ", cff_e_imag_eff_tf)
        self.debug_print("[DEBUG]: effective Re[Et]: ", cff_et_real_eff_tf)
        self.debug_print("[DEBUG]: effective Im[Et]: ", cff_et_imag_eff_tf)

        # # observables:
        # true_cross_section = true_values[:, 0]
        # true_bsa = true_values[:, 1]

        # observables:
        sigma_plus = self.compute_cross_section(
            q_sq_tf, xb_tf, t_tf, ep_tf, y_lep_tf, xi_tf, k_tf, f1_tf, f2_tf, ktilde_tf, tprime_tf, phi_tf, p1_tf, p2_tf,
            cff_h_real_tf, cff_ht_real_tf, cff_e_real_tf, cff_et_real_tf,
            cff_h_imag_tf, cff_ht_imag_tf, cff_e_imag_tf, cff_et_imag_tf,
            cff_h_real_eff_tf, cff_ht_real_eff_tf, cff_e_real_eff_tf, cff_et_real_eff_tf,
            cff_h_imag_eff_tf, cff_ht_imag_eff_tf, cff_e_imag_eff_tf, cff_et_imag_eff_tf,
            lep_lambda = tf.ones_like(lep_lambda),
            tgt_lambda = tgt_lambda
        )
        sigma_minus = self.compute_cross_section(
            q_sq_tf, xb_tf, t_tf, ep_tf, y_lep_tf, xi_tf, k_tf, f1_tf, f2_tf, ktilde_tf, tprime_tf, phi_tf, p1_tf, p2_tf,
            cff_h_real_tf, cff_ht_real_tf, cff_e_real_tf, cff_et_real_tf,
            cff_h_imag_tf, cff_ht_imag_tf, cff_e_imag_tf, cff_et_imag_tf,
            cff_h_real_eff_tf, cff_ht_real_eff_tf, cff_e_real_eff_tf, cff_et_real_eff_tf,
            cff_h_imag_eff_tf, cff_ht_imag_eff_tf, cff_e_imag_eff_tf, cff_et_imag_eff_tf,
            lep_lambda = -tf.ones_like(lep_lambda),
            tgt_lambda = tgt_lambda,
        )

        sigma_even = 0.5 * (sigma_plus + sigma_minus)
        sigma_odd = 0.5 * (sigma_plus - sigma_minus)

        # loss-computed observables:
        predicted_unp_beam_unp_cross_section = sigma_even + lep_lambda * sigma_odd
        predicted_plus_beam_unp_cross_section = sigma_plus
        predicted_minus_beam_unp_cross_section = sigma_minus
        predicted_unp_target_bsa = sigma_odd / sigma_even

        if self.debugging:
            self.debug_sigma_plus = sigma_plus
            self.debug_sigma_minus = sigma_minus
            self.debug_sigma_even = sigma_even
            self.debug_sigma_odd = sigma_odd
            self.debug_cross_section = predicted_unp_beam_unp_cross_section
            self.debug_bsa = predicted_unp_target_bsa

        predicted_observables = {
            "unp_beam_unp_target_xsec":
                predicted_unp_beam_unp_cross_section,
            "plus_beam_unp_target_xsec":
                predicted_plus_beam_unp_cross_section,
            "minus_beam_unp_target_xsec":
                predicted_minus_beam_unp_cross_section,
            "unp_target_bsa":
                predicted_unp_target_bsa,
        }
        
        predicted = tf.stack(
            [predicted_observables[name] for name in self.enabled_observables],
            axis = 1,
        )

        # compute residuals:
        residuals = true_values - predicted

        # compute the MSE:
        mean_squared_error = tf.reduce_mean(
            self.observable_weights * tf.square(residuals)
        )

        return mean_squared_error

def cff_h_model():

    kinematics_inputs = tf.keras.Input(shape = (3,), name = "input_values")
    physics_input = tf.keras.Input(shape = (19,), name = "precomputed_physics")
    dnn_kinematic_inputs = tf.keras.layers.Lambda(lambda x: x[:, :3], name = "input_kinematics")(kinematics_inputs)
    hidden = tf.keras.layers.Dense(32, kernel_initializer = "he_normal", activation = "relu")(dnn_kinematic_inputs)
    hidden = tf.keras.layers.Dense(64, kernel_initializer = "he_normal", activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(32, kernel_initializer = "he_normal", activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(16, kernel_initializer = "he_normal", activation = "relu")(hidden)
    # Re[H], Im[H], Re[Ht], Im[Ht], Re[E], Im[E], Re[Et], Im[Et]
    cff_outputs = tf.keras.layers.Dense(8, activation = "linear", name = "cff_h_htilde")(hidden)
    full_model_outputs = tf.keras.layers.Concatenate(name = "physics_and_cffs")([cff_outputs, physics_input])
    model = tf.keras.Model(inputs = [kinematics_inputs, physics_input], outputs = full_model_outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = LEARNING_RATE
            ),
        loss = UnfoldedSimultaneousFitLoss(
            enabled_observables = enabled_observables,
            observable_weights = observable_weights
        ),
        jit_compile = True,
        )
    
    # return model:
    return model

for replica in range(NUMBER_OF_REPLICAS):

    tf.keras.backend.clear_session()
    gc.collect()

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
    print(f"The model ran for {number_of_epochs_run} epochs before early stopping.")

    start_dnn_save_time = time.perf_counter()
    dnn_model.save(f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras")
    print(f"[TIMING]: DNN saving time: {time.perf_counter() - start_dnn_save_time:.2f} s")

    training_loss_data = dnn_model_history.history["loss"]
    validation_loss_data = dnn_model_history.history["val_loss"]

    start_dnn_evaluation_time = time.perf_counter()
    dnn_evaluation_statistics = dnn_model.evaluate(
        x = {
            "input_values": x_testing,
            "precomputed_physics": precomputed_physics_testing,
        },
        y = y_testing,
        verbose = 0)
    print(f"[TIMING]: DNN evaluation time: {time.perf_counter() - start_dnn_evaluation_time:.2f} s")
    print(f"[INFO]: Test Loss for Replica {replica_number}: {dnn_evaluation_statistics}")

    pd.DataFrame({
        'testing_loss': [dnn_evaluation_statistics],
    }).to_csv(
        f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_loss_data.csv", 
        index = False)

    # save npz with DNN training information:
    np.savez(
        file = f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_losses_vs_epochs.npz",
        training_loss = training_loss_data,
        validation_loss = validation_loss_data
        )

    # cleanup
    del dnn_model

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

    curves_fig.savefig(f"./local/version_{MAJOR_MINOR_NUMBER}/learning_curves/lc_replica_{replica_index}_v{MAJOR_MINOR_NUMBER}.png")
    curves_fig.savefig(f"./local/version_{MAJOR_MINOR_NUMBER}/learning_curves/lc_replica_{replica_index}_v{MAJOR_MINOR_NUMBER}.eps")

    log_curves_fig.savefig(f"./local/version_{MAJOR_MINOR_NUMBER}/learning_curves/log_lc_replica_{replica_index}_v{MAJOR_MINOR_NUMBER}.png")
    log_curves_fig.savefig(f"./local/version_{MAJOR_MINOR_NUMBER}/learning_curves/log_lc_replica_{replica_index}_v{MAJOR_MINOR_NUMBER}.eps")

    plt.close(curves_fig)
    plt.close(log_curves_fig)

    del curves_fig
    del log_curves_fig

range_of_t = np.linspace(t.min(), t.max())
range_of_x_b = np.linspace(xb.min(), xb.max())
range_of_q_squared = np.linspace(q2.min(), q2.max())

replica_paths = sorted(glob.glob(f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_*_v{MAJOR_MINOR_NUMBER}.keras"))
replicas = [tf.keras.models.load_model(
    path,
    compile = False,
    safe_mode = False) for path in replica_paths]
print(f"[INFO]: Loaded {len(replicas)} replica models.")

all_predictions = []

for replica in replicas:
    
    predicted_outputs = replica.predict(
        {
            "input_values": dnn_inputs,
            "precomputed_physics": physics_data,
        },
        verbose = 0,
    ) # predicting using x_data
    all_predictions.append(predicted_outputs)

all_predictions = np.array(all_predictions)

replicas_cross_predictions = []
replicas_bsa_predictions = []

for index, _ in enumerate(all_predictions):

    prediction = all_predictions[index]

    cff_h_real = prediction[:, 0]
    cff_h_imag = prediction[:, 1]
    cff_ht_real = prediction[:, 2]
    cff_ht_imag = prediction[:, 3]

    t = prediction[:, 4]
    xb = prediction[:, 5]
    q_squared = prediction[:, 6]
    phi = prediction[:, 7]

    fe = prediction[:, 8]
    fg = prediction[:, 9]
    f1 = prediction[:,10]
    f2 = prediction[:,11]

    epsilon = prediction[:,12]
    y_lep = prediction[:,13]
    xi = prediction[:,14]
    tprime = prediction[:,15]
    ktilde = prediction[:,16]
    k = prediction[:,17]

    kdd = prediction[:,18]
    p1 = prediction[:,19]
    p2 = prediction[:,20]

    cross_section = bkm10_cross_section(
        0.0, 0.0,
        q_squared, xb, t, epsilon, y_lep, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_h_real, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)

    predicted_bsa = bkm10_bsa(
        0.0,
        q_squared, xb, t, epsilon, y_lep, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_h_real, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)

    replicas_cross_predictions.append(cross_section)
    replicas_bsa_predictions.append(predicted_bsa)

replicas_cross_predictions = np.array(replicas_cross_predictions)
replicas_bsa_predictions = np.array(replicas_bsa_predictions)

mean_xs = np.mean(replicas_cross_predictions, axis = 0)
std_dev_xs = np.std(replicas_cross_predictions, axis = 0)

xs_mean = np.mean(replicas_cross_predictions, axis = 0)
xs_min = np.min(replicas_cross_predictions, axis = 0)
xs_max = np.max(replicas_cross_predictions, axis = 0)
xs_q1 = np.percentile(replicas_cross_predictions, 25, axis = 0)
xs_q3 = np.percentile(replicas_cross_predictions, 75, axis = 0)

xs_percentile_10 = np.percentile(replicas_cross_predictions, 10, axis = 0)
xs_percentile_20 = np.percentile(replicas_cross_predictions, 20, axis = 0)
xs_percentile_30 = np.percentile(replicas_cross_predictions, 30, axis = 0)
xs_percentile_40 = np.percentile(replicas_cross_predictions, 40, axis = 0)
xs_median = np.percentile(replicas_cross_predictions, 50, axis = 0)
xs_percentile_60 = np.percentile(replicas_cross_predictions, 60, axis = 0)
xs_percentile_70 = np.percentile(replicas_cross_predictions, 70, axis = 0)
xs_percentile_80 = np.percentile(replicas_cross_predictions, 80, axis = 0)
xs_percentile_90 = np.percentile(replicas_cross_predictions, 90, axis = 0)

mean_bsa = np.mean(replicas_bsa_predictions, axis = 0)
std_dev_bsa = np.std(replicas_bsa_predictions, axis = 0)

bsa_mean = np.mean(replicas_bsa_predictions, axis = 0)
bsa_min = np.min(replicas_bsa_predictions, axis = 0)
bsa_max = np.max(replicas_bsa_predictions, axis = 0)
bsa_q1 = np.percentile(replicas_bsa_predictions, 25, axis = 0)
bsa_q3 = np.percentile(replicas_bsa_predictions, 75, axis = 0)

bsa_percentile_10 = np.percentile(replicas_bsa_predictions, 10, axis = 0)
bsa_percentile_20 = np.percentile(replicas_bsa_predictions, 20, axis = 0)
bsa_percentile_30 = np.percentile(replicas_bsa_predictions, 30, axis = 0)
bsa_percentile_40 = np.percentile(replicas_bsa_predictions, 40, axis = 0)
bsa_median = np.percentile(replicas_bsa_predictions, 50, axis = 0)
bsa_percentile_60 = np.percentile(replicas_bsa_predictions, 60, axis = 0)
bsa_percentile_70 = np.percentile(replicas_bsa_predictions, 70, axis = 0)
bsa_percentile_80 = np.percentile(replicas_bsa_predictions, 80, axis = 0)
bsa_percentile_90 = np.percentile(replicas_bsa_predictions, 90, axis = 0)

fig2, ax2 = plt.subplots(1, figsize = (10, 7))

ax2.scatter(
    phi_array_in_radians, bkm10_unp_beam_unp_target_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

ax2.plot(
    phi_array_in_radians,
    mean_xs,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_max,
    y2 = xs_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_90,
    y2 = xs_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_80,
    y2 = xs_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_70,
    y2 = xs_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_60,
    y2 = xs_percentile_40,
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
    phi_array_in_radians, bkm10_bsa_unp_target_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

ax3.plot(
    phi_array_in_radians,
    mean_bsa,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_max,
    y2 = bsa_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_90,
    y2 = bsa_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_80,
    y2 = bsa_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_70,
    y2 = bsa_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_60,
    y2 = bsa_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

ax3.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
ax3.set_ylabel(r"BSA", fontsize = 16)
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

post_cff_fit_xsec_figure, post_cff_fit_xsec_axis = plt.subplots(nrows = 2, ncols = 1,
    gridspec_kw = {'height_ratios': [3, 1]},
    figsize = (9, 9))

post_cff_fit_xsec_axis[1].plot(mean_xs - bkm10_unp_beam_unp_target_km15, color = 'gray', label = "Residuals")

post_cff_fit_xsec_axis[0].scatter(
    phi_array_in_radians, bkm10_unp_beam_unp_target_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

post_cff_fit_xsec_axis[0].plot(
    phi_array_in_radians,
    mean_xs,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

post_cff_fit_xsec_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = xs_max,
    y2 = xs_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

post_cff_fit_xsec_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_90,
    y2 = xs_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

post_cff_fit_xsec_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_80,
    y2 = xs_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

post_cff_fit_xsec_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_70,
    y2 = xs_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

post_cff_fit_xsec_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_60,
    y2 = xs_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

post_cff_fit_xsec_axis[0].set_xlabel(r"$\phi$ [radians]", fontsize = 16)
post_cff_fit_xsec_axis[0].set_ylabel(r"$d^{4}\sigma$ [nb / GeV$^{4}$]", fontsize = 16)
post_cff_fit_xsec_axis[1].set_ylabel(r"Residuals", fontsize = 16)
post_cff_fit_xsec_axis[0].set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}")

post_cff_fit_xsec_axis[0].legend()

post_cff_fit_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cross_section_comparison_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    post_cff_fit_xsec_figure.savefig(
        f"{post_cff_fit_plotname}.{extension}",
        facecolor = 'white')

plt.close(post_cff_fit_xsec_figure)

post_cff_fit_bsa_figure, post_cff_fit_bsa_axis = plt.subplots(
    nrows = 2, ncols = 1,  gridspec_kw = {'height_ratios': [3, 1]}, figsize = (7, 7))

post_cff_fit_bsa_axis[1].plot(mean_bsa - bkm10_bsa_unp_target_km15, color = 'gray', label = "Residuals")

post_cff_fit_bsa_axis[0].scatter(
    phi_array_in_radians, bkm10_bsa_unp_target_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

post_cff_fit_bsa_axis[0].plot(
    phi_array_in_radians,
    mean_bsa,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

post_cff_fit_bsa_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = bsa_max,
    y2 = bsa_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

post_cff_fit_bsa_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_90,
    y2 = bsa_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

post_cff_fit_bsa_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_80,
    y2 = bsa_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

post_cff_fit_bsa_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_70,
    y2 = bsa_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

post_cff_fit_bsa_axis[0].fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_60,
    y2 = bsa_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

post_cff_fit_bsa_axis[0].set_xlabel(r"$\phi$ [radians]", fontsize = 16)
post_cff_fit_bsa_axis[0].set_ylabel(r"BSA", fontsize = 16)
post_cff_fit_bsa_axis[1].set_ylabel(r"Residuals", fontsize = 16)
post_cff_fit_bsa_axis[0].set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}")

post_cff_fit_bsa_axis[0].legend()

post_cff_fit_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/bsa_comparison_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    post_cff_fit_bsa_figure.savefig(
        f"{post_cff_fit_plotname}.{extension}",
        facecolor = 'white')

plt.close(post_cff_fit_bsa_figure)

cff_h_real_pred_per_replica = np.mean(all_predictions[:, :, 0], axis = 1)
cff_h_imag_pred_per_replica = np.mean(all_predictions[:, :, 1], axis = 1)
cff_ht_real_pred_per_replica = np.mean(all_predictions[:, :, 2], axis = 1)
cff_ht_imag_pred_per_replica = np.mean(all_predictions[:, :, 3], axis = 1)

cff_h_real_mean, cff_h_real_stddev = norm.fit(cff_h_real_pred_per_replica)
cff_h_imag_mean, cff_h_imag_stddev = norm.fit(cff_h_imag_pred_per_replica)
cff_ht_real_mean, cff_ht_real_stddev = norm.fit(cff_ht_real_pred_per_replica)
cff_ht_imag_mean, cff_ht_imag_stddev = norm.fit(cff_ht_imag_pred_per_replica)

print(f"[INFO]: Re[H] mean of {cff_h_real_mean} and stdddev of {cff_h_real_stddev}")
print(f"[INFO]: Im[H] mean of {cff_h_imag_mean} and stdddev of {cff_h_imag_stddev}")
print(f"[INFO]: Re[Ht] mean of {cff_ht_real_mean} and stdddev of {cff_ht_real_stddev}")
print(f"[INFO]: Im[Ht] mean of {cff_ht_imag_mean} and stdddev of {cff_ht_imag_stddev}")

burner_x_values_cff_h_real = np.linspace(
    cff_h_real_mean - 4.*cff_h_real_stddev,
    cff_h_real_mean + 4.*cff_h_real_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (10, 7))

ax4.hist(cff_h_real_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_h_real, norm.pdf(burner_x_values_cff_h_real, cff_h_real_mean, cff_h_real_stddev),
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_h_real_mean:.3f}$, $\sigma = {cff_h_real_stddev:.3f}$")
ax4.axvline(real_h_values[0], color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {real_h_values[0]:.3f}")

ax4.set_ylabel("Frequency", rotation = 90., fontsize = 16.)
ax4.set_xlabel(r"Re$[\mathcal{H}]$", fontsize = 16.)
ax4.set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}", fontsize = 16.)

ax4.legend()

cff_h_real_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cff_h_real_fits_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig4.savefig(
        f"{cff_h_real_plotname}.{extension}",
        facecolor = 'white')

plt.close(fig4)

burner_x_values_cff_h_imag = np.linspace(
    cff_h_imag_mean - 4.*cff_h_imag_stddev,
    cff_h_imag_mean + 4.*cff_h_imag_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (10, 7))

ax4.hist(cff_h_imag_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_h_imag, norm.pdf(burner_x_values_cff_h_imag, cff_h_imag_mean, cff_h_imag_stddev),
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_h_imag_mean:.3f}$, $\sigma = {cff_h_imag_stddev:.3f}$")
ax4.axvline(imag_h_values[0], color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {imag_h_values[0]:.3f}")

ax4.set_ylabel("Frequency", rotation = 90., fontsize = 16.)
ax4.set_xlabel(r"Im$[\mathcal{H}]$", fontsize = 16.)
ax4.set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}", fontsize = 16.)

ax4.legend()

cff_h_imag_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cff_h_imag_fits_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig4.savefig(
        f"{cff_h_imag_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig4)

burner_x_values_cff_ht_real = np.linspace(
    cff_ht_real_mean - 4.*cff_ht_real_stddev,
    cff_ht_real_mean + 4.*cff_ht_real_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (10, 7))

ax4.hist(cff_ht_real_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_ht_real, norm.pdf(burner_x_values_cff_ht_real, cff_ht_real_mean, cff_ht_real_stddev),
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_ht_real_mean:.3f}$, $\sigma = {cff_ht_real_stddev:.3f}$")
ax4.axvline(real_ht_values[0], color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {real_ht_values[0]:.3f}")

ax4.set_ylabel("Frequency", rotation = 90., fontsize = 16.)
ax4.set_xlabel(r"Real$[\tilde{\mathcal{H}}]$", fontsize = 16.)
ax4.set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}", fontsize = 16.)

ax4.legend()

cff_ht_real_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cff_ht_real_fits_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig4.savefig(
        f"{cff_ht_real_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig4)

burner_x_values_cff_ht_imag = np.linspace(
    cff_ht_imag_mean - 4.*cff_ht_imag_stddev,
    cff_ht_imag_mean + 4.*cff_ht_imag_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (10, 7))

ax4.hist(cff_ht_imag_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_ht_imag, norm.pdf(burner_x_values_cff_ht_imag, cff_ht_imag_mean, cff_ht_imag_stddev),
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_ht_imag_mean:.3f}$, $\sigma = {cff_ht_imag_stddev:.3f}$")
ax4.axvline(imag_ht_values[0], color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {imag_ht_values[0]:.3f}")

ax4.set_ylabel("Frequency", rotation = 90., fontsize = 16.)
ax4.set_xlabel(r"Im$[\tilde{\mathcal{H}}]$", fontsize = 16.)
ax4.set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}", fontsize = 16.)

ax4.legend()

cff_ht_imag_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cff_ht_imag_fits_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig4.savefig(
        f"{cff_ht_imag_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig4)