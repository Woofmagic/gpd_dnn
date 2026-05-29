#################################################################################
# FILE INFORMATION:
# Purpose: move individual "set" data to their respective kinematic set folder.
# Created: 20260504
# Last changed: 20260528
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import os 

import pandas as pd
import numpy as np

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
# Required functions for numerical computations:
#################################################################################

_MASS_OF_PROTON_IN_GEV = 0.93827208816
_ELECTRIC_FORM_FACTOR_CONSTANT = 0.710649
_PROTON_MAGNETIC_MOMENT = 2.79284734463

def compute_fe(t):
    return 1./ (1. - (t / _ELECTRIC_FORM_FACTOR_CONSTANT))**2

def compute_fg(fe):
    return _PROTON_MAGNETIC_MOMENT * fe

def compute_f2(t, fe, fg):
    tau = -1. * t/ (4. * _MASS_OF_PROTON_IN_GEV**2)
    numerator = fg - fe
    denominator = 1. + tau
    return numerator / denominator

def compute_f1(fg, f2):
    return fg - f2

def compute_epsilon(xb, q_squared):
    return 2. * xb * _MASS_OF_PROTON_IN_GEV / np.sqrt(q_squared)

def compute_y(k_beam, q_squared, ep):
    return np.sqrt(q_squared) / (ep * k_beam)

def compute_skewness(xb, t, q_squared):
    return xb * (1. + (t / (2. * q_squared))) / (2. - xb + (xb * t / q_squared))

def compute_t_min(xb, q_squared, ep):
    return -1. * q_squared * ((2. * (1. - xb) * (1. - np.sqrt(1. + ep**2))) + ep**2) / ((4. * xb * (1. - xb)) + ep**2)

def compute_t_prime(t, tmin):
    return (t-tmin)

def compute_k_tilde(xb, q_squared, t, tmin, ep):
    return np.sqrt(tmin - t) * np.sqrt(((1. - xb) * np.sqrt((1. + ep**2))) + (((tmin - t) * (ep**2 + (4. * (1. - xb) * xb))) / (4. * q_squared)))

def compute_k(q_squared, y_lep, ep, k_tilde):
    return np.sqrt(((1. - y_lep + (ep**2 * y_lep**2 / 4.)) / q_squared)) * k_tilde

def compute_k_dot_delta(q_squared, xb, t, phi_azi, ep, y_lep, k):
    return (-1.*q_squared / (2.*y_lep*(1.+ep**2))) * (1. + ((2.*k*np.cos(np.pi- phi_azi)) - ((t / q_squared)*(1.-(xb * (2. - y_lep)) + (y_lep * ep**2 / 2.))) + (y_lep * ep**2 / 2.)))

def prop_1(q_squared, kdd):
    return (1. + (2. * (kdd / q_squared)))

def prop_2(q_squared, t, kdd):
    return ((-2. * (kdd / q_squared)) + (t / q_squared))
    
#################################################################################
# We are now prepared to actually *make* all the required directories!
#################################################################################

main_experimental_datafile = pd.read_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/experimental_data_v{MAJOR_MINOR_NUMBER}.csv")

for kinematic_set_number, kinematic_group in main_experimental_datafile.groupby('set'):

    os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data", exist_ok = True)
    os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots", exist_ok = True)
    os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/learning_curves", exist_ok = True)
    os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/replicas", exist_ok = True)
    
    # extract the data for a single kinematic setting:
    set_data = main_experimental_datafile[main_experimental_datafile['set'] == kinematic_set_number]

    # if it's the same kinematic setting, then these mean()s should be the equivalence class value...
    fixed_k = kinematic_group['k'].mean()
    fixed_q_squared = kinematic_group['q_squared'].mean()
    fixed_x_bjorken = kinematic_group['x_b'].mean()
    fixed_t = kinematic_group['t'].mean()
    number_of_phi_points = kinematic_group['phi'].nunique()

    set_data.to_csv(
        f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/main_experimental_file_v{MAJOR_MINOR_NUMBER}.csv", 
        index = False)
    
    print(f"[INFO]: Processed Set {kinematic_set_number}!")

    with open(
        file = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/log_set_{kinematic_set_number}_v{MAJOR_MINOR_NUMBER}.txt",
        mode = "w",
        encoding = "utf-8",
        buffering = 1) as logfile:

        logfile.write(f"[INFO]: Kinematic Set Number: {kinematic_set_number}\n")
        logfile.write(f"[INFO]: k = {fixed_k}\n")
        logfile.write(f"[INFO]: Q^2 = {fixed_q_squared}\n")
        logfile.write(f"[INFO]: xb = {fixed_x_bjorken}\n")
        logfile.write(f"[INFO]: t = {fixed_t}\n")

        fixed_ep = compute_epsilon(fixed_x_bjorken, fixed_q_squared)
        fixed_y = compute_y(fixed_k, fixed_q_squared, fixed_ep)
        fixed_xi = compute_skewness(fixed_x_bjorken, fixed_t, fixed_q_squared)
        fixed_t_min = compute_t_min(fixed_x_bjorken, fixed_q_squared, fixed_ep)
        fixed_k_tilde = compute_k_tilde(fixed_x_bjorken, fixed_q_squared, fixed_t, fixed_t_min, fixed_ep)
        fixed_big_k = compute_k(fixed_q_squared, fixed_y, fixed_ep, fixed_k_tilde)
        fixed_t_prime = compute_t_prime(fixed_t, fixed_t_min)
        fixed_fe = compute_fe(fixed_t)
        fixed_fg = compute_fg(fixed_fe)
        fixed_f2 = compute_f2(fixed_t, fixed_fe, fixed_fg)
        fixed_f1 = compute_f1(fixed_fg, fixed_f2)

        logfile.write(f"[INFO]: ep = {fixed_ep}\n")
        logfile.write(f"[INFO]: y = {fixed_y}\n")
        logfile.write(f"[INFO]: xi = {fixed_xi}\n")
        logfile.write(f"[INFO]: tmin = {fixed_t_min}\n")
        logfile.write(f"[INFO]: Ktilde = {fixed_k_tilde}\n")
        logfile.write(f"[INFO]: K = {fixed_big_k}\n")
        logfile.write(f"[INFO]: tprime = {fixed_t_prime}\n")
        logfile.write(f"[INFO]: FE = {fixed_fe}\n")
        logfile.write(f"[INFO]: FG = {fixed_fg}\n")
        logfile.write(f"[INFO]: F2 = {fixed_f2}\n")
        logfile.write(f"[INFO]: F1 = {fixed_f1}\n")

        logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data\n")
        logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots\n")
        logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/replicas\n")
        logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/learning_curves\n")

        logfile.write(f"[INFO]: Each point has {number_of_phi_points} phi values\n")
        logfile.close()

print("[INFO]: End of script reached!")
