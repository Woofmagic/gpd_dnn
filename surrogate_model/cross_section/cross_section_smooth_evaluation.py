#################################################################################
# FILE INFORMATION:
# Purpose: evaluate the DNN with a smooth input of independent variables
# Created: 20260602
# Last changed: 20260602
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import sys

import pandas as pd
import numpy as np
import tensorflow as tf

#################################################################################
# HPC logic:
#################################################################################

replica_number = int(sys.argv[1])

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

#################################################################################
# Smooth replica surface across t, xb, q_squared, and phi:
#################################################################################

print("[INFO]: Computing smooth phi predictions...")

t_min = x_data['t'].min()
t_max = x_data['t'].max()

print(f"[INFO]: bounding for t: {t_min} < t < {t_max}")

xb_min = x_data['x_b'].min()
xb_max = x_data['x_b'].max()

print(f"[INFO]: bounding for xb: {xb_min} < x_b < {xb_max}")

q2_min = x_data['q_squared'].min()
q2_max = x_data['q_squared'].max()

print(f"[INFO]: bounding for Q^2: {q2_min} < Q^2 < {q2_max}")

NUMBER_OF_T = 10
NUMBER_OF_XB = 10
NUMBER_OF_Q2 = 10
NUMBER_OF_PHI = 361

t_grid = np.round(np.linspace(t_min, t_max, NUMBER_OF_T), 3)
xb_grid = np.round(np.linspace(xb_min, xb_max, NUMBER_OF_XB), 4)
q2_grid = np.round(np.linspace(q2_min, q2_max, NUMBER_OF_Q2), 3)
phi_grid = np.linspace(-np.pi, np.pi, NUMBER_OF_PHI)

mesh = np.meshgrid(t_grid, xb_grid, q2_grid, phi_grid, indexing = 'ij')

t_flat = mesh[0].ravel()
xb_flat = mesh[1].ravel()
q2_flat = mesh[2].ravel()
phi_flat = mesh[3].ravel()

smooth_input = pd.DataFrame({
    't': t_flat,
    'x_b': xb_flat,
    'q_squared': q2_flat,
    'phi': phi_flat
})

smooth_predictions = dnn_model.predict(smooth_input, verbose = 0)

# making predictions
smooth_input['pred_xsec'] = smooth_predictions[:, 0]

smooth_input.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/"
    f"replica_{replica_number}_smooth_predictions.csv",
    index = False
)

print("[INFO]: Smooth predictions saved.")