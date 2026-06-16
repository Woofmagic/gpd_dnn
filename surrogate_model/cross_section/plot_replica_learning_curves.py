#################################################################################
# FILE INFORMATION:
# Purpose: makes plots of individual replica learning curves
# Created: 20260602
# Last changed: 20260603
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import glob
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#################################################################################
# Matplotlib Plotting Customizability
#################################################################################

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 3.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 8.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 3.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['savefig.dpi'] = 300

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Collect all of the DNN replica data:
#################################################################################

csv_files = sorted(
    glob.glob(f"./hpc/version_{MAJOR_MINOR_NUMBER}/data/replica_*_history.csv")
    )

print(f"[INFO]: Glob collected {len(csv_files)} files!")

#################################################################################
# Begin iteration over valid sets:
#################################################################################

for csv_file in csv_files:
    file_path = Path(csv_file)
    try:
        # files are: replica_X_history.csv => ["replica", "X", "history", ...]
        replica_number = file_path.name.split('_')[1]
    except IndexError:
        replica_number = "unknown"

    print(f"[INFO]: Processing replica ID{replica_number}")

    df = pd.read_csv(csv_file)

    training_loss_data = df['loss']
    validation_loss_data = df['val_loss']
    # below should usually work...
    number_of_epochs_run = len(df)

    # load this annoying dataframe:
    testing_loss_dataframe = pd.read_csv(f"./hpc/version_{MAJOR_MINOR_NUMBER}/data/replica_{replica_number}_test_metrics.csv")
    # then read its single column...:   
    testing_loss = testing_loss_dataframe['loss'].iloc[0]

#################################################################################
# Making the loss vs. epoch plot:
#################################################################################

    curves_figure, curves_axis = plt.subplots(1, figsize = (8, 8))

    curves_axis.plot(np.arange(0, number_of_epochs_run, 1), np.array([np.max(training_loss_data) for number in training_loss_data]),
                    color = "red", label = "Initial Loss Value")
    curves_axis.plot(np.arange(0, number_of_epochs_run, 1), np.zeros(shape = (number_of_epochs_run)),
                    color = "green", label = r"Loss $= 0$")
    curves_axis.plot(np.arange(0, number_of_epochs_run, 1), training_loss_data,
                    color = "blue", label = "Training Loss")
    curves_axis.plot(np.arange(0, number_of_epochs_run, 1), validation_loss_data,
                    color = "purple", label = "Validation Loss")

    curves_axis.legend(fontsize = 16.)

    curves_axis.set_xlabel("Epoch", fontsize = 16.)
    curves_axis.set_ylabel("MSE Loss", fontsize = 16.)
    curves_axis.set_title(f"Replica {replica_number} Learning Curves\n(Test Loss $= {testing_loss:.3e}$)", fontsize = 18.)

    curves_axis.text(
        0.00, -0.05,
        f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
        transform = curves_axis.transAxes,
        verticalalignment = 'top',
        horizontalalignment = 'left'
    )

    for extension in ['png', 'eps']:
        curves_figure.savefig(
            f"./hpc/version_{MAJOR_MINOR_NUMBER}/learning_curves/lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.{extension}",
            facecolor = 'white',
            transparent = False)

    plt.close(curves_figure)

    del curves_figure
    del curves_axis

#################################################################################
# Making the log(loss) plots:
#################################################################################

    log_curves_figure, log_curves_axis = plt.subplots(1, figsize = (8, 8))

    log_curves_axis.plot(np.arange(0, number_of_epochs_run, 1), np.log(np.array([np.max(training_loss_data) for number in training_loss_data])), 
                        color = "red", label = "Initial Loss Value")
    log_curves_axis.plot(np.arange(0, number_of_epochs_run, 1), np.log(np.zeros(shape = (number_of_epochs_run)) + 1e-20), 
                        color = "green", label = r"Loss $= 0$")
    log_curves_axis.plot(np.arange(0, number_of_epochs_run, 1), np.log(training_loss_data), 
                        color = "blue", label = "Log Training Loss")
    log_curves_axis.plot(np.arange(0, number_of_epochs_run, 1), np.log(validation_loss_data), 
                        color = "purple", label = "Log Validation Loss")

    log_curves_axis.legend(fontsize = 16.)

    log_curves_axis.set_xlabel("Epoch", fontsize = 16.)
    log_curves_axis.set_ylabel("Log MSE Loss", fontsize = 16.)
    log_curves_axis.set_title(f"Replica {replica_number} Learning Curves\n(Test Loss $= {testing_loss:.3e}$)", fontsize = 18.)

    log_curves_axis.text(
        0.00, -0.05,
        f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
        transform = log_curves_axis.transAxes,
        verticalalignment = 'top',
        horizontalalignment = 'left'
    )

    for extension in ['png', 'eps']:
        log_curves_figure.savefig(
            f"./hpc/version_{MAJOR_MINOR_NUMBER}/learning_curves/log_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.{extension}",
            facecolor = 'white',
            transparent = False)

    plt.close(log_curves_figure)

    del log_curves_figure
    del log_curves_axis

    del df

#################################################################################
# Script finishes:
#################################################################################

print("[INFO]: Processing complete!")
