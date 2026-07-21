####################################################################################################
# FILE INFORMATION:
# Purpose: in case `kinematic_grid.py` does not finish running entirely, we run this
# analysis script to extract the data from each kinematic setting anyway.
# Created: 20260302
# Last changed: 20260721
####################################################################################################

print("[INFO]: Script began running!")

####################################################################################################
# Importing Python Libraries
####################################################################################################

import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("[INFO]: Libraries imported!")

####################################################################################################
# [IMPORTANT]: Static quantities parametrizing the program. Change these if you need!
####################################################################################################

# verify this is what you want
SCRATCH_PATH = 'placeholder!'

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

MAJOR_NUMBER = config["versioning"]["major"]
MINOR_NUMBER = config["versioning"]["minor"]
MAJOR_MINOR_NUMBER = f"{MAJOR_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

####################################################################################################
# Reading the Datafile:
####################################################################################################

# read the dataframe:
test_dataframe = pd.read_csv(
        filepath_or_buffer = f"./kinematic_grid_data_v{MAJOR_MINOR_NUMBER}.csv"
    )

unique_kinematic_sets_dataframe = test_dataframe.groupby(['set'], as_index = False).mean()
number_of_unique_kinematic_sets = len(unique_kinematic_sets_dataframe)

print(f"[INFO]: Total number of unique sets: {number_of_unique_kinematic_sets}")

q_squared_column = unique_kinematic_sets_dataframe["q_squared"]
xb_column = unique_kinematic_sets_dataframe['x_b']
t_column = unique_kinematic_sets_dataframe["t"]

label_plot = ""

for column in test_dataframe.columns:
    
    # unpolarized beam | unpolarized target
    if column == "unp_beam_unp_target_xsec":
        label_plot = r"$d^{4}\sigma^{UU}$"
    # + beam | unpolarized target
    elif column == "plus_beam_unp_target_xsec":
        label_plot = r"$d^{4}\sigma^{+U}$"
    # - beam | unpolarized target
    elif column == "minus_beam_unp_target_xsec":
        label_plot = r"$d^{4}\sigma^{-U}$"
    # unpolarized beam | LP target
    elif column == "unp_beam_lp_target_xsec":
        label_plot = r"$d^{4}\sigma^{UL}$"
    # + beam | LP target
    elif column == "plus_beam_lp_target_xsec":
        label_plot = r"$d^{4}\sigma^{+L}$"
    # - beam | LP target
    elif column == "minus_beam_lp_target_xsec":
        label_plot = r"$d^{4}\sigma^{-L}$"

    # BSA | unpolarized target:
    elif column == "unp_target_bsa":
        label_plot = r"$\text{BSA} \left( \Lambda = 0 \right)$"
    # BSA | + target:
    elif column == "plus_target_bsa":
        label_plot = r"$\text{BSA} \left( \Lambda = +1/2 \right)$"
    # BSA | - target:
    elif column == "minus_target_bsa":
        label_plot = r"$\text{BSA} \left( \Lambda = -1/2 \right)$"

    # TSA | unpolarized beam:
    elif column == "unp_beam_tsa":
        label_plot = r"$\text{BSA} \left( \Lambda = 0 \right)$"
    # TSA | + beam:
    elif column == "plus_beam_tsa":
        label_plot = r"$\text{BSA} \left( \lambda = +1 \right)$"
    # TSA | - target:
    elif column == "minus_beam_tsa":
        label_plot = r"$\text{BSA} \left( \lambda = -1 \right)$"

    # DSA |
    elif column == "dsa":
        label_plot = r"$\text{DSA}$"

    else:
        continue

    input_space_figure = plt.figure(figsize = (9, 7))
    input_space_axis = input_space_figure.add_subplot(1, 1, 1, projection = "3d")
    input_space_axis.scatter(xb_column, q_squared_column, t_column, alpha = 0.3, s = 6.4)
    input_space_axis.set_xlabel(r"$x_{\text{B}}$")
    input_space_axis.set_ylabel(r"$Q^{2}$")
    input_space_axis.set_zlabel(r"$-t$")
    input_space_axis.set_title(rf"Kinematic Settings Space for {label_plot}")

    input_space_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/kinematic_space_for_{column}_v{MAJOR_MINOR_NUMBER}.png")
    input_space_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/kinematic_space_for_{column}_v{MAJOR_MINOR_NUMBER}.eps")
    plt.close(input_space_figure)

    t_vs_xb_figure, t_vs_x_axis = plt.subplots(1, 1, figsize = (8, 7))
    t_vs_x_axis.scatter(xb_column, t_column, alpha = 0.3, s = 10.4)
    t_vs_x_axis.set_xlabel(r"$x_{\text{B}}$")
    t_vs_x_axis.set_ylabel(r"$-t$")
    t_vs_x_axis.set_title(r"$-t$ vs. $x_{\text{B}}$")

    t_vs_xb_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/t_vs_xb_for_{column}_v{MAJOR_MINOR_NUMBER}.png")
    t_vs_xb_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/t_vs_xb_for_{column}_v{MAJOR_MINOR_NUMBER}.eps")
    plt.close(t_vs_xb_figure)

    qsq_vs_t_figure, qsq_vs_t_axis = plt.subplots(1, 1, figsize = (8, 7))
    qsq_vs_t_axis.scatter(q_squared_column, t_column, alpha = 0.1, s = 10.4)
    qsq_vs_t_axis.set_xlabel(r"$Q^{2}$")
    qsq_vs_t_axis.set_ylabel(r"$-t$")
    qsq_vs_t_axis.set_title(r"$-t$ vs. $Q^{2}$")

    qsq_vs_t_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/qsquared_vs_t_for_{column}_v{MAJOR_MINOR_NUMBER}.png")
    qsq_vs_t_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/qsquared_vs_t_for_{column}_v{MAJOR_MINOR_NUMBER}.eps")
    plt.close(qsq_vs_t_figure)

    xb_vs_qsq_figure, xb_vs_qsq_axis = plt.subplots(1, 1, figsize = (8, 7))
    xb_vs_qsq_axis.scatter(xb_column, q_squared_column, alpha = 0.1, s = 10.4)
    xb_vs_qsq_axis.set_xlabel(r"$x_{\text{B}}$")
    xb_vs_qsq_axis.set_ylabel(r"$Q^{2}$")
    xb_vs_qsq_axis.set_title(r"$Q^{2}$ vs. $x_{\text{B}}$")

    xb_vs_qsq_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/xb_vs_qsquared_for_{column}_v{MAJOR_MINOR_NUMBER}.png")
    xb_vs_qsq_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/xb_vs_qsquared_for_{column}_v{MAJOR_MINOR_NUMBER}.eps")
    plt.close(xb_vs_qsq_figure)

####################################################################################################
# end the script
####################################################################################################

print("[INFO]: End of script reached!")
