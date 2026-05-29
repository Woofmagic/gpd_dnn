#################################################################################
# FILE INFORMATION:
# Purpose: makes plots corresponding to replica DNN training/validation/testing data
# Created: 20260402
# Last changed: 20260402
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import glob
import datetime

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#################################################################################
# Matplotlib Plotting Customizability
#################################################################################

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 8.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['savefig.dpi'] = 300

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Read the textfile:
#################################################################################

with open(
    f"./hpc/errors/version_{MAJOR_MINOR_NUMBER}/valid_kinematic_sets_v{MAJOR_MINOR_NUMBER}.txt", 
    'r',
    encoding = "utf8") as valid_sets_file:
    valid_sets = [line.strip() for line in valid_sets_file if line.strip()]

print(f"[INFO]: Found {len(valid_sets)} valid kinematic sets.")

#################################################################################
# Begin iteration over valid sets:
#################################################################################

for valid_kinematic_set in valid_sets:
    print(f"[INFO]: Now processing set #{valid_kinematic_set}")

    csv_files = sorted(
        glob.glob(
            f"./hpc/errors/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{valid_kinematic_set}/data/dnn_data_replica_*_v{MAJOR_MINOR_NUMBER}.csv"))

    print(f"[INFO]: Glob collected {len(csv_files)} files!")
    
    for csv_file in csv_files:
        file_path = Path(csv_file)
        try:
            # files are: dnn_data_replica_X_vY_Z.csv => ["dnn", "data", "replica", X, ...]
            replica_id = file_path.name.split('_')[3]
        except IndexError:
            replica_id = "unknown"

        print(f"[INFO]: Processing replica ID{replica_id}")
    
        df = pd.read_csv(csv_file)
        
        # kinematic setting
        # we can index them in this way because they're fixed!
        this_kinematic_set_title_string = (
            rf"$k = {df['k'].iloc[0]:.2f}$ GeV, "
            rf"$x_B = {df['x_b'].iloc[0]:.2f}$, "
            rf"$t = {df['t'].iloc[0]}$, "
            rf"$Q^2 = {df['q_squared'].iloc[0]}$ GeV$^2$"
        )

        cff_h_km15 = complex(df['Re[H]'].iloc[0], df['Im[H]'].iloc[0])
        cff_ht_km15 = complex(df['Re[Ht]'].iloc[0], df['Im[Ht]'].iloc[0])
        cff_e_km15 = complex(df['Re[E]'].iloc[0], df['Im[E]'].iloc[0])
        cff_et_km15 = complex(df['Re[Et]'].iloc[0], df['Im[Et]'].iloc[0])

        km15_cff_string = (
            rf"$\mathcal{{H}} = {cff_h_km15:.3f}$, "
            rf"$\mathcal{{E}} = {cff_e_km15:.3f}$, "
            rf"$\widetilde{{\mathcal{{H}}}} = {cff_ht_km15:.3f}$, "
            rf"$\widetilde{{\mathcal{{E}}}} = {cff_et_km15:.3f}$ "
        )

        data_vis_figure, data_vis_axis = plt.subplots(1, 1, figsize = (10, 9))
        palette = {'train': 'green', 'validation': 'orange', 'test': 'red'}

        for split, color in palette.items():
            subset = df[df['split'] == split]
            if split == 'train': label = "Training Points"
            if split == 'validation': label = "Validation Points"
            if split == 'test': label = "Testing Points"

            data_vis_axis.errorbar(
                x = subset['phi'], y = subset['unp_beam_unp_target_xsec'], yerr = subset['unp_beam_unp_target_xsec_err'],
                color = color, capsize = 4, fmt = 'o', label = rf"{label} ($N$ = {len(subset)})", alpha = 0.7)

        data_vis_axis.legend(fontsize = 14.)

        data_vis_axis.set_xlabel(r"$\phi$ [radians]", fontsize = 16.)
        data_vis_axis.set_ylabel(r"$d^{4}\sigma^{UU}$", fontsize = 16.)
        data_vis_axis.set_title(
            rf"(Set {valid_kinematic_set}, Replica {replica_id}) $d^{{4}}\sigma^{{UU}}$ vs. $\phi$, {this_kinematic_set_title_string}"
            "\n"
            f"(KM15): {km15_cff_string}", fontsize = 16)
        data_vis_axis.grid(visible = True)
        data_vis_axis.text(
            0.00, -0.05,
            f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
            transform = data_vis_axis.transAxes,
            verticalalignment = 'top',
            horizontalalignment = 'left'
        )

        for extension in ['png', 'eps']:
            data_vis_figure.savefig(
                fname = f"./hpc/errors/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{valid_kinematic_set}/plots/datasplit_xsec_replica_{replica_id}_v{MAJOR_MINOR_NUMBER}.{extension}",
                facecolor = 'white',
                transparent = False)
            
        plt.close(data_vis_figure)

        del data_vis_figure
        del data_vis_axis
        
        data_vis_figure, data_vis_axis = plt.subplots(1, 1, figsize = (10, 9))
        palette = {'train': 'green', 'validation': 'orange', 'test': 'red'}

        for split, color in palette.items():
            subset = df[df['split'] == split]
            if split == 'train': label = "Training Points"
            if split == 'validation': label = "Validation Points"
            if split == 'test': label = "Testing Points"

            data_vis_axis.errorbar(
                x = subset['phi'], y = subset['unp_target_bsa'], yerr = subset['unp_target_bsa_err'],
                color = color, capsize = 2, fmt = 'o', label = rf"{label} ($N$ = {len(subset)})", alpha = 0.7)

        data_vis_axis.legend(fontsize = 14.)

        data_vis_axis.set_xlabel(r"$\phi$ [radians]", fontsize = 16.)
        data_vis_axis.set_ylabel(r"BSA $\left( \Lambda = 0 \right)$", fontsize = 16.)
        data_vis_axis.set_title(
            rf"(Set {valid_kinematic_set}, Replica {replica_id}) BSA $\left( \Lambda = 0 \right)$ vs. $\phi$, {this_kinematic_set_title_string}"
            "\n"
            f"(KM15): {km15_cff_string}", fontsize = 16)
        data_vis_axis.grid(visible = True)
        data_vis_axis.text(
            0.00, -0.05,
            f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
            transform = data_vis_axis.transAxes,
            verticalalignment = 'top',
            horizontalalignment = 'left'
        )

        for extension in ['png', 'eps']:
            data_vis_figure.savefig(
                fname = f"./hpc/errors/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{valid_kinematic_set}/plots/datasplit_bsa_replica_{replica_id}_v{MAJOR_MINOR_NUMBER}.{extension}",
                facecolor = 'white',
                transparent = False)
            
        plt.close(data_vis_figure)

        del data_vis_figure
        del data_vis_axis

        del df

print("[INFO]: Processing complete!")
