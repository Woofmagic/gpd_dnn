#################################################################################
# FILE INFORMATION:
# Purpose: make a huge dataset with extant experimental data
# Created: 20260325
# Last changed: 20260528
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import pandas as pd
import gepard as g
from gepard.fits import th_KM15
from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

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
# Now, we find which gepard sets contain valid kinematics *and* observables!
#################################################################################

target_observables = {'XUU', 'ALU'}

# dictionary of string-to-list key-value pairs:
desired_observable_dictionary = {observable: [] for observable in target_observables}
required_attributes = ['xB', 't', 'Q2', 'in1energy', 'observable', 'val', 'err']
valid_datasets = []

for dataset_index, dataset in g.dset.items():

    # check the first datapoint in the DataSet for contents!
    first_gepard_datapoint = dataset[0] if len(dataset) > 0 else None

    if first_gepard_datapoint and all(hasattr(first_gepard_datapoint, kinematic_attribute) for kinematic_attribute in required_attributes):
        valid_datasets.append(dataset_index)

print(f"[INFO]: Valid dataset indices are:\n{sorted(valid_datasets)}")
print(f"[INFO]: Length of valid datasets = {len(valid_datasets)}")
print(f"[INFO]: Compare length of *all* dataset = {len(g.dset)}")
print(f"[INFO]: Total invalid (according to our criteria) datasets = {len(g.dset) - len(valid_datasets)}")

for dataset_index in valid_datasets:
    if not hasattr(g.dset[dataset_index][0], "observable"):
        print(f"[WARN]: Dataset {g.dset[dataset_index]} had datapoint without observable property...")
        continue

    observable_name = g.dset[dataset_index][0].observable

    if observable_name in desired_observable_dictionary:
        desired_observable_dictionary[observable_name].append(dataset_index)

#################################################################################
# The MAJOR loop of the code!
#################################################################################

for name, indices in desired_observable_dictionary.items():
    print(f"[INFO]: {name}: {indices}")

# rows for model predictions:
rows_for_experiment_w_ground_truth = []
# rows for raw experimental data:
rows_for_experimental_data_only = []

total_rows = 0

for observable_key, experiment_ids in desired_observable_dictionary.items():
    for experiment_id in sorted(experiment_ids):

        # query the dataset from gepard:
        dataset = g.dset[experiment_id]
        print(f"[INFO]: Experiment {dataset.collaboration} ({dataset.year}), ID = {experiment_id}, {len(dataset)} datapoints")
        
        for datapoint_index, datapoint in enumerate(dataset):

            total_rows = total_rows + 1

            # this should always pass:
            if not hasattr(datapoint, "observable"):
                print(f"[WARN]: Datapoint for Experiment {dataset.collaboration} ({dataset.year}) ID = {experiment_id} has no observable...")
            
            # check if the datapoint has all of the required kinematic variables...
            if all(hasattr(datapoint, attr) for attr in ["in1energy", "xB", "Q2", "t", "phi"]):
            
                # predict KM15 CFFs using Gepard's KM15:
                km15_real_h = th_KM15.ReH(datapoint)
                km15_imag_h = th_KM15.ImH(datapoint)
                km15_real_e = th_KM15.ReE(datapoint)
                km15_imag_e = th_KM15.ImE(datapoint)
                km15_real_ht = th_KM15.ReHt(datapoint)
                km15_imag_ht = th_KM15.ImHt(datapoint)
                km15_real_et = th_KM15.ReEt(datapoint)
                km15_imag_et = th_KM15.ImEt(datapoint)

                # initialize a BKM10 computation hub:
                km15_bkm10_cross_section = DifferentialCrossSection(
                    configuration = {
                        "kinematics": BKM10Inputs(
                            lab_kinematics_k = datapoint.in1energy,
                            squared_Q_momentum_transfer = datapoint.Q2,
                            x_Bjorken = datapoint.xB,
                            squared_hadronic_momentum_transfer_t = datapoint.t),
                        "cff_inputs": CFFInputs(
                            compton_form_factor_h = complex(km15_real_h, km15_imag_h),
                            compton_form_factor_h_tilde = complex(km15_real_ht, km15_imag_ht),
                            compton_form_factor_e = complex(km15_real_e, km15_imag_e),
                            compton_form_factor_e_tilde = complex(km15_real_et, km15_imag_et)),
                        "using_ww": True
                    },
                    verbose = False, debugging = False)
            
                # compute cross-section (XUU)
                unpolarized_cross_section = km15_bkm10_cross_section.compute_cross_section(
                    datapoint.phi, lepton_helicity = 0.0, target_polarization = 0.0).real
                # compute BSA (ALU)
                bkm10_bsa_km15 = km15_bkm10_cross_section.compute_bsa(
                    datapoint.phi, target_polarization = 0.0).real

                # need to initialize these garbage variables for looping purposes:
                exp_xsec, exp_xsec_err, exp_xsec_errstat, exp_xsec_errsyst = 0.0, 0.0, 0.0, 0.0 # beam-averaged cross-section
                exp_bsa, exp_bsa_err, exp_bsa_errstat, exp_bsa_errsyst = 0.0, 0.0, 0.0, 0.0 # beam-spin asym.
                    
                if observable_key == 'XUU': # UNPOLARIZED CROSS SECTION
                    exp_xsec = datapoint.val
                    exp_xsec_err = datapoint.err
                    exp_xsec_errstat = getattr(datapoint, "errstat", datapoint.err)
                    exp_xsec_errsyst = getattr(datapoint, "errsyst", datapoint.err)

                elif observable_key == 'ALU': # BSA
                    exp_bsa = datapoint.val
                    exp_bsa_err = datapoint.err
                    exp_bsa_errstat = getattr(datapoint, "errstat", datapoint.err)
                    exp_bsa_errsyst = getattr(datapoint, "errsyst", datapoint.err)

                else:
                    print(f"[ERROR]: Unrecognized observable key: {observable_key}")

                # this is the *row* we will insert into the dataframe:
                experimental_data_point = {
                    "experiment_id": experiment_id,
                    "k": datapoint.in1energy,
                    "q_squared": datapoint.Q2,
                    "x_b": datapoint.xB,
                    "t": datapoint.t,
                    "phi": datapoint.phi,
                    "unp_beam_unp_target_xsec": exp_xsec,
                    "unp_beam_unp_target_xsec_err": exp_xsec_err,
                    "unp_beam_unp_target_xsec_errstat": exp_xsec_errstat,
                    "unp_beam_unp_target_xsec_errsyst": exp_xsec_errsyst,
                    "unp_target_bsa": exp_bsa,
                    "unp_target_bsa_err": exp_bsa_err,
                    "unp_target_bsa_errstat": exp_bsa_errstat,
                    "unp_target_bsa_errsyst": exp_bsa_errsyst,
                    "Re[H]": km15_real_h, "Im[H]": km15_imag_h,
                    "Re[E]": km15_real_e, "Im[E]": km15_imag_e,
                    "Re[Ht]": km15_real_ht, "Im[Ht]": km15_imag_ht,
                    "Re[Et]": km15_real_et, "Im[Et]": km15_imag_et,
                    "coordinate_frame": datapoint.frame,
                    "experiment_year": f"{dataset.collaboration}_{dataset.year}",
                    "flag": "unknown"
                }

                pseudodata_point = experimental_data_point.copy()
                pseudodata_point.update({
                    "unp_beam_unp_target_xsec": unpolarized_cross_section[0], # [0] index needed because datapoint.phi is *not* an array...
                    "unp_target_bsa": bkm10_bsa_km15[0], # [0] index needed because datapoint.phi is *not* an array...
                })

                rows_for_experiment_w_ground_truth.append(pseudodata_point)
                rows_for_experimental_data_only.append(experimental_data_point)

                del pseudodata_point
                del experimental_data_point
                del km15_bkm10_cross_section
            else:
                print(f"[WARN]: Missing kinematics for datapoint {datapoint_index + 1} in {dataset.collaboration}, ID = {experiment_id}")
                
print(f"[INFO]: Total rows expected: {total_rows}")
#################################################################################
# Saving the dataframes:
#################################################################################

kinematic_columns = ['k', 'x_b', 'q_squared', 't']

##########################################################################################
# [NOTE]: this part of the code makes a dataset with "ground truth" values:
##########################################################################################

df_exp_data_w_ground_truth = pd.DataFrame(rows_for_experiment_w_ground_truth)
print(f"[INFO]: Total number of rows in exp-derived DF: {len(df_exp_data_w_ground_truth)}")

# this relabels the sets actually...
df_exp_data_w_ground_truth['set'] = df_exp_data_w_ground_truth.groupby(kinematic_columns, sort = False).ngroup() + 1
unique_sets_exp_derived = df_exp_data_w_ground_truth['set'].nunique()
print(f"[INFO]: Total unique kinematic settings in the exp-derived DF: {unique_sets_exp_derived}")

df_exp_data_w_ground_truth.to_csv(
    path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/main_pseudodata_file_v{MAJOR_MINOR_NUMBER}.csv",
    index = False)

##########################################################################################
# [NOTE]: this part of the code makes a dataset with *only* the experimental data---not any CFF info!
##########################################################################################

df_experimental_data = pd.DataFrame(rows_for_experimental_data_only)
print(f"[INFO]: Total number of rows in exp-only DF: {len(df_experimental_data)}")

# relabeling the sets:
df_experimental_data['set'] = df_experimental_data.groupby(kinematic_columns, sort = False).ngroup() + 1
unique_sets_exp_data = df_experimental_data['set'].nunique()
print(f"[INFO]: Total unique kinematic settings in the  exp-only DF: {unique_sets_exp_data}")

df_experimental_data.to_csv(
    path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/experimental_data_v{MAJOR_MINOR_NUMBER}.csv",
    index = False)

print("[INFO]: End of script reached!")
