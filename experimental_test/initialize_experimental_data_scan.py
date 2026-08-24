##########################################
# FILE INFORMATION:
# Purpose: make a huge dataset with extant
# experimental data
# Created: 20260325
# Last changed: 20260504
##########################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import yaml

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

with open("./../closure_test/closure_test_config.yml", "r") as file:
    config = yaml.safe_load(file)

MAJOR_NUMBER = config["versioning"]["major"]
MINOR_NUMBER = config["versioning"]["minor"]
MAJOR_MINOR_NUMBER = f"{MAJOR_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

IS_CFF_REAL_H_FREE = config["cff_config"]["enable_cff_real_h"]
IS_CFF_IMAG_H_FREE = config["cff_config"]["enable_cff_imag_h"]
IS_CFF_REAL_HT_FREE = config["cff_config"]["enable_cff_real_ht"]
IS_CFF_IMAG_HT_FREE = config["cff_config"]["enable_cff_imag_ht"]
IS_CFF_REAL_E_FREE = config["cff_config"]["enable_cff_real_e"]
IS_CFF_IMAG_E_FREE = config["cff_config"]["enable_cff_imag_e"]
IS_CFF_REAL_ET_FREE = config["cff_config"]["enable_cff_real_et"]
IS_CFF_IMAG_ET_FREE = config["cff_config"]["enable_cff_imag_et"]

#################################################################################
# Now, we find which gepard sets contain valid kinematics *and* observables!
#################################################################################

target_observables = {'XUU', 'ALU', }

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

for name, indices in desired_observable_dictionary.items():
    print(f"{name}: {indices}")

# rows for model predictions:
rows_for_experimentally_derived_pseudodata = []
# rows for raw experimental data:
rows_for_experimental_data_only = []

for observable_key, experiment_ids in desired_observable_dictionary.items():
    for experiment_id in experiment_ids:

        dataset = g.dset[experiment_id]
        print(f"[INFO]: Now iterating on Experiment {dataset.collaboration} ({dataset.year})")
        
        for datapoint in dataset:
            if not hasattr(datapoint, "observable"):
                print(f"[WARN]: Datapoint for Experiment {g.dset[experiment_id].collaboration} ({g.dset[experiment_id].year}) has no observable...")
            if all(hasattr(datapoint, attr) for attr in ["in1energy", "xB", "Q2", "t", "phi"]):
            
                # predict KM15 CFFs using Gepard's KM15:
                km15_real_h = th_KM15.ReH(datapoint) if IS_CFF_REAL_H_FREE else 0.0
                km15_imag_h = th_KM15.ImH(datapoint) if IS_CFF_IMAG_H_FREE else 0.0
                km15_real_e = th_KM15.ReE(datapoint) if IS_CFF_REAL_HT_FREE else 0.0
                km15_imag_e = th_KM15.ImE(datapoint) if IS_CFF_IMAG_HT_FREE else 0.0
                km15_real_ht = th_KM15.ReHt(datapoint) if IS_CFF_REAL_E_FREE else 0.0
                km15_imag_ht = th_KM15.ImHt(datapoint) if IS_CFF_IMAG_E_FREE else 0.0
                km15_real_et = th_KM15.ReEt(datapoint) if IS_CFF_REAL_ET_FREE else 0.0
                km15_imag_et = th_KM15.ImEt(datapoint) if IS_CFF_IMAG_ET_FREE else 0.0

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
                    verbose = False,
                    debugging = False)
            
                # compute cross-section (XUU)
                unpolarized_cross_section = km15_bkm10_cross_section.compute_cross_section(
                    datapoint.phi, lepton_helicity = 0.0, target_polarization = 0.0).real
                # compute BSA (ALU)
                bkm10_bsa_km15 = km15_bkm10_cross_section.compute_bsa(datapoint.phi, target_polarization = 0.0).real
                # compute AUL (AUL):
                bkm10_lp_target_km15 = km15_bkm10_cross_section.compute_cross_section(
                    datapoint.phi, lepton_helicity = 0.0, target_polarization = +0.5).real
                
                # compute AC (AC) [NOT IMPLEMENTED YET!]
                bkm10_bca_km15 = 0.0 * datapoint.phi
                # compute TSA: [NOT IMPLEMENTED YET!]
                bkm10_tp_target_km15 = 0.0 * datapoint.phi
                # compute XGAMMA: [Implemented, but a pain in the neck]
                bkm10_xgamma_km15 = 0.0 * datapoint.phi

                # need to initialize these garbage variables for looping purposes:
                exp_xgamma, exp_xgamma_err, exp_xgamma_errstat, exp_xgamma_errsyst = 0.0, 0.0, 0.0, 0.0 # DVCS cross-section
                exp_xsec, exp_xsec_err, exp_xsec_errstat, exp_xsec_errsyst = 0.0, 0.0, 0.0, 0.0 # beam-averaged cross-section
                exp_bca, exp_bca_err, exp_bca_errstat, exp_bca_errsyst = 0.0, 0.0, 0.0, 0.0 # beam charge asymmetry
                exp_bsa, exp_bsa_err, exp_bsa_errstat, exp_bsa_errsyst = 0.0, 0.0, 0.0, 0.0 # beam-spin asym.
                exp_lp_xsec, exp_lp_xsec_err, exp_lp_xsec_errstat, exp_lp_xsec_errsyst = 0.0, 0.0, 0.0, 0.0 # longitudinally-polarized target asymmetry
                exp_tp_xsec, exp_tp_xsec_err, exp_tp_xsec_errstat, exp_tp_xsec_errsyst = 0.0, 0.0, 0.0, 0.0 # transversely-polarized target asymmetry

                if observable_key == 'XGAMMA': # DVCS
                    exp_xgamma = datapoint.val
                    exp_xgamma_err = datapoint.err
                    exp_xgamma_errstat = getattr(datapoint, "errstat", datapoint.err)
                    exp_xgamma_errsyst = getattr(datapoint, "errsyst", datapoint.err)
                    
                elif observable_key == 'XUU': # UNPOLARIZED CROSS SECTION
                    exp_xsec = datapoint.val
                    exp_xsec_err = datapoint.err
                    exp_xsec_errstat = getattr(datapoint, "errstat", datapoint.err)
                    exp_xsec_errsyst = getattr(datapoint, "errsyst", datapoint.err)

                elif observable_key == 'AC': # BCA
                    exp_bca = datapoint.val
                    exp_bca_err = datapoint.err
                    exp_bca_errstat = getattr(datapoint, "errstat", datapoint.err)
                    exp_bca_errsyst = getattr(datapoint, "errsyst", datapoint.err)

                elif observable_key == 'ALU': # BSA
                    exp_bsa = datapoint.val
                    exp_bsa_err = datapoint.err
                    exp_bsa_errstat = getattr(datapoint, "errstat", datapoint.err)
                    exp_bsa_errsyst = getattr(datapoint, "errsyst", datapoint.err)

                elif observable_key == 'AUL': # longitudinally-polarized CROSS SECTION
                    exp_lp_xsec = datapoint.val
                    exp_lp_xsec_err = datapoint.err
                    exp_lp_xsec_errstat = getattr(datapoint, "errstat", datapoint.err)
                    exp_lp_xsec_errsyst = getattr(datapoint, "errsyst", datapoint.err)

                elif observable_key == 'TSA': # transversely-polarized CROSS SECTION
                    exp_tp_xsec = datapoint.val
                    exp_tp_xsec_err = datapoint.err
                    exp_tp_xsec_errstat = getattr(datapoint, "errstat", datapoint.err)
                    exp_tp_xsec_errsyst = getattr(datapoint, "errsyst", datapoint.err)

                else:
                    print(f"[ERROR]: Unrecognized observable key: {observable_key}")

                # this is the *row* we will insert into the dataframe:
                experimental_data_point = {
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
                    "unp_target_bca": exp_bca,
                    "unp_target_bca_err": exp_bca_err,
                    "unp_target_bca_errstat": exp_bca_errstat,
                    "unp_target_bca_errsyst": exp_bca_errsyst,
                    "unp_target_lp_target_xsec": exp_lp_xsec,
                    "unp_target_lp_target_xsec_err": exp_lp_xsec_err,
                    "unp_target_lp_target_xsec_errstat": exp_lp_xsec_errstat,
                    "unp_target_lp_target_xsec_errsyst": exp_lp_xsec_errsyst,
                    "unp_target_tp_target_xsec": exp_tp_xsec,
                    "unp_target_tp_target_xsec_err": exp_tp_xsec_err,
                    "unp_target_tp_target_xsec_errstat": exp_tp_xsec_errstat,
                    "unp_target_tp_target_xsec_errsyst": exp_tp_xsec_errsyst,
                    "unp_target_xgamma": exp_xgamma,
                    "unp_target_xgamma_err": exp_xgamma_err,
                    "unp_target_xgamma_errstat": exp_xgamma_errstat,
                    "unp_target_xgamma_errsyst": exp_xgamma_errsyst,
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
                    "unp_target_bca": bkm10_bca_km15,
                    "unp_target_lp_target_xsec": bkm10_lp_target_km15[0], # [0] index needed because datapoint.phi is *not* an array...
                    "unp_target_tp_target_xsec": bkm10_tp_target_km15,
                    "unp_target_xgamma": bkm10_xgamma_km15
                })

                rows_for_experimentally_derived_pseudodata.append(pseudodata_point)
                rows_for_experimental_data_only.append(experimental_data_point)

                del pseudodata_point
                del experimental_data_point
                del km15_bkm10_cross_section
            else:
                print(f"[WARN]: Missing kinematics in {dataset.collaboration}")

####################################
# HERE WE SAVE THE DATAFRAMES!
####################################
kinematic_columns = ['k', 'x_b', 'q_squared', 't']

df_experimentally_derived_pseudodata = pd.DataFrame(rows_for_experimentally_derived_pseudodata)
print(f"[INFO]: Total number of rows in exp-derived DF: {len(df_experimentally_derived_pseudodata)}")

# creates the "set" column:
df_experimentally_derived_pseudodata['set'] = df_experimentally_derived_pseudodata.round(4).groupby(kinematic_columns, sort = False).ngroup() + 1
unique_sets_exp_derived = df_experimentally_derived_pseudodata['set'].nunique()
print(f"[INFO]: Total Datapoints in exp-derived (pseudodata) DF: {unique_sets_exp_derived}")

df_experimentally_derived_pseudodata.to_csv(
    path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/main_pseudodata_file_v{MAJOR_MINOR_NUMBER}.csv",
    index = False)

df_experimental_data = pd.DataFrame(rows_for_experimental_data_only)
print(f"[INFO]: Total number of rows in exp-only DF: {len(df_experimental_data)}")

# creates the "set" column:
df_experimental_data['set'] = df_experimental_data.round(4).groupby(kinematic_columns, sort = False).ngroup() + 1
unique_sets_exp_data = df_experimental_data['set'].nunique()
print(f"[INFO]: Total Datapoints in exp-only (pseudodata) DF: {unique_sets_exp_data}")

df_experimental_data.to_csv(
    path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/experimental_data_v{MAJOR_MINOR_NUMBER}.csv",
    index = False)

print("[INFO]: End of script reached!")
