#################################################################################
# FILE INFORMATION:
# Purpose: find any duplicate rows in the original file and combine the observables
# along that row.
# Created: 20260504
# Last changed: 20260528
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import numpy as np
import pandas as pd

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
# Define the static variables:
#################################################################################

kinematic_set_columns = ['set', 'k', 'x_b', 'q_squared', 't', 'phi']

# [NOTE]: Remember that we're not doing BCA, TSA, or XGAMMA yet:
# observable_columnnames = [
#     "unp_beam_unp_target_xsec", "unp_beam_unp_target_xsec_err", "unp_beam_unp_target_xsec_errsyst", "unp_beam_unp_target_xsec_errstat",
#     "unp_target_bsa", "unp_target_bsa_err", "unp_target_bsa_errsyst", "unp_target_bsa_errstat", 
#     "unp_target_bca", "unp_target_bca_err", "unp_target_bca_errsyst", "unp_target_bca_errstat", 
#     "unp_target_lp_target_xsec", "unp_target_lp_target_xsec_err", "unp_target_lp_target_xsec_errsyst", "unp_target_lp_target_xsec_errstat", 
#     "unp_target_tp_target_xsec", "unp_target_tp_target_xsec_err", "unp_target_tp_target_xsec_errsyst", "unp_target_tp_target_xsec_errstat", 
#     "unp_target_xgamma", "unp_target_xgamma_err", "unp_target_xgamma_errsyst", "unp_target_xgamma_errstat", 
#     ]

observable_columnnames = [
    "unp_beam_unp_target_xsec", "unp_beam_unp_target_xsec_err", "unp_beam_unp_target_xsec_errsyst", "unp_beam_unp_target_xsec_errstat",
    "unp_target_bsa", "unp_target_bsa_err", "unp_target_bsa_errsyst", "unp_target_bsa_errstat", 
    ]

#################################################################################
# Read the two files:
#################################################################################

df_experimental_data = pd.read_csv(
    f'{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/main_pseudodata_file_v{MAJOR_MINOR_NUMBER}.csv')

df_length_before_merge = len(df_experimental_data)
print(f"[INFO]: Number of initial rows: {df_length_before_merge}")
df_cleaned = df_experimental_data.drop_duplicates().copy()
print(f"[INFO]: Number of remaining rows after initial drop: {len(df_cleaned)}")

#################################################################################
# Begin the logic of combining redundant data:
#################################################################################

df_unique_experimental_settings = df_cleaned.groupby(
    [ 'k', 'q_squared', 'x_b', 't', 'phi'],
    sort = False)

print(f"[INFO]: Found {len(df_unique_experimental_settings)} unique kinematic points.")

merged_rows = []

number_of_fragmented_points = 0
number_of_successful_merges = 0
number_of_all_zero_observables = 0
number_of_fully_zero_rows_removed = 0
number_of_conflicts = 0

for group_key, group_df in df_unique_experimental_settings:

    if len(group_df) == 1:
        # just take the entire row and shove it into the list:
        merged_rows.append(group_df.iloc[0])
        continue

    print(f"[INFO]: Found {len(group_df)} rows at {group_key}")
    number_of_fragmented_points += 1

    # define a "template row" that we'll now dynamically change:
    merged_row = group_df.iloc[0].copy()

    for observable in observable_columnnames:

        values = group_df[observable].values
        # this removes NaN values, but we don't really expect that:
        values = values[~pd.isna(values)]

        if len(values) == 0:
            print("[WARN]: All values are NaN.")
            merged_row[observable] = np.nan

        nonzero_values = values[~np.isclose(values, 0.0)]

        if len(nonzero_values) == 0:

            print(f"[WARN]: All values are zero at {group_key}")
            number_of_all_zero_observables += 1
            merged_row[observable] = 0.0

            continue

        unique_nonzero_values = np.unique(nonzero_values)
        # print(f"[INFO]: Nonzero values: {unique_nonzero_values}")

        if len(unique_nonzero_values) == 1:

            merged_value = unique_nonzero_values[0]

            # print(f"[INFO]: Safe merge with value {merged_value}")
            merged_row[observable] = merged_value
            number_of_successful_merges += 1

        else:
            print(f"[ERROR]: Conflicting nonzero values detected: {unique_nonzero_values}")
            number_of_conflicts += 1
            merged_row[observable] = unique_nonzero_values[0]

    merged_observable_values = pd.to_numeric(merged_row[observable_columnnames], errors = 'coerce' ).values
    merged_observable_values = merged_observable_values[ ~pd.isna(merged_observable_values)]
    all_observables_zero = np.all(np.isclose(merged_observable_values, 0))

    if all_observables_zero:
        print("[WARN]: Entire merged row has zero observables.")

        number_of_fully_zero_rows_removed += 1

        continue

    merged_rows.append(merged_row)

df_removed_redunancies = pd.DataFrame(merged_rows)

print(f"[INFO]: Fragmented points found: {number_of_fragmented_points}")
print(f"[INFO]: Successful observable merges: {number_of_successful_merges}")
print(f"[INFO]: All-zero observable cases: {number_of_all_zero_observables}")
print(f"[INFO]: Conflicting observables detected: {number_of_conflicts}")
print(f"[INFO]: Final dataframe rows: {len(df_removed_redunancies)}")

df_removed_redunancies.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/refined_experimental_data_v{MAJOR_MINOR_NUMBER}.csv", 
    index = False)

print("[INFO]: End of script reached!")
