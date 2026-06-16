#################################################################################
# FILE INFORMATION:
# Purpose: averages over all the cross-section surrogate replicas
# Created: 20260603
# Last changed: 20260603
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import glob

import pandas as pd
import numpy as np

print("[INFO]: Libraries imported!")

#################################################################################
# Scratch path
#################################################################################

SCRATCH_PATH = 'placeholder!'

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Helper function that computes the statistics of a columnar distribution:
#################################################################################

def crunch_statistics(data):

    stats = {
        "mean": np.mean(data, axis = 0),
        "std": np.std(data, axis = 0),
        "min": np.min(data, axis = 0),
        "max": np.max(data, axis = 0),
    }

    for p in [10, 20, 30, 40, 60, 70, 80, 90]:
        stats[f"p{p}"] = np.percentile(data, p, axis = 0)

    return stats

#################################################################################
# Collecting the individual replica predictions:
#################################################################################

# find every replica's unique predictions:
individual_replica_predictions = sorted(
    glob.glob(
        f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/replica_*_test_predictions.csv"
        )
    )

print(f"[INFO]: Loaded {len(individual_replica_predictions)} individual replica datasets.")

#################################################################################
# Iterate over all of the individual replica predictions to construct the 
# datatype we need to make a statistical distribution
#################################################################################

replica_predictions = []

# we need a reference dataframe to pull out the columns:
reference_df = None

for replica_index, replica_prediction in enumerate(individual_replica_predictions):

    df = pd.read_csv(replica_prediction)

    if reference_df is None:
        reference_df = df.copy()

    replica_predictions.append(df["predicted_cross_section"].to_numpy())

    del df

# this creates a matrix of shape (replicas, predictions)
replica_predictions = np.vstack(replica_predictions)
print(f"[INFO] Prediction matrix shape = {replica_predictions.shape}")

#################################################################################
# We now actually make the statistical distribution:
#################################################################################

cross_section_prediction_statistics = crunch_statistics(replica_predictions)

output_df = reference_df.drop(columns = ["predicted_cross_section", "replica_number"]).copy()

output_df["mean_predicted_cross_section"] = cross_section_prediction_statistics["mean"]
output_df["std_predicted_cross_section"] = cross_section_prediction_statistics["std"]
output_df["min_predicted_cross_section"] = cross_section_prediction_statistics["min"]
output_df["max_predicted_cross_section"] = cross_section_prediction_statistics["max"]

# [NOTE]: you know why we exclude 50, right?
for percentile in [10, 20, 30, 40, 60, 70, 80, 90]:
    output_df[f"p{percentile}_predicted_cross_section"] = cross_section_prediction_statistics[f"p{percentile}"]

#################################################################################
# Save the file:
#################################################################################

output_df.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/surrogate_cross_section_replica_average_v{MAJOR_MINOR_NUMBER}.csv", 
    index = False)

#################################################################################
# Script finishes:
#################################################################################

print("[INFO]: End of script reached!")
