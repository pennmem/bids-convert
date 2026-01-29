import cmlreaders as cml
import mne
import numpy as np
import pandas as pd
pd.options.display.max_columns=None
pd.options.display.max_rows=200
import os
from scipy.io import loadmat
# from ScalpBIDSConverter import *
import cmldask.CMLDask as da
import os, sys, importlib

# add the package ROOT, not the intracranial folder
sys.path.insert(0, os.path.expanduser("~/bids-convert"))

import intracranial.run_intracranial_convert_maint as rim
importlib.reload(rim)  # if you edited it recently

# now this should work because relative imports have a parent package

from intracranial.intracranial_BIDS_converter import intracranial_BIDS_converter

from intracranial.catFR1.catFR1_BIDS_converter import catFR1_BIDS_converter
from intracranial.catFR2.catFR2_BIDS_converter import catFR2_BIDS_converter
from intracranial.FR1.FR1_BIDS_converter import FR1_BIDS_converter
from intracranial.FR2.FR2_BIDS_converter import FR2_BIDS_converter
from intracranial.PAL1.PAL1_BIDS_converter import PAL1_BIDS_converter
from intracranial.PAL2.PAL2_BIDS_converter import PAL2_BIDS_converter
from intracranial.pyFR.pyFR_BIDS_converter import pyFR_BIDS_converter
from intracranial.RepFR1.RepFR1_BIDS_converter import RepFR1_BIDS_converter
from intracranial.YC1.YC1_BIDS_converter import YC1_BIDS_converter
from intracranial.YC2.YC2_BIDS_converter import YC2_BIDS_converter

# # import sys
# # sys.path.insert(0, "/ABS/PATH/to/bids-convert/intracranial")

# %matplotlib inline

def convert_to_bids(subject, experiment, session,system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root="/home1/maint/LTP_BIDS/"):
    # if os.path.exists(f"/data8/PEERS_BIDS/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{experiment}_events.json"):
    #     if (os.path.getmtime(f"/data8/PEERS_BIDS/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{experiment}_events.json") > 1684160000):
    #         return True
    root = '/home1/maint/LTP_BIDS/'
    if experiment == "catFR1":
        converter = catFR1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "catFR2":
        converter = catFR2_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "FR1":
        converter = FR1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "FR2":
        converter = FR2_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "PAL1":
        converter = PAL1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "PAL2":
        converter = PAL2_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "pyFR":
        converter = pyFR_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "RepFR1":
        converter = RepFR1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "YC1":
        converter = YC1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    elif experiment == "YC2":
        converter = YC2_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
    return converter.run()

if __name__ == "__main__":
    df = cml.get_data_index()
    max_subjects = 10
    experiments = ["catFR1", "catFR2", "FR1", "FR2", "PAL1", "PAL2", "pyFR", "RepFR1", "YC1", "YC2"]

    # subjects_to_exclude = {"LTP001"}  # <-- your list here

    df = cml.get_data_index()

    df_exp = df[df["experiment"].isin(experiments)].copy()

    # remove excluded subjects up front
    df_exp = df_exp[~df_exp["subject"].isin(subjects_to_exclude)].copy()

    dfs = []

    for exp in experiments:
        df_this = df_exp[df_exp["experiment"] == exp]

        subjects = (
            df_this["subject"]
            .drop_duplicates()
            .sort_values()      # deterministic
            .head(max_subjects)
        )

        df_keep = df_this[df_this["subject"].isin(subjects)].copy()
        dfs.append(df_keep)

    df_subset = pd.concat(dfs, ignore_index=True)
    
    client = da.new_dask_client_slurm(
        job_name="bids_convert",
        memory_per_job="50GB",
        max_n_jobs=10, threads_per_job=1, 
        adapt=True,
        log_directory="~/logs/",
    )
    conversion_df = pd.read_csv('system_1_unit_conversions.csv')
    # df_jobs = df_subset[["subject", "experiment", "session"]].copy()

    # conversion_df = pd.read_csv("system_1_unit_conversions.csv")

    # Normalize dtypes (session often mismatches int vs str)
    df_jobs["session"] = df_jobs["session"].astype(int)
    conversion_df["session"] = conversion_df["session"].astype(int)

    # Merge on the actual keys
    merge_keys = ["subject", "experiment", "session"]

    df_jobs2 = df_jobs.merge(
        conversion_df[merge_keys + ["system_version", "conversion_to_V"]],
        on=merge_keys,
        how="left",
    )

    # Drop non-matching rows
    missing = df_jobs2["system_version"].isna() | df_jobs2["conversion_to_V"].isna()
    if missing.any():
        print(f"Skipping {missing.sum()} job(s) with no matching conversion row:")
        print(df_jobs2.loc[missing, merge_keys].to_string(index=False))

    df_jobs2 = df_jobs2.loc[~missing].copy()

    # Per-job parameters
    df_jobs2["system_version"] = df_jobs2["system_version"].astype(float)
    df_jobs2["unit_scale"] = df_jobs2["conversion_to_V"].astype(float)

    print("Jobs to run:", len(df_jobs2))
    print(df_jobs2.head())

    # Constants
    brain_regions = {br: 1 for br in intracranial_BIDS_converter.BRAIN_REGIONS}
    monopolar = True
    bipolar = True
    mni = True
    tal = False
    area = False
    root = "/home1/maint/LTP_BIDS/"

    # Map with per-job values (IMPORTANT)
    futures = client.map(
        convert_to_bids,
        df_jobs2["subject"].tolist(),
        df_jobs2["experiment"].tolist(),
        df_jobs2["session"].tolist(),
        df_jobs2["system_version"].tolist(),   # <- per job
        df_jobs2["unit_scale"].tolist(),       # <- per job
        [monopolar] * len(df_jobs2),
        [bipolar] * len(df_jobs2),
        [mni] * len(df_jobs2),
        [tal] * len(df_jobs2),
        [area] * len(df_jobs2),
        [brain_regions] * len(df_jobs2),
        [root] * len(df_jobs2),
    )

#     df_jobs = df_subset[["subject", "experiment", "session"]].copy()

#     brain_regions = {br: 1 for br in intracranial_BIDS_converter.BRAIN_REGIONS}
#     system_version = 4.0
#     unit_scale = float(conversion_df[conversion_df['subject'] == row["subject"]]["conversion_to_V"].iloc[0])
#     monopolar = True
#     bipolar = True
#     mni = True
#     tal = False
#     area = False
#     root = "/home1/maint/LTP_BIDS/"


     
#     futures = client.map(
#         convert_to_bids,
#         df_jobs["subject"].tolist(),
#         df_jobs["experiment"].tolist(),
#         df_jobs["session"].tolist(),
#         [system_version] * len(df_jobs),
#         [unit_scale] * len(df_jobs),
#         [monopolar] * len(df_jobs),
#         [bipolar] * len(df_jobs),
#         [mni] * len(df_jobs),
#         [tal] * len(df_jobs),
#         [area] * len(df_jobs),
#         [brain_regions] * len(df_jobs),
#         [root] * len(df_jobs),
#     )
    # results = client.gather(futures)
    from dask.distributed import as_completed

    for future in as_completed(futures):
        try:
            result = future.result()
            print("✓ finished:", future.key)
        except Exception as e:
            print("✗ failed:", future.key)
            print(e)
