import cmlreaders as cml
import mne
import numpy as np
import pandas as pd
pd.options.display.max_columns=None
pd.options.display.max_rows=200
import os
from scipy.io import loadmat
from ScalpBIDSConverter import *
import cmldask.CMLDask as da

%matplotlib inline

def convert_to_bids(subject, experiment, session):
    # if os.path.exists(f"/data8/PEERS_BIDS/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{experiment}_events.json"):
    #     if (os.path.getmtime(f"/data8/PEERS_BIDS/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{experiment}_events.json") > 1684160000):
    #         return True
    converter = ScalpBIDSConverter(subject, experiment, session, root="/home1/maint/",
                                   overwrite_eeg=True, overwrite_beh=True, )
    return True

if __name__ == "__main__":
    df = cml.get_data_index()
    max_subjects = 10
    experiments = ["ValueCourier", "ltpFR", "ltpFR2", "VFFR"]

    subjects_to_exclude = {"LTP001"}  # <-- your list here

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
    df_jobs = df_subset[["subject", "experiment", "session"]].copy()

    futures = client.map(
        convert_to_bids,
        df_jobs["subject"].tolist(),
        df_jobs["experiment"].tolist(),
        df_jobs["session"].tolist(),
    )
    # results = client.gather(futures)
    from dask.distributed import as_completed

    for future in as_completed(futures):
        try:
            result = future.result()
            print("✓ finished:", future.key)
        except Exception as e:
            print("✗ failed:", future.key)
            print(e)
