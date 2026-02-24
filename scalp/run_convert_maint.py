import argparse
import cmlreaders as cml
import pandas as pd
import os
from ScalpBIDSConverter import *
import cmldask.CMLDask as da
from dask.distributed import as_completed

pd.options.display.max_columns = None
pd.options.display.max_rows = 200


def convert_to_bids(subject, experiment, session):
    converter = ScalpBIDSConverter(
        subject,
        experiment,
        session,
        root="/home1/maint/",
        overwrite_eeg=True,
        overwrite_beh=True,
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Convert scalp EEG data to BIDS.")

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["ValueCourier", "ltpFR", "ltpFR2", "VFFR", "VCBehOnly"],
        help="Experiments to run (default: predefined list)",
    )

    parser.add_argument(
        "--exclude-subjects",
        nargs="+",
        default=["LTP001", "LTP9000", "LTP9001"],
        help="Subjects to exclude",
    )

    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects per experiment (default: no limit)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    experiments = args.experiments
    subjects_to_exclude = set(args.exclude_subjects)
    max_subjects = args.max_subjects

    print("Running with settings:")
    print("Experiments:", experiments)
    print("Excluded subjects:", subjects_to_exclude)
    print("Max subjects per experiment:", max_subjects)

    df = cml.get_data_index()

    df_exp = df[df["experiment"].isin(experiments)].copy()

    # Remove excluded subjects
    df_exp = df_exp[~df_exp["subject"].isin(subjects_to_exclude)].copy()

    dfs = []

    for exp in experiments:
        df_this = df_exp[df_exp["experiment"] == exp]

        subjects = (
            df_this["subject"]
            .drop_duplicates()
            .sort_values()
        )

        if max_subjects is not None:
            subjects = subjects.head(max_subjects)

        df_keep = df_this[df_this["subject"].isin(subjects)].copy()
        dfs.append(df_keep)

    df_subset = pd.concat(dfs, ignore_index=True)

    client = da.new_dask_client_slurm(
        job_name="bids_convert",
        memory_per_job="50GB",
        max_n_jobs=10,
        threads_per_job=1,
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

    for future in as_completed(futures):
        try:
            result = future.result()
            print("✓ finished:", future.key)
        except Exception as e:
            print("✗ failed:", future.key)
            print(e)