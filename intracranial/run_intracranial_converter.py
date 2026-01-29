#!/usr/bin/env python3
import argparse
import os
import sys
import importlib
import pandas as pd
import cmlreaders as cml

# Dask is only needed in parallel mode; import lazily in main()

# --- package path setup ---
sys.path.insert(0, os.path.expanduser("~/bids-convert"))

import intracranial.run_intracranial_convert_maint as rim
importlib.reload(rim)

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


EXPERIMENT_TO_CONVERTER = {
    "catFR1": catFR1_BIDS_converter,
    "catFR2": catFR2_BIDS_converter,
    "FR1": FR1_BIDS_converter,
    "FR2": FR2_BIDS_converter,
    "PAL1": PAL1_BIDS_converter,
    "PAL2": PAL2_BIDS_converter,
    "pyFR": pyFR_BIDS_converter,
    "RepFR1": RepFR1_BIDS_converter,
    "YC1": YC1_BIDS_converter,
    "YC2": YC2_BIDS_converter,
}


def lookup_conversion_params(conversion_df: pd.DataFrame, subject: str, experiment: str, session: int):
    session = int(session)
    rows = conversion_df.loc[
        (conversion_df["subject"] == subject)
        & (conversion_df["experiment"] == experiment)
        & (conversion_df["session"] == session)
    ]
    if rows.empty:
        return None
    r0 = rows.iloc[0]
    return float(r0["system_version"]), float(r0["conversion_to_V"])


def convert_one_job(
    subject: str,
    experiment: str,
    session: int,
    system_version: float,
    unit_scale: float,
    *,
    monopolar: bool,
    bipolar: bool,
    mni: bool,
    tal: bool,
    area: bool,
    brain_regions: dict,
    root: str,
):
    # IMPORTANT: don't overwrite root here
    Converter = EXPERIMENT_TO_CONVERTER.get(experiment)
    if Converter is None:
        raise ValueError(f"Unknown experiment '{experiment}'. Expected one of {sorted(EXPERIMENT_TO_CONVERTER)}")

    converter = Converter(
        subject, experiment, session,
        system_version, unit_scale,
        monopolar, bipolar, mni, tal, area,
        brain_regions,
        root=root,
    )
    return converter.run()


def build_jobs_parallel(
    *,
    experiments: list[str],
    max_subjects: int,
    subjects_to_exclude: set[str],
    conversion_df: pd.DataFrame,
):
    # Pull candidate rows from data index
    df = cml.get_data_index()
    df = df[df["experiment"].isin(experiments)].copy()
    if subjects_to_exclude:
        df = df[~df["subject"].isin(subjects_to_exclude)].copy()

    dfs = []
    for exp in experiments:
        df_this = df[df["experiment"] == exp]
        subjects = (
            df_this["subject"]
            .drop_duplicates()
            .sort_values()
            .head(max_subjects)
        )
        dfs.append(df_this[df_this["subject"].isin(subjects)].copy())

    df_subset = pd.concat(dfs, ignore_index=True)
    df_jobs = df_subset[["subject", "experiment", "session"]].copy()

    # Normalize dtypes for merge
    df_jobs["session"] = df_jobs["session"].astype(int)
    conversion_df["session"] = conversion_df["session"].astype(int)

    merge_keys = ["subject", "experiment", "session"]
    df_jobs2 = df_jobs.merge(
        conversion_df[merge_keys + ["system_version", "conversion_to_V"]],
        on=merge_keys,
        how="left",
    )

    missing = df_jobs2["system_version"].isna() | df_jobs2["conversion_to_V"].isna()
    if missing.any():
        print(f"Skipping {missing.sum()} job(s) with no matching conversion row.")
        print(df_jobs2.loc[missing, merge_keys].to_string(index=False))

    df_jobs2 = df_jobs2.loc[~missing].copy()
    df_jobs2["system_version"] = df_jobs2["system_version"].astype(float)
    df_jobs2["unit_scale"] = df_jobs2["conversion_to_V"].astype(float)

    return df_jobs2


def main():
    ap = argparse.ArgumentParser(description="Intracranial BIDS conversion (single or parallel).")
    ap.add_argument("--mode", choices=["single", "parallel"], required=True)

    # Conversion lookup
    ap.add_argument("--conversion-csv", default="system_1_unit_conversions.csv")
    ap.add_argument("--root", default="/home1/maint/LTP_BIDS/")

    # Flags shared by both modes
    ap.add_argument("--monopolar", action="store_true", default=True)
    ap.add_argument("--no-monopolar", dest="monopolar", action="store_false")
    ap.add_argument("--bipolar", action="store_true", default=True)
    ap.add_argument("--no-bipolar", dest="bipolar", action="store_false")
    ap.add_argument("--mni", action="store_true", default=True)
    ap.add_argument("--no-mni", dest="mni", action="store_false")
    ap.add_argument("--tal", action="store_true", default=False)
    ap.add_argument("--area", action="store_true", default=False)

    # Single mode args
    ap.add_argument("--subject")
    ap.add_argument("--experiment", choices=sorted(EXPERIMENT_TO_CONVERTER.keys()))
    ap.add_argument("--session", type=int)

    # Parallel mode args
    ap.add_argument("--experiments", nargs="*", default=list(EXPERIMENT_TO_CONVERTER.keys()))
    ap.add_argument("--max-subjects", type=int, default=10)
    ap.add_argument("--exclude-subjects", nargs="*", default=["LTP001"])

    ap.add_argument("--job-name", default="bids_convert")
    ap.add_argument("--memory-per-job", default="50GB")
    ap.add_argument("--max-n-jobs", type=int, default=10)
    ap.add_argument("--threads-per-job", type=int, default=1)
    ap.add_argument("--adapt", action="store_true", default=True)
    ap.add_argument("--no-adapt", dest="adapt", action="store_false")
    ap.add_argument("--log-directory", default="~/logs/")

    args = ap.parse_args()

    conversion_df = pd.read_csv(args.conversion_csv)
    conversion_df["session"] = conversion_df["session"].astype(int)

    brain_regions = {br: 1 for br in intracranial_BIDS_converter.BRAIN_REGIONS}

    if args.mode == "single":
        # Validate required single args
        if args.subject is None or args.experiment is None or args.session is None:
            ap.error("--mode single requires --subject, --experiment, and --session")

        params = lookup_conversion_params(conversion_df, args.subject, args.experiment, args.session)
        if params is None:
            print(f"SKIP: no conversion row for (subject={args.subject}, experiment={args.experiment}, session={args.session})")
            sys.exit(2)

        system_version, unit_scale = params

        print(
            f"Running SINGLE conversion:\n"
            f"  subject={args.subject}\n"
            f"  experiment={args.experiment}\n"
            f"  session={args.session}\n"
            f"  system_version={system_version}\n"
            f"  unit_scale={unit_scale}\n"
        )

        ok = convert_one_job(
            args.subject, args.experiment, args.session,
            system_version, unit_scale,
            monopolar=args.monopolar, bipolar=args.bipolar, mni=args.mni,
            tal=args.tal, area=args.area,
            brain_regions=brain_regions,
            root=args.root,
        )
        sys.exit(0 if ok else 1)

    # ---- parallel mode ----
    import cmldask.CMLDask as da
    from dask.distributed import as_completed

    df_jobs2 = build_jobs_parallel(
        experiments=args.experiments,
        max_subjects=args.max_subjects,
        subjects_to_exclude=set(args.exclude_subjects),
        conversion_df=conversion_df,
    )

    print("Jobs to run:", len(df_jobs2))
    print(df_jobs2.head())

    client = da.new_dask_client_slurm(
        job_name=args.job_name,
        memory_per_job=args.memory_per_job,
        max_n_jobs=args.max_n_jobs,
        threads_per_job=args.threads_per_job,
        adapt=args.adapt,
        log_directory=args.log_directory,
    )

    futures = client.map(
        lambda s, e, sess, sv, us: convert_one_job(
            s, e, sess, sv, us,
            monopolar=args.monopolar, bipolar=args.bipolar, mni=args.mni,
            tal=args.tal, area=args.area,
            brain_regions=brain_regions,
            root=args.root,
        ),
        df_jobs2["subject"].tolist(),
        df_jobs2["experiment"].tolist(),
        df_jobs2["session"].tolist(),
        df_jobs2["system_version"].tolist(),
        df_jobs2["unit_scale"].tolist(),
    )

    n_ok = 0
    n_fail = 0
    for fut in as_completed(futures):
        try:
            _ = fut.result()
            n_ok += 1
            print("✓ finished:", fut.key)
        except Exception as e:
            n_fail += 1
            print("✗ failed:", fut.key)
            print(e)

    print(f"Done. ok={n_ok} fail={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()



# import cmlreaders as cml
# import mne
# import numpy as np
# import pandas as pd
# pd.options.display.max_columns=None
# pd.options.display.max_rows=200
# import os
# from scipy.io import loadmat
# # from ScalpBIDSConverter import *
# import cmldask.CMLDask as da
# import os, sys, importlib

# # add the package ROOT, not the intracranial folder
# sys.path.insert(0, os.path.expanduser("~/bids-convert"))

# import intracranial.run_intracranial_convert_maint as rim
# importlib.reload(rim)  # if you edited it recently

# # now this should work because relative imports have a parent package

# from intracranial.intracranial_BIDS_converter import intracranial_BIDS_converter

# from intracranial.catFR1.catFR1_BIDS_converter import catFR1_BIDS_converter
# from intracranial.catFR2.catFR2_BIDS_converter import catFR2_BIDS_converter
# from intracranial.FR1.FR1_BIDS_converter import FR1_BIDS_converter
# from intracranial.FR2.FR2_BIDS_converter import FR2_BIDS_converter
# from intracranial.PAL1.PAL1_BIDS_converter import PAL1_BIDS_converter
# from intracranial.PAL2.PAL2_BIDS_converter import PAL2_BIDS_converter
# from intracranial.pyFR.pyFR_BIDS_converter import pyFR_BIDS_converter
# from intracranial.RepFR1.RepFR1_BIDS_converter import RepFR1_BIDS_converter
# from intracranial.YC1.YC1_BIDS_converter import YC1_BIDS_converter
# from intracranial.YC2.YC2_BIDS_converter import YC2_BIDS_converter

# # # import sys
# # # sys.path.insert(0, "/ABS/PATH/to/bids-convert/intracranial")

# # %matplotlib inline

# def convert_to_bids(subject, experiment, session,system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root="/home1/maint/LTP_BIDS/"):
#     # if os.path.exists(f"/data8/PEERS_BIDS/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{experiment}_events.json"):
#     #     if (os.path.getmtime(f"/data8/PEERS_BIDS/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{experiment}_events.json") > 1684160000):
#     #         return True
#     root = '/home1/maint/LTP_BIDS/'
#     if experiment == "catFR1":
#         converter = catFR1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "catFR2":
#         converter = catFR2_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "FR1":
#         converter = FR1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "FR2":
#         converter = FR2_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "PAL1":
#         converter = PAL1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "PAL2":
#         converter = PAL2_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "pyFR":
#         converter = pyFR_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "RepFR1":
#         converter = RepFR1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "YC1":
#         converter = YC1_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     elif experiment == "YC2":
#         converter = YC2_BIDS_converter(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root=root)
#     return converter.run()

# if __name__ == "__main__":
#     df = cml.get_data_index()
#     max_subjects = 10
#     experiments = ["catFR1", "catFR2", "FR1", "FR2", "PAL1", "PAL2", "pyFR", "RepFR1", "YC1", "YC2"]

#     subjects_to_exclude = {"LTP001"}  # <-- your list here

#     df = cml.get_data_index()

#     df_exp = df[df["experiment"].isin(experiments)].copy()

#     # remove excluded subjects up front
#     df_exp = df_exp[~df_exp["subject"].isin(subjects_to_exclude)].copy()

#     dfs = []

#     for exp in experiments:
#         df_this = df_exp[df_exp["experiment"] == exp]

#         subjects = (
#             df_this["subject"]
#             .drop_duplicates()
#             .sort_values()      # deterministic
#             .head(max_subjects)
#         )

#         df_keep = df_this[df_this["subject"].isin(subjects)].copy()
#         dfs.append(df_keep)

#     df_subset = pd.concat(dfs, ignore_index=True)
    
#     client = da.new_dask_client_slurm(
#         job_name="bids_convert",
#         memory_per_job="50GB",
#         max_n_jobs=10, threads_per_job=1, 
#         adapt=True,
#         log_directory="~/logs/",
#     )
#     conversion_df = pd.read_csv('system_1_unit_conversions.csv')
#     df_jobs = df_subset[["subject", "experiment", "session"]].copy()

#     # conversion_df = pd.read_csv("system_1_unit_conversions.csv")

#     # Normalize dtypes (session often mismatches int vs str)
#     df_jobs["session"] = df_jobs["session"].astype(int)
#     conversion_df["session"] = conversion_df["session"].astype(int)

#     # Merge on the actual keys
#     merge_keys = ["subject", "experiment", "session"]

#     df_jobs2 = df_jobs.merge(
#         conversion_df[merge_keys + ["system_version", "conversion_to_V"]],
#         on=merge_keys,
#         how="left",
#     )

#     # Drop non-matching rows
#     missing = df_jobs2["system_version"].isna() | df_jobs2["conversion_to_V"].isna()
#     if missing.any():
#         print(f"Skipping {missing.sum()} job(s) with no matching conversion row:")
#         print(df_jobs2.loc[missing, merge_keys].to_string(index=False))

#     df_jobs2 = df_jobs2.loc[~missing].copy()

#     # Per-job parameters
#     df_jobs2["system_version"] = df_jobs2["system_version"].astype(float)
#     df_jobs2["unit_scale"] = df_jobs2["conversion_to_V"].astype(float)

#     print("Jobs to run:", len(df_jobs2))
#     print(df_jobs2.head())

#     # Constants
#     brain_regions = {br: 1 for br in intracranial_BIDS_converter.BRAIN_REGIONS}
#     monopolar = True
#     bipolar = True
#     mni = True
#     tal = False
#     area = False
#     root = "/home1/maint/LTP_BIDS/"

#     # Map with per-job values (IMPORTANT)
#     futures = client.map(
#         convert_to_bids,
#         df_jobs2["subject"].tolist(),
#         df_jobs2["experiment"].tolist(),
#         df_jobs2["session"].tolist(),
#         df_jobs2["system_version"].tolist(),   # <- per job
#         df_jobs2["unit_scale"].tolist(),       # <- per job
#         [monopolar] * len(df_jobs2),
#         [bipolar] * len(df_jobs2),
#         [mni] * len(df_jobs2),
#         [tal] * len(df_jobs2),
#         [area] * len(df_jobs2),
#         [brain_regions] * len(df_jobs2),
#         [root] * len(df_jobs2),
#     )

# #     df_jobs = df_subset[["subject", "experiment", "session"]].copy()

# #     brain_regions = {br: 1 for br in intracranial_BIDS_converter.BRAIN_REGIONS}
# #     system_version = 4.0
# #     unit_scale = float(conversion_df[conversion_df['subject'] == row["subject"]]["conversion_to_V"].iloc[0])
# #     monopolar = True
# #     bipolar = True
# #     mni = True
# #     tal = False
# #     area = False
# #     root = "/home1/maint/LTP_BIDS/"


     
# #     futures = client.map(
# #         convert_to_bids,
# #         df_jobs["subject"].tolist(),
# #         df_jobs["experiment"].tolist(),
# #         df_jobs["session"].tolist(),
# #         [system_version] * len(df_jobs),
# #         [unit_scale] * len(df_jobs),
# #         [monopolar] * len(df_jobs),
# #         [bipolar] * len(df_jobs),
# #         [mni] * len(df_jobs),
# #         [tal] * len(df_jobs),
# #         [area] * len(df_jobs),
# #         [brain_regions] * len(df_jobs),
# #         [root] * len(df_jobs),
# #     )
#     # results = client.gather(futures)
#     from dask.distributed import as_completed

#     for future in as_completed(futures):
#         try:
#             result = future.result()
#             print("✓ finished:", future.key)
#         except Exception as e:
#             print("✗ failed:", future.key)
#             print(e)
