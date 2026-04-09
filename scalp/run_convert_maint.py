#!/usr/bin/env python

import argparse
import os
import cmlreaders as cml
import pandas as pd
from ScalpBIDSConverter import *
import cmldask.CMLDask as da
from dask.distributed import as_completed

pd.options.display.max_columns = None
pd.options.display.max_rows = 200


def bids_session_outputs_exist(root, subject, experiment, session):
    """
    Decide whether this (subject, session, experiment) appears already converted.

    The converter writes BIDS files using the lowercase experiment as the
    task name (see ScalpBIDSConverter.write_bids_beh / write_bids_eeg)
    and may put outputs under either ``beh/`` (behavioral-only sessions)
    or ``eeg/`` (sessions with raw EEG). We treat the presence of any of
    these as evidence the session has been converted. The subject is
    sanitized the same way ScalpBIDSConverter does (BIDS forbids ``_``
    in entity values; we map ``_`` to ``v`` for "visit", e.g.
    ``LTP220_03`` → ``LTP220v03``).
    """
    sanitized_subject = "".join(
        ch if ch.isalnum() else ("v" if ch == "_" else "")
        for ch in str(subject)
    )
    sub = f"sub-{sanitized_subject}"
    ses = f"ses-{session}"
    task = experiment.lower()
    sess_dir = os.path.join(root, sub, ses)

    candidates = [
        # behavioral-only sessions
        os.path.join(sess_dir, "beh", f"{sub}_{ses}_task-{task}_beh.json"),
        os.path.join(sess_dir, "beh", f"{sub}_{ses}_task-{task}_beh.tsv"),
        # full EEG sessions
        os.path.join(sess_dir, "eeg", f"{sub}_{ses}_task-{task}_eeg.json"),
        os.path.join(sess_dir, "eeg", f"{sub}_{ses}_task-{task}_eeg.tsv"),
        os.path.join(sess_dir, "eeg", f"{sub}_{ses}_task-{task}_channels.tsv"),
        os.path.join(sess_dir, "eeg", f"{sub}_{ses}_task-{task}_events.json"),
        os.path.join(sess_dir, "eeg", f"{sub}_{ses}_task-{task}_events.tsv"),
    ]
    return any(os.path.exists(p) for p in candidates)


def convert_to_bids(subject, experiment, session, root, overwrite_eeg, overwrite_beh, skip_if_exists):
    """
    Worker function. If skip_if_exists is True and outputs exist, skip.
    """
    if skip_if_exists and bids_session_outputs_exist(root, subject, experiment, session):
        return f"SKIP existing outputs: {subject} {experiment} {session}"

    converter = ScalpBIDSConverter(
        subject,
        experiment,
        session,
        root=root,
        overwrite_eeg=overwrite_eeg,
        overwrite_beh=overwrite_beh,
    )
    return f"DONE: {subject} {experiment} {session}"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert scalp EEG data to BIDS.")

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["ValueCourier", "ltpFR", "ltpFR2", "VFFR", "VCBehOnly"],
        help="Experiments to run (default: predefined list)",
    )
    
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Single subject to convert",
    )

    parser.add_argument(
        "--session",
        type=int,
        default=None,
        help="Single session to convert",
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

    parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Run sequentially (no Dask / Slurm)",
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/data/LTP_BIDS/",
        help="BIDS root directory (default: /home1/maint/)",
    )

    parser.add_argument(
        "--overwrite-eeg",
        action="store_true",
        default=False,
        help="Overwrite EEG outputs even if they already exist",
    )

    parser.add_argument(
        "--overwrite-beh",
        action="store_true",
        default=False,
        help="Overwrite behavioral outputs even if they already exist",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    experiments = args.experiments
    subjects_to_exclude = set(args.exclude_subjects)
    max_subjects = args.max_subjects
    root = args.root

    # Skip if exists unless user explicitly asked to overwrite something
    skip_if_exists = not (args.overwrite_eeg or args.overwrite_beh)

    print("\nRunning with settings:")
    print("Experiments:", experiments)
    print("Excluded subjects:", subjects_to_exclude)
    print("Max subjects per experiment:", max_subjects)
    print("Sequential mode:", args.sequential)
    print("Root:", root)
    print("Overwrite EEG:", args.overwrite_eeg)
    print("Overwrite BEH:", args.overwrite_beh)
    print("Skip if exists:", skip_if_exists)
    print("--------------------------------------------------\n")

    df = cml.get_data_index()

    df_exp = df[df["experiment"].isin(experiments)].copy()
    df_exp = df_exp[~df_exp["subject"].isin(subjects_to_exclude)].copy()

    dfs = []
    for exp in experiments:
        df_this = df_exp[df_exp["experiment"] == exp]

        subjects = df_this["subject"].drop_duplicates().sort_values()
        if max_subjects is not None:
            subjects = subjects.head(max_subjects)

        df_keep = df_this[df_this["subject"].isin(subjects)].copy()
        dfs.append(df_keep)

    if not dfs:
        raise SystemExit("No jobs found after filtering experiments/subjects.")

    df_subset = pd.concat(dfs, ignore_index=True)
    df_jobs = df_subset[["subject", "experiment", "session"]].copy()
    if args.subject is not None:
        df_jobs = df_jobs[df_jobs["subject"] == args.subject]
    if args.session is not None:
        df_jobs = df_jobs[df_jobs["session"] == args.session]

    # ---------------- SEQUENTIAL MODE ----------------
    if args.sequential:
        print("Running SEQUENTIALLY (no Dask)\n")

        for _, row in df_jobs.iterrows():
            subject, experiment, session = row["subject"], row["experiment"], row["session"]
            try:
                msg = convert_to_bids(
                    subject, experiment, session,
                    root=root + f"/{experiment}/",
                    overwrite_eeg=args.overwrite_eeg,
                    overwrite_beh=args.overwrite_beh,
                    skip_if_exists=skip_if_exists,
                )
                if msg.startswith("SKIP"):
                    print(f"↷ {msg}")
                else:
                    print(f"✓ {msg}")
            except Exception as e:
                print(f"✗ FAILED: {subject} {experiment} {session}")
                print(e)

    # ---------------- PARALLEL MODE ----------------
    else:
        print("Running in PARALLEL via Slurm+Dask\n")

        client = da.new_dask_client_slurm(
            job_name="bids_convert",
            memory_per_job="50GB",
            max_n_jobs=10,
            threads_per_job=1,
            adapt=True,
            log_directory="~/logs/",
        )

        roots = [root + f"/{exp}/" for exp in df_jobs["experiment"].tolist()]

        futures = client.map(
            convert_to_bids,
            df_jobs["subject"].tolist(),
            df_jobs["experiment"].tolist(),
            df_jobs["session"].tolist(),
            roots,
            overwrite_eeg=args.overwrite_eeg,
            overwrite_beh=args.overwrite_beh,
            skip_if_exists=skip_if_exists,
        )

        for future in as_completed(futures):
            try:
                msg = future.result()
                if isinstance(msg, str) and msg.startswith("SKIP"):
                    print(f"↷ {msg}")
                else:
                    print(f"✓ {msg}")
            except Exception as e:
                print("✗ failed:", future.key)
                print(e)