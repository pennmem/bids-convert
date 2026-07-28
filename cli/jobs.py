"""Build the (subject, experiment, session) job list for a conversion run.

One implementation for both modalities. Scalp jobs are just the triple;
intracranial jobs additionally carry the per-session ``system_version`` and
``unit_scale`` needed by the converter.
"""

from __future__ import annotations

import json
import os

import cmlreaders as cml
import pandas as pd

from . import registry

BASE_COLUMNS = ["subject", "experiment", "session"]
INTRACRANIAL_COLUMNS = BASE_COLUMNS + ["system_version", "unit_scale"]

DEFAULT_UNIT_SCALE = 1e6  # 1 µV, used when a session is absent from the CSV


def parse_sessions(spec_list: list[str], available_sessions: list[int]) -> list[int]:
    """Parse session specifiers into a list of session ints.

    Each specifier can be:
      - A single int:   "3"   -> [3]
      - A slice string: "0:5" -> sessions 0..4, ":3" -> 0..2, "2:" -> 2..max

    Slice semantics follow Python slicing applied to the sorted list of
    available sessions.
    """
    available = sorted(available_sessions)
    result = set()
    for spec in spec_list:
        if ":" in spec:
            parts = spec.split(":", 1)
            start = int(parts[0]) if parts[0] else None
            stop = int(parts[1]) if parts[1] else None
            result.update(available[start:stop])
        else:
            result.add(int(spec))
    return sorted(result)


def load_recent_pairs(path: str) -> set[tuple[str, int]]:
    """Read a ``recently_modified.json`` ({subject: [sessions]}) into pairs."""
    with open(path) as f:
        recent = json.load(f)
    return {
        (str(subject), int(session))
        for subject, sessions in recent.items()
        for session in (sessions if isinstance(sessions, (list, tuple)) else [sessions])
    }


def _empty(columns):
    return pd.DataFrame(columns=columns)


def build_jobs(
    *,
    modality: str,
    experiments: list[str] | None = None,
    subjects: list[str] | None = None,
    sessions_spec: list[str] | None = None,
    exclude_subjects=(),
    smokescreen: bool = False,
    recently_modified: str | None = None,
    conversion_csv: str | None = None,
) -> pd.DataFrame:
    """Return the job table for this run, filtered by every selection flag."""
    columns = INTRACRANIAL_COLUMNS if modality == registry.INTRACRANIAL else BASE_COLUMNS

    df = cml.get_data_index()
    df = df.copy()
    df["session"] = df["session"].astype(int)

    experiments = experiments or registry.experiments_for(modality)
    df = df[df["experiment"].isin(experiments)].copy()

    if subjects:
        df = df[df["subject"].isin(subjects)].copy()

    if exclude_subjects:
        df = df[~df["subject"].isin(set(exclude_subjects))].copy()

    if df.empty:
        return _empty(columns)

    # Quick test mode: one subject per experiment.
    if smokescreen:
        keep = []
        for exp in df["experiment"].unique():
            df_this = df[df["experiment"] == exp]
            chosen = df_this["subject"].drop_duplicates().sort_values().head(1)
            keep.append(df_this[df_this["subject"].isin(chosen)])
        df = pd.concat(keep, ignore_index=True)

    if sessions_spec is not None:
        rows = []
        for (subject, exp), group in df.groupby(["subject", "experiment"]):
            available = sorted(group["session"].tolist())
            for session in parse_sessions(sessions_spec, available):
                if session in available:
                    rows.append(group[group["session"] == session])
                else:
                    print(f"WARNING: session {session} does not exist for {subject}/{exp}, skipping.")
        if not rows:
            return _empty(columns)
        df = pd.concat(rows, ignore_index=True)

    if recently_modified is not None:
        pairs = load_recent_pairs(recently_modified)
        df = df[
            df.apply(lambda r: (str(r["subject"]), int(r["session"])) in pairs, axis=1)
        ].copy()
        print(
            f"Filtered to {len(df)} job(s) from {recently_modified} "
            f"({len(pairs)} pair(s) listed)."
        )

    if df.empty:
        return _empty(columns)

    if modality != registry.INTRACRANIAL:
        return df[BASE_COLUMNS].reset_index(drop=True)

    return _attach_intracranial_params(df, conversion_csv)


def _attach_intracranial_params(df: pd.DataFrame, conversion_csv: str | None) -> pd.DataFrame:
    """Attach system_version (from the data index) and unit_scale (from CSV)."""
    jobs = df[BASE_COLUMNS + ["system_version"]].copy()
    jobs["system_version"] = jobs["system_version"].astype(float)

    if conversion_csv and os.path.exists(conversion_csv):
        conversion_df = pd.read_csv(conversion_csv)
        conversion_df["session"] = conversion_df["session"].astype(int)
        jobs = jobs.merge(
            conversion_df[BASE_COLUMNS + ["conversion_to_V"]],
            on=BASE_COLUMNS,
            how="left",
        )
    else:
        if conversion_csv:
            print(f"NOTE: conversion CSV not found at {conversion_csv} — "
                  f"using default unit_scale={DEFAULT_UNIT_SCALE:g} for all jobs.")
        jobs["conversion_to_V"] = pd.NA

    missing = jobs["conversion_to_V"].isna()
    if missing.any():
        print(
            f"NOTE: {int(missing.sum())} job(s) not found in the conversion CSV — "
            f"using default unit_scale={DEFAULT_UNIT_SCALE:g} (1 µV)."
        )
        jobs.loc[missing, "conversion_to_V"] = DEFAULT_UNIT_SCALE

    jobs["unit_scale"] = jobs["conversion_to_V"].astype(float)
    return jobs[INTRACRANIAL_COLUMNS].reset_index(drop=True)
