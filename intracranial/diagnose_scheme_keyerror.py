#!/usr/bin/env python
"""Diagnose contact-scheme `KeyError(<int>)` failures across FR1 sessions.

Standalone diagnostic for the bug seen in
``intracranial_BIDS_converter.eeg_mono_to_BIDS()``::

    eeg = self.reader.load_eeg(scheme=self.contacts)
    # → cmlreaders/.../eeg.py:221
    #   channels = [contact_to_index[c] for c in self.scheme["contact"]]
    # → KeyError(32) / KeyError(64) / KeyError(128) / ...

The cmlreaders error means the scheme references contact numbers that
don't exist in the recorded ``contact_to_index``. This script reproduces
**only that failing step** for every FR1 session in the data index, for
both the monopolar (``contacts``) and bipolar (``pairs``) schemes, and
writes a CSV + a human-readable report so we can answer:

  1. Which subjects fail?
  2. Are the failures clustered by testing site (RAM convention: the
     trailing letter of ``R1###X`` is the lab — M=MGH, J=Jefferson,
     P=Penn, T=Temple, E=Emory, D=Dartmouth, etc.)?
  3. Are the same missing contact numbers (32, 64, 128) appearing
     across subjects?

The script does **not** touch the converter and does **not** write any
BIDS files. It is read-only with respect to the data archive.

Usage::

    python diagnose_scheme_keyerror.py                # full FR1 sweep
    python diagnose_scheme_keyerror.py --max-subjects 5
    python diagnose_scheme_keyerror.py --out-dir ~/diagnostics
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

import cmlreaders as cml
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-session probe
# ---------------------------------------------------------------------------

def _missing_contacts(scheme_df: pd.DataFrame, exc: BaseException) -> Tuple[Optional[int], int]:
    """Best-effort extraction of (first_missing, total_missing) from a KeyError.

    cmlreaders raises ``KeyError(<int>)`` from inside a list comprehension,
    so the only thing in the exception is the integer of the first missing
    contact. We can't recover the *full* set of missing contacts without
    the recorded ``contact_to_index`` (which lives inside the EEGReader and
    is gone by the time we catch). Return (first_missing, n_missing=1) so
    the caller at least has the trigger value.
    """
    first = None
    if isinstance(exc, KeyError) and exc.args:
        try:
            first = int(exc.args[0])
        except (TypeError, ValueError):
            first = None
    return first, 1 if first is not None else 0


def _probe_scheme(reader: cml.CMLReader, scheme_kind: str) -> Dict[str, Any]:
    """Try ``reader.load_eeg(scheme=<scheme_kind>)`` and return a status dict.

    Skips the bulk read by passing a tiny single-event window — the cmlreaders
    scheme validation (which is what raises) happens before sample IO.
    """
    out: Dict[str, Any] = {
        f"{scheme_kind}_status": "ok",
        f"{scheme_kind}_n_in_scheme": np.nan,
        f"{scheme_kind}_missing_contact": np.nan,
        f"{scheme_kind}_n_missing": np.nan,
        f"{scheme_kind}_error_repr": "",
    }

    # 1. Load the scheme.
    try:
        scheme_df = reader.load(scheme_kind)
    except Exception as exc:
        out[f"{scheme_kind}_status"] = "scheme_unloadable"
        out[f"{scheme_kind}_error_repr"] = repr(exc)[:300]
        return out

    out[f"{scheme_kind}_n_in_scheme"] = int(len(scheme_df))

    # 2. Need an event for the tiny load. Skip cleanly if events are missing.
    try:
        events = reader.load("events")
    except Exception as exc:
        out[f"{scheme_kind}_status"] = "events_unloadable"
        out[f"{scheme_kind}_error_repr"] = repr(exc)[:300]
        return out
    if len(events) == 0:
        out[f"{scheme_kind}_status"] = "events_empty"
        return out

    # 3. The actual probe — load 50ms around the first event with the scheme.
    try:
        reader.load_eeg(
            events=events.iloc[[0]],
            rel_start=0,
            rel_stop=50,
            scheme=scheme_df,
        )
    except KeyError as exc:
        first, n = _missing_contacts(scheme_df, exc)
        out[f"{scheme_kind}_status"] = "key_error"
        out[f"{scheme_kind}_missing_contact"] = first
        out[f"{scheme_kind}_n_missing"] = n
        out[f"{scheme_kind}_error_repr"] = repr(exc)[:300]
    except Exception as exc:
        out[f"{scheme_kind}_status"] = "other_error"
        out[f"{scheme_kind}_error_repr"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def probe_session(subject: str, experiment: str, session: int) -> Dict[str, Any]:
    """Run both contact + pair scheme probes for one session."""
    row: Dict[str, Any] = {
        "subject": subject,
        "subject_site": subject[-1] if subject else "",
        "experiment": experiment,
        "session": int(session),
    }
    try:
        reader = cml.CMLReader(subject=subject, experiment=experiment, session=int(session))
    except Exception as exc:
        row["contacts_status"] = "reader_error"
        row["pairs_status"] = "reader_error"
        row["contacts_error_repr"] = repr(exc)[:300]
        row["pairs_error_repr"] = repr(exc)[:300]
        return row

    row.update(_probe_scheme(reader, "contacts"))
    row.update(_probe_scheme(reader, "pairs"))
    return row


# ---------------------------------------------------------------------------
# Sweep + report
# ---------------------------------------------------------------------------

def build_jobs(experiment: str, max_subjects: Optional[int]) -> pd.DataFrame:
    df = cml.get_data_index()
    df = df[df["experiment"] == experiment].copy()
    df["session"] = df["session"].astype(int)
    if max_subjects:
        keep = df["subject"].drop_duplicates().sort_values().head(max_subjects)
        df = df[df["subject"].isin(keep)]
    return df[["subject", "experiment", "session"]].reset_index(drop=True)


def _print_and_capture(lines, msg=""):
    print(msg)
    lines.append(msg)


def make_report(results: pd.DataFrame) -> str:
    lines: list[str] = []

    def emit(msg=""):
        _print_and_capture(lines, msg)

    n = len(results)
    emit(f"Total sessions probed: {n}")
    emit()

    for kind in ("contacts", "pairs"):
        col = f"{kind}_status"
        if col not in results.columns:
            continue
        emit(f"=== {kind} scheme status ===")
        emit(results[col].value_counts(dropna=False).to_string())
        emit()

        ke = results[results[col] == "key_error"]
        if ke.empty:
            emit(f"  no key_error rows for {kind}")
            emit()
            continue

        emit(f"--- {kind}: key_error by site letter ---")
        site_summary = (
            ke.groupby("subject_site")
            .agg(
                n_key_error=("subject", "size"),
                n_subjects=("subject", "nunique"),
            )
            .sort_values("n_key_error", ascending=False)
        )
        # Add total sessions per site so we can compute %
        all_by_site = results.groupby("subject_site")["subject"].size()
        site_summary["n_total"] = site_summary.index.map(all_by_site)
        site_summary["pct"] = (
            site_summary["n_key_error"] / site_summary["n_total"] * 100
        ).round(1)
        emit(site_summary.to_string())
        emit()

        emit(f"--- {kind}: missing contact value counts ---")
        emit(
            ke[f"{kind}_missing_contact"]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )
        emit()

        emit(f"--- {kind}: subjects per site (only key_error) ---")
        per_site = (
            ke.groupby("subject_site")["subject"]
            .agg(lambda s: ", ".join(sorted(set(s))))
        )
        for site, subs in per_site.items():
            emit(f"  {site}: {subs}")
        emit()

        emit(f"--- {kind}: failing sessions per site (subject ses-N) ---")
        ke_sorted = ke.sort_values(["subject_site", "subject", "session"])
        for site, group in ke_sorted.groupby("subject_site"):
            pairs = ", ".join(
                f"{row.subject} ses-{int(row.session)}"
                for row in group.itertuples(index=False)
            )
            emit(f"  {site} ({len(group)}): {pairs}")
        emit()

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default="FR1")
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument(
        "--out-dir",
        default=os.path.expanduser("~/diagnostics"),
        help="Where to write contact_scheme_keyerror.csv and the report",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after probing N sessions (debug)",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    jobs = build_jobs(args.experiment, args.max_subjects)
    if args.limit:
        jobs = jobs.head(args.limit)
    print(f"Probing {len(jobs)} sessions of {args.experiment}")

    rows: list[dict] = []
    for i, job in enumerate(jobs.itertuples(index=False), start=1):
        try:
            row = probe_session(job.subject, job.experiment, job.session)
        except Exception as exc:
            row = {
                "subject": job.subject,
                "subject_site": job.subject[-1] if job.subject else "",
                "experiment": job.experiment,
                "session": int(job.session),
                "contacts_status": "probe_crash",
                "pairs_status": "probe_crash",
                "contacts_error_repr": repr(exc)[:300],
                "pairs_error_repr": repr(exc)[:300],
            }
            traceback.print_exc()
        rows.append(row)
        # Light progress so the user can see life on long runs.
        if i % 10 == 0 or i == len(jobs):
            print(
                f"  [{i:4d}/{len(jobs)}] {job.subject} ses-{job.session} "
                f"contacts={row.get('contacts_status')} "
                f"pairs={row.get('pairs_status')}"
            )

    results = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "contact_scheme_keyerror.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    report = make_report(results)
    report_path = os.path.join(args.out_dir, "contact_scheme_keyerror_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
