#!/usr/bin/env python3
import argparse
import os
import sys
import importlib
import importlib.util
import pandas as pd
import cmlreaders as cml

# --- package path setup ---
sys.path.insert(0, os.path.expanduser("~/bids-convert"))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from intracranial.intracranial_BIDS_converter import intracranial_BIDS_converter
from conversion_error_log import ConversionErrorLog, cmlreader_involved
from bids_validation import (
    run_bids_validator,
    session_log_dir,
    session_tag,
    tee_to_file,
    validate_jobs,
    write_session_heartbeat_status,
)
sys.path.append("/home1/zrentala/bidsreader")
from bidsreader import CMLBIDSReader

# Converters are imported lazily (on first use) so that class-level operations
# in each converter (e.g. loading wordpool files) only run when needed.
_EXPERIMENT_MODULES = {
    "catFR1":  ("intracranial.catFR1.catFR1_BIDS_converter",   "catFR1_BIDS_converter"),
    "catFR2":  ("intracranial.catFR2.catFR2_BIDS_converter",   "catFR2_BIDS_converter"),
    "FR1":     ("intracranial.FR1.FR1_BIDS_converter",         "FR1_BIDS_converter"),
    "FR2":     ("intracranial.FR2.FR2_BIDS_converter",         "FR2_BIDS_converter"),
    "PAL1":    ("intracranial.PAL1.PAL1_BIDS_converter",       "PAL1_BIDS_converter"),
    "PAL2":    ("intracranial.PAL2.PAL2_BIDS_converter",       "PAL2_BIDS_converter"),
    "pyFR":    ("intracranial.pyFR.pyFR_BIDS_converter",       "pyFR_BIDS_converter"),
    "RepFR1":  ("intracranial.RepFR1.RepFR1_BIDS_converter",   "RepFR1_BIDS_converter"),
    "YC1":     ("intracranial.YC1.YC1_BIDS_converter",         "YC1_BIDS_converter"),
    "YC2":     ("intracranial.YC2.YC2_BIDS_converter",         "YC2_BIDS_converter"),
    "PS2":     ("intracranial.PS2.PS2_BIDS_converter",         "PS2_BIDS_converter"),
    # PS2.1 folder name contains a dot so it cannot be imported via importlib.import_module;
    # store the absolute file path instead and load via spec_from_file_location.
    "PS2.1":   (os.path.join(_SCRIPT_DIR, "PS2.1", "PS2.1_BIDS_converter.py"), "PS21_BIDS_converter"),
}


def _get_converter(experiment: str):
    if experiment not in _EXPERIMENT_MODULES:
        raise ValueError(f"Unknown experiment '{experiment}'. Expected one of {sorted(_EXPERIMENT_MODULES)}")
    module_path, class_name = _EXPERIMENT_MODULES[experiment]
    if module_path.endswith(".py"):
        # File-based loading for modules whose directory name cannot be used as a
        # Python identifier (e.g. "PS2.1" contains a dot).
        spec = importlib.util.spec_from_file_location(class_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    return getattr(module, class_name)


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


STAGES = ('behavioral', 'electrodes', 'mono-eeg', 'bi-eeg', 'mono-channels', 'bi-channels')


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


def convert_one_job(
    subject: str,
    experiment: str,
    session: int,
    system_version: float,
    unit_scale: float,
    *,
    brain_regions: dict,
    root: str,
    overrides: dict | None = None,
):
    """Run one (subject, experiment, session) job and return a result dict.

    Returns a dict with keys: subject, experiment, session, files_written,
    files_not_written, any_failure, error_stage, error_type, error_message,
    cmlreader_failure. The orchestrator consumes this to drive the per-task
    error CSV. The dict is picklable so it round-trips through dask workers.
    """
    Converter = _get_converter(experiment)

    if experiment == "pyFR":
        # pyFR has its own constructor signature: (subject, experiment, session,
        # montage, math_events, system_version, unit_scale,
        # monopolar, bipolar, mni, tal, overrides=None, root=...).
        # Defaults match run_intracranial_convert_maint.py.
        converter = Converter(
            subject, experiment, session,
            0,               # montage
            False,           # math_events
            system_version, unit_scale,
            True,            # monopolar
            True,            # bipolar
            True,            # mni
            False,           # tal
            overrides=overrides or {},
            root=root,
        )
    else:
        converter = Converter(
            subject, experiment, session,
            system_version, unit_scale,
            False,           # area
            brain_regions,
            overrides=overrides or {},
            root=root,
        )

    exc = None
    try:
        converter.run()
    except Exception as e:
        exc = e
        # first_error_stage is already set by _mark_stage if the failure was
        # inside a tracked stage; fill it in if the exception came from
        # elsewhere (e.g. load_contacts, eeg_metadata).
        if not getattr(converter, 'first_error_stage', None):
            converter.first_error_stage = 'run'
            converter.first_exception = e
        import traceback
        traceback.print_exc()

    report = converter.stage_report()
    error_stage = report['error_stage']
    first_exc = report['exception'] or exc
    if first_exc is not None:
        error_type = type(first_exc).__name__
        error_message = " ".join(str(first_exc).splitlines()).strip()
        cml = cmlreader_involved(first_exc)
    else:
        error_type = ""
        error_message = ""
        cml = False
    hb_status = getattr(converter, 'heartbeat_status', None)
    return {
        'subject': str(subject),
        'experiment': experiment,
        'session': int(session),
        'files_written': report['files_written'],
        'files_not_written': report['files_not_written'],
        'any_failure': report['any_failure'] or exc is not None,
        'error_stage': error_stage or ('run' if exc is not None else ''),
        'error_type': error_type,
        'error_message': error_message,
        'cmlreader_failure': cml,
        'raised': exc is not None,
        'heartbeat_status': (hb_status or {}).get('status', 'unknown'),
        'heartbeat_applied': bool((hb_status or {}).get('applied', False)),
    }


def build_jobs(
    *,
    subjects: list[str] | None,
    experiments: list[str] | None,
    sessions_spec: list[str] | None,
    max_subjects: int,
    subjects_to_exclude: set[str],
    conversion_df: pd.DataFrame,
):
    df = cml.get_data_index()
    df["session"] = df["session"].astype(int)

    # Filter by experiments
    if experiments:
        df = df[df["experiment"].isin(experiments)].copy()
    else:
        df = df[df["experiment"].isin(_EXPERIMENT_MODULES.keys())].copy()

    # Filter by subjects
    if subjects:
        df = df[df["subject"].isin(subjects)].copy()

    # Exclude subjects
    if subjects_to_exclude:
        df = df[~df["subject"].isin(subjects_to_exclude)].copy()

    # Apply max_subjects per experiment
    dfs = []
    for exp in df["experiment"].unique():
        df_this = df[df["experiment"] == exp]
        exp_subjects = (
            df_this["subject"]
            .drop_duplicates()
            .sort_values()
            .head(max_subjects)
        )
        dfs.append(df_this[df_this["subject"].isin(exp_subjects)].copy())

    if not dfs:
        return pd.DataFrame(columns=["subject", "experiment", "session", "system_version", "unit_scale"])
    df = pd.concat(dfs, ignore_index=True)

    # Filter by sessions if specified
    if sessions_spec is not None:
        filtered_rows = []
        for (subj, exp), group in df.groupby(["subject", "experiment"]):
            available = sorted(group["session"].tolist())
            requested = parse_sessions(sessions_spec, available)
            for ses in requested:
                if ses in available:
                    filtered_rows.append(group[group["session"] == ses])
                else:
                    print(f"WARNING: session {ses} does not exist for {subj}/{exp}, skipping.")
        if filtered_rows:
            df = pd.concat(filtered_rows, ignore_index=True)
        else:
            return pd.DataFrame(columns=["subject", "experiment", "session", "system_version", "unit_scale"])

    df_jobs = df[["subject", "experiment", "session", "system_version"]].copy()
    df_jobs["session"] = df_jobs["session"].astype(int)
    # system_version from the data index (always available)
    df_jobs["system_version"] = df_jobs["system_version"].astype(float)
    conversion_df["session"] = conversion_df["session"].astype(int)

    merge_keys = ["subject", "experiment", "session"]
    df_jobs2 = df_jobs.merge(
        conversion_df[merge_keys + ["conversion_to_V"]],
        on=merge_keys,
        how="left",
    )

    # For sessions not in the CSV, use the default unit_scale (1e6)
    missing = df_jobs2["conversion_to_V"].isna()
    if missing.any():
        print(
            f"NOTE: {missing.sum()} job(s) not found in the conversion CSV — "
            f"using default unit_scale=1e6 (1 µV)."
        )
        df_jobs2.loc[missing, "conversion_to_V"] = 1e6

    df_jobs2["unit_scale"] = df_jobs2["conversion_to_V"].astype(float)

    return df_jobs2


# Top-level function for Dask mapping (avoid lambda/pickle weirdness)
def run_job(
    subject, experiment, session, system_version, unit_scale,
    brain_regions, root, overrides,
):
    import sys, os
    p = os.path.expanduser("~/bids-convert")
    if p not in sys.path:
        sys.path.insert(0, p)
    from bids_validation import (
        session_log_dir, session_tag, tee_to_file, write_session_heartbeat_status,
    )
    log_dir = session_log_dir(experiment, subject, int(session))
    tag = session_tag(subject, experiment, int(session))
    log_path = os.path.join(log_dir, f"{tag}_bids_convert_log.txt")
    with tee_to_file(log_path, mode="w"):
        result = convert_one_job(
            subject, experiment, int(session), float(system_version), float(unit_scale),
            brain_regions=brain_regions,
            root=root,
            overrides=overrides,
        )
    write_session_heartbeat_status(log_dir, tag, result)
    return result

def _run_convert_with_tee(subject, experiment, session, **kwargs):
    """Run convert_one_job while teeing its stdout/stderr into a per-session log."""
    log_dir = session_log_dir(experiment, subject, int(session))
    tag = session_tag(subject, experiment, int(session))
    log_path = os.path.join(log_dir, f"{tag}_bids_convert_log.txt")
    with tee_to_file(log_path, mode="w"):
        result = convert_one_job(subject, experiment, int(session), **kwargs)
    write_session_heartbeat_status(log_dir, tag, result)
    return result


def validate_bids(args, df_jobs, error_logs):
    """Run per-session eeg-validation pipelines, then dataset-wide BIDS Validator."""
    return validate_jobs(
        df_jobs,
        bids_root_for_job=lambda row: args.root,
        error_logs=error_logs,
        intracranial=True,
        log_root_per_experiment=False,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Intracranial BIDS conversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Convert one subject, one experiment, one session (serial)
  %(prog)s --subjects R1001P --experiments FR1 --sessions 0 --serial

  # Convert sessions 0-4 for two subjects
  %(prog)s --subjects R1001P R1002P --experiments FR1 --sessions 0:5 --serial

  # Convert all sessions for all FR1 subjects (parallel, default)
  %(prog)s --experiments FR1

  # Convert everything for one subject across all experiments
  %(prog)s --subjects R1001P --serial

  # Quick smokescreen test (1 subject per experiment)
  %(prog)s --smokescreen --serial
""",
    )

    ap.add_argument("--conversion-csv", default=os.path.join(_SCRIPT_DIR, "system_1_unit_conversions.csv"))
    ap.add_argument("--root", default="/home1/maint/LTP_BIDS/")

    # Selection
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Subject IDs to convert. If omitted, all subjects are included.")
    ap.add_argument("--experiments", nargs="+", default=None,
                    choices=sorted(_EXPERIMENT_MODULES.keys()),
                    help="Experiments to convert. If omitted, all experiments are included.")
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="Session specifiers: int (e.g. 3) or slice (e.g. 0:5, :3, 2:). "
                         "Requires --subjects or --experiments.")

    # Per-stage override flags. Default: each stage runs only when its
    # outputs are missing on disk. Pass --override-<stage> to force a
    # re-run regardless of existing files. Mono vs bipolar acquisitions
    # are auto-detected from system_version inside the converter.
    for stage in STAGES:
        ap.add_argument(f"--override-{stage}", action="store_true", default=False,
                        help=f"Force re-conversion of the '{stage}' stage even if outputs exist.")

    # Behavior flags
    ap.add_argument("--serial", action="store_true", default=False,
                    help="Run jobs sequentially instead of parallel (Dask).")
    ap.add_argument("--validate", action="store_true", default=False,
                    help="Run BIDS validation on the output directory after conversion.")
    ap.add_argument("--validate-only", action="store_true", default=False,
                    help="Skip conversion and only run BIDS validation on --root.")

    # Job filtering
    ap.add_argument("--max-subjects", type=int, default=10,
                    help="Max subjects per experiment (default: 10).")
    ap.add_argument("--exclude-subjects", nargs="*", default=["LTP001"],
                    help="Subjects to exclude.")
    ap.add_argument("--smokescreen", action="store_true", default=False,
                    help="Quick test: limit to 1 subject per experiment.")

    # Parallel (Dask) args
    ap.add_argument("--job-name", default="bids_convert")
    ap.add_argument("--memory-per-job", default="50GB")
    ap.add_argument("--max-n-jobs", type=int, default=10)
    ap.add_argument("--threads-per-job", type=int, default=1)
    ap.add_argument("--adapt", action="store_true", default=True)
    ap.add_argument("--no-adapt", dest="adapt", action="store_false")
    ap.add_argument("--log-directory", default="~/logs/")

    args = ap.parse_args()

    # --- Validate argument combinations ---
    if args.sessions is not None and args.subjects is None and args.experiments is None:
        ap.error("--sessions requires at least --subjects or --experiments to be specified.")

    if args.validate_only:
        # Build the same job set that conversion would have iterated over,
        # so per-session pipelines can run before the dataset-wide validator.
        conversion_df = pd.read_csv(args.conversion_csv)
        conversion_df["session"] = conversion_df["session"].astype(int)
        max_subjects = 1 if args.smokescreen else args.max_subjects
        df_jobs = build_jobs(
            subjects=args.subjects,
            experiments=args.experiments,
            sessions_spec=args.sessions,
            max_subjects=max_subjects,
            subjects_to_exclude=set(args.exclude_subjects or []),
            conversion_df=conversion_df,
        )
        error_logs: dict[str, ConversionErrorLog] = {}
        for exp in df_jobs["experiment"].unique():
            error_logs[exp] = ConversionErrorLog(args.root, exp)
        valid = validate_bids(args, df_jobs, error_logs)
        sys.exit(0 if valid else 1)

    conversion_df = pd.read_csv(args.conversion_csv)
    conversion_df["session"] = conversion_df["session"].astype(int)

    brain_regions = {br: 1 for br in intracranial_BIDS_converter.BRAIN_REGIONS}

    # Build the per-stage overrides dict from the parsed args
    # (argparse converts hyphens to underscores in the attribute name).
    overrides = {stage: getattr(args, f"override_{stage.replace('-', '_')}") for stage in STAGES}

    # --- Build jobs ---
    max_subjects = 1 if args.smokescreen else args.max_subjects
    df_jobs = build_jobs(
        subjects=args.subjects,
        experiments=args.experiments,
        sessions_spec=args.sessions,
        max_subjects=max_subjects,
        subjects_to_exclude=set(args.exclude_subjects or []),
        conversion_df=conversion_df,
    )

    if df_jobs.empty:
        print("No jobs to run.")
        sys.exit(0)

    print(f"Jobs to run: {len(df_jobs)}")
    print(df_jobs.to_string(index=False))

    # One error log per experiment (task) at args.root.
    error_logs: dict[str, ConversionErrorLog] = {}
    for exp in df_jobs["experiment"].unique():
        error_logs[exp] = ConversionErrorLog(args.root, exp)
    for _, row in df_jobs.iterrows():
        error_logs[row["experiment"]].record_attempt(row["subject"], int(row["session"]))

    def _record_result(result: dict):
        log = error_logs.get(result['experiment'])
        if log is not None:
            log.record_result(result)

    # ---- SERIAL ----
    if args.serial:
        n_ok = 0
        n_fail = 0
        df_jobs = df_jobs.reset_index(drop=True)

        for i, row in df_jobs.iterrows():
            subj = row["subject"]
            exp = row["experiment"]
            sess = int(row["session"])
            sv = float(row["system_version"])
            us = float(row["unit_scale"])

            print(f"\n[{i+1}/{len(df_jobs)}] {subj} {exp} ses-{sess} (sv={sv}, unit_scale={us})")

            try:
                result = _run_convert_with_tee(
                    subj, exp, sess,
                    system_version=sv, unit_scale=us,
                    brain_regions=brain_regions,
                    root=args.root,
                    overrides=overrides,
                )
                _record_result(result)
                if result.get('any_failure') or result.get('raised'):
                    n_fail += 1
                    print("✗ failed")
                else:
                    n_ok += 1
                    print("✓ finished")
            except Exception:
                import traceback
                n_fail += 1
                print("✗ failed (unexpected orchestrator error)")
                traceback.print_exc()

        for log in error_logs.values():
            log.flush()

        print(f"\nDone. ok={n_ok} fail={n_fail}")
        if args.validate:
            valid = validate_bids(args, df_jobs, error_logs)
            sys.exit(0 if n_fail == 0 and valid else 1)
        sys.exit(0 if n_fail == 0 else 1)

    # ---- PARALLEL (default) ----
    import cmldask.CMLDask as da
    from dask.distributed import as_completed

    log_dir = os.path.expanduser(args.log_directory)
    os.makedirs(log_dir, exist_ok=True)

    client = da.new_dask_client_slurm(
        job_name=args.job_name,
        memory_per_job=args.memory_per_job,
        max_n_jobs=args.max_n_jobs,
        threads_per_job=args.threads_per_job,
        adapt=args.adapt,
        log_directory=log_dir,
    )

    futures = client.map(
        run_job,
        df_jobs["subject"].tolist(),
        df_jobs["experiment"].tolist(),
        df_jobs["session"].tolist(),
        df_jobs["system_version"].tolist(),
        df_jobs["unit_scale"].tolist(),
        [brain_regions] * len(df_jobs),
        [args.root] * len(df_jobs),
        [overrides] * len(df_jobs),
    )

    n_ok = 0
    n_fail = 0
    for fut in as_completed(futures):
        try:
            result = fut.result()
            _record_result(result)
            if result.get('any_failure') or result.get('raised'):
                n_fail += 1
                print("✗ failed:", fut.key)
            else:
                n_ok += 1
                print("✓ finished:", fut.key)
        except Exception as e:
            n_fail += 1
            print("✗ failed:", fut.key)
            print(e)

    for log in error_logs.values():
        log.flush()

    print(f"Done. ok={n_ok} fail={n_fail}")
    if args.validate:
        valid = validate_bids(args, df_jobs, error_logs)
        sys.exit(0 if n_fail == 0 and valid else 1)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
