#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import importlib
import importlib.util
import pandas as pd
import cmlreaders as cml

# --- package path setup ---
sys.path.insert(0, os.path.expanduser("~/bids-convert"))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from intracranial.intracranial_BIDS_converter import intracranial_BIDS_converter

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
    Converter = _get_converter(experiment)

    converter = Converter(
        subject, experiment, session,
        system_version, unit_scale,
        monopolar, bipolar, mni, tal, area,
        brain_regions,
        root=root,
    )
    return converter.run()


def build_jobs(
    *,
    experiments: list[str],
    max_subjects: int,
    subjects_to_exclude: set[str],
    conversion_df: pd.DataFrame,
):
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

    # Normalize dtypes
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
        print(
            f"WARNING: Skipping {missing.sum()} job(s) not found in the conversion CSV.\n"
            f"  Add the following rows to system_1_unit_conversions.csv "
            f"(columns: subject, experiment, session, system_version, conversion_to_V):"
        )
        print(df_jobs2.loc[missing, merge_keys].to_string(index=False))

    df_jobs2 = df_jobs2.loc[~missing].copy()
    df_jobs2["system_version"] = df_jobs2["system_version"].astype(float)
    df_jobs2["unit_scale"] = df_jobs2["conversion_to_V"].astype(float)

    return df_jobs2


# Top-level function for Dask mapping (avoid lambda/pickle weirdness)
def run_job(
    subject, experiment, session, system_version, unit_scale,
    monopolar, bipolar, mni, tal, area, brain_regions, root
):
    return convert_one_job(
        subject, experiment, int(session), float(system_version), float(unit_scale),
        monopolar=monopolar, bipolar=bipolar, mni=mni,
        tal=tal, area=area,
        brain_regions=brain_regions,
        root=root,
    )


def validate_bids_output(root: str):
    """
    Validate a BIDS dataset at `root`.

    Two-layer approach:
      1. Python path validation (bids_validator): checks every file's name and
         path against BIDS naming conventions.  Always available.
      2. Full CLI validation (bids-validator npm tool): checks file contents,
         required metadata fields, and sidecar completeness.  Only runs when
         the CLI is on PATH.

    Returns True if no errors were found, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"BIDS Validation: {root}")
    print(f"{'='*60}")

    any_errors = False

    # --- Layer 1: Python path/naming validation ---
    try:
        from bids_validator import BIDSValidator
        validator = BIDSValidator()
        naming_errors = []
        for dirpath, _, files in os.walk(root):
            for fname in files:
                full = os.path.join(dirpath, fname)
                rel = "/" + os.path.relpath(full, root)
                if not validator.is_bids(rel):
                    naming_errors.append(rel)

        if naming_errors:
            print(f"\n[Path validation] {len(naming_errors)} file(s) with non-BIDS-compliant names:")
            for p in sorted(naming_errors):
                print(f"  ✗ {p}")
            any_errors = True
        else:
            print(f"\n[Path validation] All file names are BIDS-compliant. ✓")

    except ImportError:
        print("[Path validation] bids_validator not installed — skipping. (pip install bids_validator)")

    # --- Layer 2: Full CLI validation (npm bids-validator) ---
    # Prefer direct invocation; fall back to npx (local npm install).
    if subprocess.run(["which", "bids-validator"], capture_output=True).returncode == 0:
        cmd = ["bids-validator", root, "--verbose"]
    elif subprocess.run(["which", "npx"], capture_output=True).returncode == 0:
        cmd = ["npx", "bids-validator", root, "--verbose"]
    else:
        cmd = None

    if cmd is not None:
        print(f"\n[Full validation] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            # Distinguish a validator crash (e.g. Node.js too old) from actual BIDS errors.
            if "SyntaxError" in result.stderr or "SyntaxError" in result.stdout:
                print(
                    f"WARNING: bids-validator failed to start (likely Node.js version too old).\n"
                    f"  Run manually in your shell: {' '.join(cmd)}\n"
                    f"  stderr: {result.stderr.strip()}"
                )
            else:
                print(result.stderr)
                any_errors = True
    else:
        print(
            "\n[Full validation] bids-validator not found — skipping content/metadata checks.\n"
            "  Install via:  npm install -g bids-validator\n"
            "  Or run via Docker:\n"
            f"    docker run --rm -v {root}:/data:ro bids/validator /data"
        )

    print(f"\nValidation {'PASSED ✓' if not any_errors else 'FAILED ✗'}")
    print(f"{'='*60}\n")
    return not any_errors


def main():
    ap = argparse.ArgumentParser(description="Intracranial BIDS conversion (single / serial / parallel).")
    ap.add_argument("--mode", choices=["single", "serial", "parallel"], required=False)

    ap.add_argument("--conversion-csv", default=os.path.join(_SCRIPT_DIR, "system_1_unit_conversions.csv"))
    ap.add_argument("--root", default="/home1/maint/LTP_BIDS/")

    # shared flags
    ap.add_argument("--monopolar", action="store_true", default=True)
    ap.add_argument("--no-monopolar", dest="monopolar", action="store_false")
    ap.add_argument("--bipolar", action="store_true", default=True)
    ap.add_argument("--no-bipolar", dest="bipolar", action="store_false")
    ap.add_argument("--mni", action="store_true", default=True)
    ap.add_argument("--no-mni", dest="mni", action="store_false")
    ap.add_argument("--tal", action="store_true", default=False)
    ap.add_argument("--area", action="store_true", default=False)
    ap.add_argument("--validate", action="store_true", default=False,
                    help="Run BIDS validation on the output directory after conversion.")
    ap.add_argument("--validate-only", action="store_true", default=False,
                    help="Skip conversion and only run BIDS validation on --root.")

    # single mode args
    ap.add_argument("--subject")
    ap.add_argument("--experiment", choices=sorted(_EXPERIMENT_MODULES.keys()))
    ap.add_argument("--session", type=int)

    # serial/parallel job-building args
    ap.add_argument("--experiments", nargs="*", default=list(_EXPERIMENT_MODULES.keys()))
    ap.add_argument("--max-subjects", type=int, default=10)
    ap.add_argument("--exclude-subjects", nargs="*", default=["LTP001"])
    ap.add_argument("--smokescreen", action="store_true", default=False,
                    help="Quick test: limit to 1 subject per experiment.")

    # parallel args
    ap.add_argument("--job-name", default="bids_convert")
    ap.add_argument("--memory-per-job", default="50GB")
    ap.add_argument("--max-n-jobs", type=int, default=10)
    ap.add_argument("--threads-per-job", type=int, default=1)
    ap.add_argument("--adapt", action="store_true", default=True)
    ap.add_argument("--no-adapt", dest="adapt", action="store_false")
    ap.add_argument("--log-directory", default="~/logs/")

    args = ap.parse_args()

    if args.validate_only:
        valid = validate_bids_output(args.root)
        sys.exit(0 if valid else 1)

    conversion_df = pd.read_csv(args.conversion_csv)
    conversion_df["session"] = conversion_df["session"].astype(int)

    brain_regions = {br: 1 for br in intracranial_BIDS_converter.BRAIN_REGIONS}

    # ---- SINGLE ----
    if args.mode == "single":
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
        if args.validate:
            valid = validate_bids_output(args.root)
            sys.exit(0 if ok and valid else 1)
        sys.exit(0 if ok else 1)

    # Build jobs for serial/parallel
    max_subjects = 1 if args.smokescreen else args.max_subjects
    df_jobs2 = build_jobs(
        experiments=args.experiments,
        max_subjects=max_subjects,
        subjects_to_exclude=set(args.exclude_subjects),
        conversion_df=conversion_df,
    )

    print("Jobs to run:", len(df_jobs2))
    print(df_jobs2.head())

    # ---- SERIAL ----
    if args.mode == "serial":
        n_ok = 0
        n_fail = 0
        df_jobs2 = df_jobs2.reset_index(drop=True)

        for i, row in df_jobs2.iterrows():
            subj = row["subject"]
            exp = row["experiment"]
            sess = int(row["session"])
            sv = float(row["system_version"])
            us = float(row["unit_scale"])

            print(f"\n[{i+1}/{len(df_jobs2)}] {subj} {exp} ses-{sess} (sv={sv}, unit_scale={us})")

            try:
                _ = convert_one_job(
                    subj, exp, sess, sv, us,
                    monopolar=args.monopolar,
                    bipolar=args.bipolar,
                    mni=args.mni,
                    tal=args.tal,
                    area=args.area,
                    brain_regions=brain_regions,
                    root=args.root,
                )
                n_ok += 1
                print("✓ finished")
            except Exception:
                import traceback
                n_fail += 1
                print("✗ failed")
                traceback.print_exc()

        print(f"\nDone. ok={n_ok} fail={n_fail}")
        if args.validate:
            valid = validate_bids_output(args.root)
            sys.exit(0 if n_fail == 0 and valid else 1)
        sys.exit(0 if n_fail == 0 else 1)

    # ---- PARALLEL ----
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
        log_directory=log_dir,  # expanded + ensured exists
    )

    futures = client.map(
        run_job,
        df_jobs2["subject"].tolist(),
        df_jobs2["experiment"].tolist(),
        df_jobs2["session"].tolist(),
        df_jobs2["system_version"].tolist(),
        df_jobs2["unit_scale"].tolist(),
        [args.monopolar] * len(df_jobs2),
        [args.bipolar] * len(df_jobs2),
        [args.mni] * len(df_jobs2),
        [args.tal] * len(df_jobs2),
        [args.area] * len(df_jobs2),
        [brain_regions] * len(df_jobs2),
        [args.root] * len(df_jobs2),
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
