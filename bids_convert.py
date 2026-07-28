#!/usr/bin/env python3
"""Unified BIDS conversion entry point for the CML ``bids-convert`` repo.

One CLI for both modalities. The modality is inferred from the experiments
being converted (``--modality`` overrides / disambiguates), and everything
downstream — job building, stage gating, overwrite resolution, the Slurm+Dask
orchestrator, the per-task error CSV and validation — is shared. See the
``cli`` package for the engine itself.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import registry  # noqa: E402
from cli.jobs import build_jobs  # noqa: E402
from cli.overwrite import resolve_overwrite, valid_tokens  # noqa: E402
from cli.runner import make_error_logs, run_jobs  # noqa: E402
from cli.validation import validate_bids, validation_requested  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONVERSION_CSV = os.path.join(_HERE, "intracranial", "system_1_unit_conversions.csv")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="bids_convert.py",
        description="Convert CML scalp or intracranial EEG data to BIDS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
experiments:
  intracranial: {', '.join(registry.experiments_for(registry.INTRACRANIAL))}
  scalp:        {', '.join(registry.experiments_for(registry.SCALP))}

overwrite components:
  intracranial: {', '.join(valid_tokens(registry.INTRACRANIAL))}
  scalp:        {', '.join(valid_tokens(registry.SCALP))}

examples:
  # One intracranial subject, one experiment, one session (serial)
  %(prog)s --experiments FR1 --subjects R1001P --sessions 0 --serial \\
      --root /scratch/me/BIDS

  # Sessions 0-4 for two subjects, forcing the behavioral stage to re-run
  %(prog)s --experiments FR1 --subjects R1001P R1002P --sessions 0:5 \\
      --overwrite beh --root /scratch/me/BIDS

  # A whole scalp experiment in parallel, re-converting everything
  %(prog)s --experiments ltpFR2 --root /data/LTP_BIDS/ltpFR2 --overwrite

  # Just show which jobs would run
  %(prog)s --modality scalp --smokescreen --root /scratch/me/BIDS --dry-run
""",
    )

    ap.add_argument("--modality", choices=list(registry.MODALITIES), default=None,
                    help="Conversion modality. Inferred from --experiments when omitted.")
    ap.add_argument("--root", required=True,
                    help="BIDS root directory. Used verbatim — the experiment is "
                         "never appended, so point this at the dataset you want.")

    # ---- selection ----
    sel = ap.add_argument_group("selection")
    sel.add_argument("--experiments", nargs="+", default=None, metavar="EXP",
                     choices=sorted(registry.EXPERIMENTS),
                     help="Experiments to convert. Default: every experiment of "
                          "the selected modality.")
    sel.add_argument("--subjects", nargs="+", default=None, metavar="SUBJ",
                     help="Subject IDs to convert. Default: all subjects.")
    sel.add_argument("--sessions", nargs="+", default=None, metavar="SPEC",
                     help="Session specifiers: int (e.g. 3) or slice (e.g. 0:5, :3, 2:). "
                          "Requires --subjects or --experiments.")
    sel.add_argument("--exclude-subjects", nargs="*", default=None, metavar="SUBJ",
                     help="Subjects to exclude. Default: modality-specific test IDs.")
    sel.add_argument("--recently-modified", default=None, metavar="JSON",
                     help="Path to a recently_modified.json ({subject: [sessions]}). "
                          "Restricts jobs to exactly those (subject, session) pairs.")
    sel.add_argument("--smokescreen", action="store_true", default=False,
                     help="Quick test: limit to 1 subject per experiment.")

    # ---- behavior ----
    beh = ap.add_argument_group("behavior")
    beh.add_argument("--overwrite", nargs="*", default=None, metavar="COMPONENT",
                     help="Re-convert existing outputs. Bare, it overwrites everything; "
                          "with component names (beh, eeg, montage, electrodes, channels, "
                          "or exact stage names) only those are overwritten. Without this "
                          "flag each stage runs only when its outputs are missing.")
    beh.add_argument("--serial", action="store_true", default=False,
                     help="Run jobs one at a time instead of in parallel (Dask).")
    beh.add_argument("--force", action="store_true", default=False,
                     help="Downgrade per-stage conversion failures to [WARN] and keep "
                          "going. By default any stage failure aborts that session.")
    beh.add_argument("--dry-run", action="store_true", default=False,
                     help="Print the resolved settings and job table, then exit.")
    beh.add_argument("--verbose", action="store_true", default=False,
                     help="Verbose output from the validation pipelines.")

    # ---- validation ----
    val = ap.add_argument_group("validation")
    val.add_argument("--validate", action="store_true", default=False,
                     help="Run both validations after conversion (alias for "
                          "--bids-validator --eeg-validator).")
    val.add_argument("--bids-validator", dest="bids_validator", action="store_true",
                     default=False,
                     help="Run the official BIDS Validator only (path/naming + npm CLI).")
    val.add_argument("--eeg-validator", dest="eeg_validator", action="store_true",
                     default=False,
                     help="Run the eeg-validation pipelines only (CMLReader vs BIDS).")
    val.add_argument("--validate-only", action="store_true", default=False,
                     help="Skip conversion and only validate --root for the selected "
                          "jobs. Combine with --bids-validator / --eeg-validator to scope.")

    # ---- parallel (Dask/Slurm) ----
    par = ap.add_argument_group("parallel (Slurm + Dask)")
    par.add_argument("--job-name", default="bids_convert")
    par.add_argument("--memory-per-job", default="100GB")
    par.add_argument("--max-n-jobs", type=int, default=20)
    par.add_argument("--threads-per-job", type=int, default=1)
    par.add_argument("--adapt", action="store_true", default=True)
    par.add_argument("--no-adapt", dest="adapt", action="store_false")
    par.add_argument("--log-directory", default="~/logs/")

    # ---- intracranial only ----
    intra = ap.add_argument_group("intracranial only")
    intra.add_argument("--conversion-csv", default=DEFAULT_CONVERSION_CSV,
                       help="CSV of per-session System-1 unit conversions "
                            "(default: intracranial/system_1_unit_conversions.csv).")

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.sessions is not None and args.subjects is None and args.experiments is None:
        ap.error("--sessions requires at least --subjects or --experiments to be specified.")

    try:
        modality = registry.resolve_modality(args.experiments, args.modality)
    except ValueError as e:
        ap.error(str(e))

    try:
        overrides = resolve_overwrite(args.overwrite, modality)
    except ValueError as e:
        ap.error(str(e))

    exclude_subjects = (
        args.exclude_subjects if args.exclude_subjects is not None
        else registry.DEFAULT_EXCLUDE_SUBJECTS[modality]
    )
    experiments = args.experiments or registry.experiments_for(modality)

    print("\nRunning with settings:")
    print("Modality:           ", modality)
    print("Root:               ", args.root)
    print("Experiments:        ", ", ".join(experiments))
    print("Subjects:           ", ", ".join(args.subjects) if args.subjects else "(all)")
    print("Sessions:           ", " ".join(args.sessions) if args.sessions else "(all)")
    print("Excluded subjects:  ", ", ".join(exclude_subjects) or "(none)")
    print("Overwrite:          ", ", ".join(s for s, v in overrides.items() if v) or "(nothing — resume)")
    print("Mode:               ", "serial" if args.serial else "parallel (Slurm+Dask)")
    print("On stage failure:   ", "warn and continue (--force)" if args.force else "abort session")
    print("-" * 50 + "\n")

    df_jobs = build_jobs(
        modality=modality,
        experiments=args.experiments,
        subjects=args.subjects,
        sessions_spec=args.sessions,
        exclude_subjects=exclude_subjects,
        smokescreen=args.smokescreen,
        recently_modified=args.recently_modified,
        conversion_csv=args.conversion_csv if modality == registry.INTRACRANIAL else None,
    )

    if df_jobs.empty:
        print("No jobs to run.")
        sys.exit(0)

    print(f"Jobs to run: {len(df_jobs)}")
    print(df_jobs.to_string(index=False))

    if args.dry_run:
        print("\n--dry-run: nothing was converted.")
        sys.exit(0)

    error_logs = make_error_logs(df_jobs, args.root)

    if args.validate_only:
        valid = validate_bids(args, df_jobs, error_logs, modality)
        sys.exit(0 if valid else 1)

    brain_regions = None
    if modality == registry.INTRACRANIAL:
        from intracranial.intracranial_BIDS_converter import intracranial_BIDS_converter
        brain_regions = {br: 1 for br in intracranial_BIDS_converter.BRAIN_REGIONS}

    tally = run_jobs(
        df_jobs,
        modality=modality,
        root=args.root,
        overrides=overrides,
        force=args.force,
        serial=args.serial,
        brain_regions=brain_regions,
        dask_opts={
            "job_name": args.job_name,
            "memory_per_job": args.memory_per_job,
            "max_n_jobs": args.max_n_jobs,
            "threads_per_job": args.threads_per_job,
            "adapt": args.adapt,
            "log_directory": args.log_directory,
        },
        error_logs=error_logs,
    )

    valid = True
    if validation_requested(args):
        df_validate = pd.DataFrame(
            tally.converted_rows, columns=["subject", "experiment", "session"],
        )
        valid = validate_bids(args, df_validate, error_logs, modality)

    sys.exit(0 if tally.n_fail == 0 and valid else 1)


if __name__ == "__main__":
    main()
