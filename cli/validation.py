"""Post-conversion validation, shared by both modalities.

Thin wrapper over ``bids_validation.validate_jobs``: resolves which of the two
validation layers the flags asked for, then runs them against the single BIDS
root the CLI was given.
"""

from __future__ import annotations

from bids_validation import validate_jobs

from .registry import INTRACRANIAL


def resolve_validation_flags(args):
    """Return (run_eeg_pipelines, run_dataset_validator) from the CLI flags.

    If neither --bids-validator nor --eeg-validator was passed, --validate /
    --validate-only mean "both".
    """
    eeg = bool(getattr(args, "eeg_validator", False))
    bids = bool(getattr(args, "bids_validator", False))
    if eeg or bids:
        return eeg, bids
    return True, True


def validation_requested(args) -> bool:
    return bool(args.validate or args.bids_validator or args.eeg_validator)


def validate_bids(args, df_jobs, error_logs, modality: str) -> bool:
    """Run per-session eeg-validation pipelines and/or the BIDS Validator."""
    run_eeg, run_bids = resolve_validation_flags(args)
    return validate_jobs(
        df_jobs,
        bids_root_for_job=lambda row: args.root,
        error_logs=error_logs,
        intracranial=(modality == INTRACRANIAL),
        # --root is one dataset for both modalities now, so the dataset-wide
        # validator runs once per root rather than once per experiment.
        log_root_per_experiment=False,
        verbose=args.verbose,
        run_eeg_pipelines=run_eeg,
        run_dataset_validator=run_bids,
    )
