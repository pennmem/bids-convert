"""BIDS validation utilities used by the intracranial and scalp converters.

Combines two layers of validation:
  1. ``run_bids_validator`` — official Python + npm BIDS Validator (path,
     naming, content, sidecar metadata).
  2. eeg-validation pipelines — round-trip checks comparing CMLReader-loaded
     data against the on-disk BIDS dataset (raw signal, digital LSBs,
     behavioral events, montage).

Per-session artifacts are written under
    /data/BIDS-convert-logs/{experiment}/{subject}/{session}/

with file names following the eeg-validation session_tag convention:
    f"{subject}_{experiment}_{session}"

Each per-session directory ends up with:
    {tag}_bids_convert_log.txt        # tee'd conversion stdout/stderr
    {tag}_bids_validation.txt         # tee'd validation stdout
    {tag}_bids_conversion_error.csv   # 0- or 1-row slice of ConversionErrorLog
    df_*_summary_{tag}_*.csv          # eeg-validation pipeline outputs
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# eeg-validation lives as a git submodule at ``bids-convert/eeg-validation/``
# (note the dash). Python wants the parent of the ``eeg_validation`` package
# on sys.path, so add the submodule directory before importing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_EEG_VALIDATION_SUBMODULE = os.path.join(_HERE, "eeg-validation")
if os.path.isdir(_EEG_VALIDATION_SUBMODULE) and _EEG_VALIDATION_SUBMODULE not in sys.path:
    sys.path.insert(0, _EEG_VALIDATION_SUBMODULE)

from eeg_validation import (
    DigitalSignalPipeline,
    EventsPipeline,
    MontagePipeline,
    RawSignalPipeline,
)
from conversion_error_log import CSV_COLUMNS as ERROR_CSV_COLUMNS


LOG_ROOT = "/data/BIDS-convert-logs"
SECTION_RULE = "=" * 60

# np.isclose tolerances baked into the comparators (kept here so we can
# render them in user-facing warnings without reaching into private state).
SIGNAL_RTOL, SIGNAL_ATOL = 1e-6, 1e-9


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------

def session_tag(subject, experiment, session) -> str:
    return f"{subject}_{experiment}_{session}"


def session_log_dir(experiment, subject, session) -> str:
    p = os.path.join(LOG_ROOT, str(experiment), str(subject), str(session))
    os.makedirs(p, exist_ok=True)
    return p


# ----------------------------------------------------------------------
# Tee
# ----------------------------------------------------------------------

class _Tee:
    """Write to multiple text streams. Used to fan stdout/stderr to a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except ValueError:
                # Underlying stream closed; ignore so the other survives.
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except ValueError:
                pass

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


@contextlib.contextmanager
def tee_to_file(path: str, mode: str = "a"):
    """Tee both stdout AND stderr to `path` while still printing to the console."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, mode, buffering=1)  # line-buffered
    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _Tee(orig_out, f)
        sys.stderr = _Tee(orig_err, f)
        yield
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        f.close()


# ----------------------------------------------------------------------
# Section formatting
# ----------------------------------------------------------------------

def _print_section(title: str, ok: bool, body: str = ""):
    print(title)
    print("PASSED ✓" if ok else "FAILED ✗")
    if body:
        print(body)
    print(SECTION_RULE)
    print()


def _format_signal_warning(channels, tols, mean_diff, std_diff, mse) -> str:
    return (
        f"WARNING: Channels {channels} differ above {tols} between CML and "
        f"BIDS. Mean Difference ({mean_diff}). STD Difference ({std_diff}), "
        f"MSE ({mse})."
    )


def _format_columns_warning(columns) -> str:
    return f"WARNING: Differences in these columns: {columns}."


# ----------------------------------------------------------------------
# Result extraction
# ----------------------------------------------------------------------

def _extract_signal_summary(df_summary) -> Tuple[bool, List[str], float, float, float]:
    """Pull (ok, close_diff_channels, mean_abs_diff, std_diff, mse) from a
    SignalComparator-style df_summary (single row or mean of rows)."""
    if df_summary is None or df_summary.empty:
        return True, [], float("nan"), float("nan"), float("nan")

    # Channels: union across rows (e.g. epoched per event_type).
    channels: List[str] = []
    seen = set()
    for cell in df_summary.get("close_diff_channels", []):
        if isinstance(cell, (list, tuple)):
            for c in cell:
                if c not in seen:
                    seen.add(c)
                    channels.append(c)

    def _avg(col):
        if col not in df_summary.columns:
            return float("nan")
        vals = pd.to_numeric(df_summary[col], errors="coerce")
        return float(vals.mean()) if vals.notna().any() else float("nan")

    mean_diff = _avg("mean_abs_diff")
    std_diff = _avg("std_diff")
    mse = _avg("mse")
    ok = len(channels) == 0
    return ok, channels, mean_diff, std_diff, mse


def _extract_digital_summary(df_summary) -> Tuple[bool, List[str], float, float, float]:
    """Digital comparator stores per-channel n_diff_gt_1 lists. A 'differing'
    channel here is one with at least one sample differing by more than
    1 LSB (the unavoidable rescaling-rounding floor)."""
    if df_summary is None or df_summary.empty:
        return True, [], float("nan"), float("nan"), float("nan")

    channels: List[str] = []
    seen = set()
    for _, row in df_summary.iterrows():
        common = row.get("common_channels") or []
        n_gt_1 = row.get("channel_n_diff_gt_1") or []
        for ch, n in zip(common, n_gt_1):
            if n and ch not in seen:
                seen.add(ch)
                channels.append(str(ch))

    def _avg(col):
        if col not in df_summary.columns:
            return float("nan")
        vals = pd.to_numeric(df_summary[col], errors="coerce")
        return float(vals.mean()) if vals.notna().any() else float("nan")

    return (
        len(channels) == 0,
        channels,
        _avg("mean_abs_diff"),
        _avg("std_diff"),
        _avg("mse"),
    )


def _extract_columns_summary(df_summary) -> Tuple[bool, List[str]]:
    """Pull differing_columns from a DataFrameComparator-style df_summary
    (Events) or its multi-row Montage variant."""
    if df_summary is None or df_summary.empty:
        return True, []
    cols: List[str] = []
    seen = set()
    for cell in df_summary.get("differing_columns", []):
        if isinstance(cell, (list, tuple)):
            for c in cell:
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
    return len(cols) == 0, cols


# ----------------------------------------------------------------------
# Per-pipeline runners
# ----------------------------------------------------------------------

def _run_signal_pipeline(name, pipe, *, digital=False) -> bool:
    """Run a signal-style pipeline and emit a section. Returns True if PASSED."""
    try:
        result = pipe.run()
    except Exception as e:
        body = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _print_section(name, ok=False, body=body)
        return False

    # Pipelines may short-circuit when outputs already exist.
    if isinstance(result, dict) and result.get("skipped"):
        _print_section(name, ok=True, body=f"(skipped: {result.get('reason', '')})")
        return True

    # Collect df_summary candidates from the various return shapes.
    summaries = _collect_summaries(result, signal=not digital, digital=digital)
    df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    extractor = _extract_digital_summary if digital else _extract_signal_summary
    ok, channels, mean_diff, std_diff, mse = extractor(df)
    body = (
        ""
        if ok
        else _format_signal_warning(
            channels, (SIGNAL_RTOL, SIGNAL_ATOL), mean_diff, std_diff, mse,
        )
    )
    _print_section(name, ok=ok, body=body)
    return ok


def _run_columns_pipeline(name, pipe) -> bool:
    """Run an Events- or Montage-style pipeline and emit a section."""
    try:
        result = pipe.run()
    except Exception as e:
        body = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _print_section(name, ok=False, body=body)
        return False

    if isinstance(result, dict) and result.get("skipped"):
        _print_section(name, ok=True, body=f"(skipped: {result.get('reason', '')})")
        return True

    df = _summary_from(result)
    ok, cols = _extract_columns_summary(df)
    body = "" if ok else _format_columns_warning(cols)
    _print_section(name, ok=ok, body=body)
    return ok


def _summary_from(result) -> pd.DataFrame:
    """Best-effort df_summary extraction from any pipeline return value."""
    if result is None:
        return pd.DataFrame()
    if hasattr(result, "df_summary"):
        return result.df_summary
    if isinstance(result, dict):
        if "result" in result and hasattr(result["result"], "df_summary"):
            return result["result"].df_summary
    return pd.DataFrame()


def _collect_summaries(result, *, signal: bool, digital: bool) -> List[pd.DataFrame]:
    """Return df_summary frames from raw / epoched / digital pipeline results."""
    summaries: List[pd.DataFrame] = []
    if result is None:
        return summaries

    # Direct ComparisonResult
    if hasattr(result, "df_summary"):
        df = result.df_summary
        if df is not None and not df.empty:
            summaries.append(df)
        return summaries

    # Dict — Digital pipeline returns {acq: df_summary, "status": df_status}.
    if isinstance(result, dict):
        for key, val in result.items():
            if key == "status":
                continue
            if isinstance(val, pd.DataFrame):
                if not val.empty:
                    summaries.append(val)
            elif hasattr(val, "df_summary"):
                df = val.df_summary
                if df is not None and not df.empty:
                    summaries.append(df)
    return summaries


# ----------------------------------------------------------------------
# Public: BIDS Validator (path + content)
# ----------------------------------------------------------------------

def _read_bidsignore(root: str) -> List[str]:
    """Return non-comment, non-blank patterns from ``{root}/.bidsignore``.

    Always includes ``.bidsignore`` itself so we never flag it as a
    non-BIDS file. Patterns are kept verbatim and matched via
    :func:`fnmatch.fnmatch` against repo-relative paths (with leading
    slash stripped) and against bare basenames.
    """
    patterns: List[str] = [".bidsignore"]
    path = os.path.join(root, ".bidsignore")
    if not os.path.exists(path):
        return patterns
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                patterns.append(s)
    return patterns


def _matches_bidsignore(rel: str, patterns: List[str]) -> bool:
    """True if ``rel`` (a leading-slash repo path) matches any pattern.

    Mirrors what the npm bids-validator does in spirit: matches both
    the full path (without the leading slash) and the basename.
    """
    if not patterns:
        return False
    import fnmatch
    rel_stripped = rel.lstrip("/")
    base = os.path.basename(rel_stripped)
    for pat in patterns:
        if fnmatch.fnmatch(rel_stripped, pat) or fnmatch.fnmatch(base, pat):
            return True
        # Support directory-style patterns like 'derivatives/' that the
        # npm validator treats as "anywhere under derivatives".
        if pat.endswith("/") and rel_stripped.startswith(pat):
            return True
    return False


def run_bids_validator(root: str, *, timeout: Optional[float] = None) -> bool:
    """Validate a BIDS dataset at ``root``.

    Two-layer approach:
      1. Python path/naming validation via ``bids_validator``.
      2. Full CLI validation via the npm ``bids-validator`` (file contents,
         required metadata, sidecar completeness) when on PATH. Output is
         streamed line-by-line so progress is visible while it runs.

    Returns True if no errors were found, False otherwise. If ``timeout`` is
    set and the CLI exceeds it, the subprocess is terminated and the run is
    flagged as failed.
    """
    title = f"BIDS Validator: {root}"
    any_errors = False

    # Print title up front so the streamed CLI output is visible in context.
    print(title)

    # --- Layer 1: Python path/naming validation ---
    try:
        from bids_validator import BIDSValidator
        validator = BIDSValidator()

        # Read .bidsignore (npm CLI feature) and honor it ourselves so
        # Layer 1 doesn't flag files the npm validator would skip.
        ignore_patterns = _read_bidsignore(root)

        naming_errors: List[str] = []
        for dirpath, _, files in os.walk(root):
            for fname in files:
                full = os.path.join(dirpath, fname)
                rel = "/" + os.path.relpath(full, root)
                if _matches_bidsignore(rel, ignore_patterns):
                    continue
                if not validator.is_bids(rel):
                    naming_errors.append(rel)
        if naming_errors:
            print(f"[Path validation] {len(naming_errors)} file(s) with non-BIDS-compliant names:")
            for p in sorted(naming_errors):
                print(f"  ✗ {p}")
            any_errors = True
        else:
            print("[Path validation] All file names are BIDS-compliant.")
    except ImportError:
        print("[Path validation] bids_validator not installed — skipping. "
              "(pip install bids_validator)")

    # --- Layer 2: Full CLI validation (streamed) ---
    if subprocess.run(["which", "bids-validator"], capture_output=True).returncode == 0:
        cmd = ["bids-validator", root, "--verbose"]
    elif subprocess.run(["which", "npx"], capture_output=True).returncode == 0:
        cmd = ["npx", "bids-validator", root, "--verbose"]
    else:
        cmd = None

    if cmd is not None:
        print(f"[Full validation] Running: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout so we get one stream
            text=True,
            bufsize=1,                 # line-buffered
        )

        had_syntax_error = False
        timed_out = False
        try:
            # Stream output as it arrives. tee_to_file at the call site fans
            # this to both console and the per-experiment log file.
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                if "SyntaxError" in line:
                    had_syntax_error = True
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
            print(f"\nWARNING: bids-validator exceeded timeout={timeout}s — terminated.")

        if timed_out:
            any_errors = True
        elif proc.returncode != 0:
            if had_syntax_error:
                print(
                    f"WARNING: bids-validator failed to start (likely Node.js version too old). "
                    f"Run manually: {' '.join(cmd)}"
                )
            else:
                any_errors = True
    else:
        print(
            "[Full validation] bids-validator not found — skipping content/metadata checks. "
            "Install: npm install -g bids-validator | "
            f"docker run --rm -v {root}:/data:ro bids/validator /data"
        )

    # Footer: status + rule. Order differs slightly from _print_section's
    # title/status/body/rule because the body is streamed, but the
    # PASSED/FAILED line still bookends the section.
    print("PASSED ✓" if not any_errors else "FAILED ✗")
    print(SECTION_RULE)
    print()
    return not any_errors


# ----------------------------------------------------------------------
# Public: per-session pipeline runners
# ----------------------------------------------------------------------

def run_scalp_validation(
    subject, experiment, session, bids_root, *,
    out_dir: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Run the scalp eeg-validation pipelines for one session."""
    out_dir = out_dir or session_log_dir(experiment, subject, session)
    common = dict(
        subject=subject, experiment=experiment, session=session,
        bids_root=bids_root, out_path=out_dir, verbose=verbose,
        skip_if_exists=False,
    )
    overall_ok = True
    overall_ok &= _run_signal_pipeline(
        "RawSignalPipeline", RawSignalPipeline(**common),
    )

    overall_ok &= _run_signal_pipeline(
        "DigitalSignalPipeline", DigitalSignalPipeline(**common), digital=True,
    )
    overall_ok &= _run_columns_pipeline(
        "EventsPipeline", EventsPipeline(**common),
    )
    return overall_ok


def run_intra_validation(
    subject, experiment, session, bids_root, *,
    out_dir: Optional[str] = None,
    localization: Optional[int] = None,
    montage: Optional[int] = None,
    verbose: bool = False,
) -> bool:
    """Run the intracranial eeg-validation pipelines for one session.

    RawSignalPipeline and MontagePipeline are each run twice
    (acquisition='contacts', then 'pairs'). DigitalSignalPipeline
    auto-iterates both acquisitions internally.
    """
    out_dir = out_dir or session_log_dir(experiment, subject, session)
    common = dict(
        subject=subject, experiment=experiment, session=session,
        bids_root=bids_root, out_path=out_dir,
        localization=localization, montage=montage,
        verbose=verbose, skip_if_exists=False,
    )
    overall_ok = True
    overall_ok &= _run_signal_pipeline(
        "RawSignalPipeline (contacts)",
        RawSignalPipeline(**common, acquisition="contacts"),
    )
    overall_ok &= _run_signal_pipeline(
        "RawSignalPipeline (pairs)",
        RawSignalPipeline(**common, acquisition="pairs"),
    )
    overall_ok &= _run_signal_pipeline(
        "DigitalSignalPipeline",
        DigitalSignalPipeline(**common),
        digital=True,
    )
    overall_ok &= _run_columns_pipeline(
        "EventsPipeline",
        EventsPipeline(**common),
    )
    overall_ok &= _run_columns_pipeline(
        "MontagePipeline (contacts)",
        MontagePipeline(**common, acquisition="contacts"),
    )
    overall_ok &= _run_columns_pipeline(
        "MontagePipeline (pairs)",
        MontagePipeline(**common, acquisition="pairs"),
    )
    return overall_ok


# ----------------------------------------------------------------------
# Per-session error CSV slice
# ----------------------------------------------------------------------

def write_session_heartbeat_status(log_dir: str, tag: str, result: dict) -> str:
    """Persist heartbeat-correction status for one session.

    Writes ``{log_dir}/{tag}_heartbeat_status.txt`` with two lines:

        applied=<bool>
        status=<str>

    ``status`` is one of: ``"applied"``, ``"failed: <reason>"``, or
    ``"skipped (system_version=N)"``. ``result`` is the dict returned by
    ``convert_one_job`` — values default to ``"unknown"`` / ``False`` if
    the converter didn't record them.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{tag}_heartbeat_status.txt")
    with open(path, "w") as f:
        f.write(f"applied={bool(result.get('heartbeat_applied', False))}\n")
        f.write(f"status={result.get('heartbeat_status', 'unknown')}\n")
    return path


def write_session_error_csv(error_log, subject, session, out_dir, tag) -> str:
    """Write a 0- or 1-row CSV at ``{out_dir}/{tag}_bids_conversion_error.csv``.

    The row matches the schema of ``bids_conversion_error_{task}.csv``.
    Empty if this (subject, session) had no recorded failures.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_bids_conversion_error.csv")
    rows: List[Dict[str, Any]] = []
    if error_log is not None:
        for r in getattr(error_log, "_rows", []):
            if str(r.get("subject")) == str(subject) and int(r.get("session", -1)) == int(session):
                rows.append(r)
    df = pd.DataFrame(rows, columns=ERROR_CSV_COLUMNS)
    df.to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------
# Validation orchestration shared by both converters
# ----------------------------------------------------------------------

def validate_jobs(
    df_jobs: pd.DataFrame,
    bids_root_for_job,
    error_logs: Dict[str, Any],
    *,
    intracranial: bool,
    localization_for_job=None,
    montage_for_job=None,
    log_root_per_experiment: bool = True,
    verbose: bool = False,
) -> bool:
    """Per-session pipeline validation + dataset-wide BIDS Validator.

    Parameters
    ----------
    df_jobs : DataFrame with columns at least 'subject', 'experiment', 'session'.
    bids_root_for_job : callable(row) -> str
        Returns the BIDS root for a given job row. Intracranial passes a
        single root; scalp resolves a per-experiment root.
    error_logs : {experiment: ConversionErrorLog}
    intracranial : bool
        Selects intracranial vs scalp pipeline set.
    localization_for_job, montage_for_job : optional callable(row) -> int|None
        Used only by intracranial.
    log_root_per_experiment : bool
        If True, run ``run_bids_validator`` once per (root, experiment) seen
        in df_jobs (typical scalp). If False, run once per unique root
        (typical intracranial).
    verbose : bool
        Forwarded to the eeg-validation pipelines.
    """
    overall_ok = True
    seen_roots: Dict[str, str] = {}  # root -> experiment label (for top-level log path)

    for _, row in df_jobs.iterrows():
        subj = row["subject"]
        exp = row["experiment"]
        sess = int(row["session"])
        root = bids_root_for_job(row)
        log_dir = session_log_dir(exp, subj, sess)
        tag = session_tag(subj, exp, sess)

        write_session_error_csv(error_logs.get(exp), subj, sess, log_dir, tag)

        with tee_to_file(os.path.join(log_dir, f"{tag}_bids_validation.txt"), mode="w"):
            print(f"Validation: {tag}")
            print(SECTION_RULE)
            print()
            if intracranial:
                ok = run_intra_validation(
                    subj, exp, sess, root,
                    out_dir=log_dir,
                    localization=localization_for_job(row) if localization_for_job else None,
                    montage=montage_for_job(row) if montage_for_job else None,
                    verbose=verbose,
                )
            else:
                ok = run_scalp_validation(
                    subj, exp, sess, root,
                    out_dir=log_dir, verbose=verbose,
                )

        seen_roots[root] = exp
        overall_ok = overall_ok and ok

    # Dataset-wide BIDS validator. One run per unique (root[, experiment]).
    if log_root_per_experiment:
        # Scalp: one root per experiment; one log file per experiment.
        for root, exp in seen_roots.items():
            top = os.path.join(LOG_ROOT, str(exp), "_bids_validator.txt")
            with tee_to_file(top, mode="w"):
                overall_ok = run_bids_validator(root) and overall_ok
    else:
        # Intracranial: one shared root across experiments. One log per root,
        # filed under whichever experiment we saw first (purely for layout).
        for root, exp in seen_roots.items():
            top = os.path.join(LOG_ROOT, str(exp), "_bids_validator.txt")
            with tee_to_file(top, mode="w"):
                overall_ok = run_bids_validator(root) and overall_ok
            break  # one root for all experiments

    return overall_ok
