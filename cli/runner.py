"""Job orchestration: one worker function, one serial path, one Dask path.

Everything downstream of "here is a table of (subject, experiment, session)"
lives here and is modality-agnostic — the registry resolves which converter to
build, and every job returns the same result dict so the per-task error CSV,
the console summary and the validation step have a single shape to consume.
"""

from __future__ import annotations

import os
import sys
import traceback

import pandas as pd

from . import REPO_ROOT, registry

from conversion_error_log import ConversionErrorLog, cmlreader_involved  # noqa: E402
from bids_validation import session_log_dir, session_tag, tee_to_file  # noqa: E402


# ----------------------------------------------------------------------
# Single job
# ----------------------------------------------------------------------
def _result(status, subject, experiment, session, root, *, files_written=(),
            files_not_written=(), any_failure=False, raised=False, error_stage="",
            error_type="", error_message="", cmlreader_failure=False, message="",
            no_eeg=False, no_eeg_reason=""):
    return {
        "status": status,
        "subject": str(subject),
        "experiment": experiment,
        "session": int(session),
        "root": root,
        "files_written": list(files_written),
        "files_not_written": list(files_not_written),
        "any_failure": bool(any_failure),
        "raised": bool(raised),
        "no_eeg": bool(no_eeg),
        "no_eeg_reason": no_eeg_reason or "",
        "error_stage": error_stage or "",
        "error_type": error_type or "",
        "error_message": error_message or "",
        "cmlreader_failure": bool(cmlreader_failure),
        "message": message,
    }


def convert_one_job(subject, experiment, session, *, root, overrides, force, job=None):
    """Run one (subject, experiment, session) job and return a result dict.

    Never raises: orchestration failures are reported in the returned dict so
    the result round-trips through Dask workers unchanged. When every stage's
    outputs already exist the converter is never run and the job is reported
    as ``skip_existing`` — the caller leaves the error CSV untouched in that
    case, preserving any prior error rows.
    """
    spec = registry.get(experiment)
    stages = registry.STAGES_BY_MODALITY[spec.modality]

    converter = None
    exc = None
    try:
        converter = spec.build(subject, session, root=root, overrides=overrides, job=job)
        converter.force = bool(force)
        if not converter.stages_to_run():
            return _result(
                "skip_existing", subject, experiment, session, root,
                files_written=stages,
                message=f"SKIP existing outputs: {subject} {experiment} {session}",
            )
        converter.run()
    except Exception as e:
        exc = e
        traceback.print_exc()

    if converter is None:
        # The converter could not even be constructed — nothing was written.
        return _result(
            "ran", subject, experiment, session, root,
            files_not_written=stages, any_failure=True, raised=True,
            error_stage="build", error_type=type(exc).__name__,
            error_message=" ".join(str(exc).splitlines()).strip(),
            cmlreader_failure=cmlreader_involved(exc),
            message=f"FAILED (build): {subject} {experiment} {session}",
        )

    if exc is not None and not getattr(converter, "first_error_stage", None):
        # The failure came from outside a tracked stage (e.g. cml_reader()).
        converter.first_error_stage = "run"
        converter.first_exception = exc

    report = converter.stage_report()
    first_exc = report["exception"] or exc
    failed = report["any_failure"] or exc is not None
    error_stage = report["error_stage"] or ("run" if exc is not None else "")

    if failed:
        detail = f"{type(first_exc).__name__}: {first_exc}" if first_exc is not None else "see log"
        detail = " ".join(str(detail).splitlines()).strip()
        message = (f"FAILED at '{error_stage}': {subject} {experiment} {session} — {detail}")
    elif report.get("no_eeg"):
        message = (f"DONE (no EEG): {subject} {experiment} {session} — "
                   f"{report.get('no_eeg_reason') or 'no recording to convert'}")
    else:
        message = f"DONE: {subject} {experiment} {session}"

    return _result(
        "ran", subject, experiment, session, root,
        files_written=report["files_written"],
        files_not_written=report["files_not_written"],
        any_failure=failed,
        raised=exc is not None,
        no_eeg=report.get("no_eeg", False),
        no_eeg_reason=report.get("no_eeg_reason") or "",
        error_stage=error_stage,
        error_type=type(first_exc).__name__ if first_exc is not None else "",
        error_message=" ".join(str(first_exc).splitlines()).strip() if first_exc is not None else "",
        cmlreader_failure=cmlreader_involved(first_exc) if first_exc is not None else False,
        message=message,
    )


def run_job(subject, experiment, session, job, root, overrides, force):
    """Top-level (picklable) worker: convert one session, teeing its output.

    stdout/stderr land in the per-session conversion log under
    ``/data/BIDS-convert-logs/<experiment>/<subject>/<session>/``.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    log_dir = session_log_dir(experiment, subject, int(session))
    tag = session_tag(subject, experiment, int(session))
    log_path = os.path.join(log_dir, f"{tag}_bids_convert_log.txt")
    with tee_to_file(log_path, mode="w"):
        return convert_one_job(
            subject, experiment, int(session),
            root=root, overrides=overrides, force=force, job=job,
        )


# ----------------------------------------------------------------------
# Job payloads
# ----------------------------------------------------------------------
def job_payload(row, modality, brain_regions=None):
    """Per-session extras the converter needs beyond subject/experiment/session."""
    if modality != registry.INTRACRANIAL:
        return None
    return {
        "system_version": float(row["system_version"]),
        "unit_scale": float(row["unit_scale"]),
        "brain_regions": brain_regions,
    }


def make_error_logs(df_jobs, root):
    """One ConversionErrorLog per experiment, all at the single BIDS root."""
    return {exp: ConversionErrorLog(root, exp) for exp in df_jobs["experiment"].unique()}


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
class _Tally:
    """Shared result handling for the serial and Dask paths."""

    def __init__(self, error_logs):
        self.error_logs = error_logs
        self.n_ok = 0
        self.n_fail = 0
        self.n_skip = 0
        self.converted_rows: list[dict] = []

    def handle(self, result):
        if not isinstance(result, dict):
            print(f"✗ unexpected result: {result!r}")
            self.n_fail += 1
            return

        if result.get("status") == "skip_existing":
            # Deliberately not recorded: leave any prior error rows in place.
            self.n_skip += 1
            print(f"↷ {result.get('message', '')}")
            return

        log = self.error_logs.get(result.get("experiment"))
        if log is not None:
            log.record_result(result)

        if result.get("any_failure") or result.get("raised"):
            self.n_fail += 1
            print(f"✗ {result.get('message', '')}")
        else:
            self.n_ok += 1
            print(f"✓ {result.get('message', '')}")

        if not result.get("raised"):
            self.converted_rows.append({
                "subject": result["subject"],
                "experiment": result["experiment"],
                "session": int(result["session"]),
            })

    def record_unhandled(self, subject, experiment, session, exc, stages):
        """A worker died without returning a result dict."""
        self.n_fail += 1
        print(f"✗ FAILED (orchestrator): {subject} {experiment} {session}")
        print(exc)
        log = self.error_logs.get(experiment)
        if log is None:
            return
        log.record_result({
            "subject": str(subject),
            "experiment": experiment,
            "session": int(session),
            "files_written": [],
            "files_not_written": list(stages),
            "any_failure": True,
            "raised": True,
            "error_stage": "run",
            "error_type": type(exc).__name__,
            "error_message": " ".join(str(exc).splitlines()).strip(),
            "cmlreader_failure": cmlreader_involved(exc),
        })


def _run_serial(df_jobs, *, modality, root, overrides, force, brain_regions, tally):
    stages = registry.STAGES_BY_MODALITY[modality]
    total = len(df_jobs)
    for i, (_, row) in enumerate(df_jobs.iterrows(), start=1):
        subject, experiment, session = row["subject"], row["experiment"], int(row["session"])
        print(f"\n[{i}/{total}] {subject} {experiment} ses-{session}")
        try:
            result = run_job(
                subject, experiment, session,
                job_payload(row, modality, brain_regions),
                root, overrides, force,
            )
            tally.handle(result)
        except Exception as e:
            tally.record_unhandled(subject, experiment, session, e, stages)


def _run_parallel(df_jobs, *, modality, root, overrides, force, brain_regions, tally, dask_opts):
    import cmldask.CMLDask as da
    from dask.distributed import as_completed
    from distributed.diagnostics.plugin import WorkerPlugin

    class _BidsConvertPath(WorkerPlugin):
        """Make the repo importable on every worker (current + adaptive)."""

        def setup(self, worker):
            import sys as _sys
            if REPO_ROOT not in _sys.path:
                _sys.path.insert(0, REPO_ROOT)

    stages = registry.STAGES_BY_MODALITY[modality]

    log_dir = os.path.expanduser(dask_opts["log_directory"])
    os.makedirs(log_dir, exist_ok=True)

    client = da.new_dask_client_slurm(
        job_name=dask_opts["job_name"],
        memory_per_job=dask_opts["memory_per_job"],
        max_n_jobs=dask_opts["max_n_jobs"],
        threads_per_job=dask_opts["threads_per_job"],
        adapt=dask_opts["adapt"],
        log_directory=log_dir,
    )
    client.register_worker_plugin(_BidsConvertPath())
    # Ship conversion_error_log.py so pickled ConversionErrorLog objects
    # deserialize even before the path plugin has run on a new worker.
    client.upload_file(os.path.join(REPO_ROOT, "conversion_error_log.py"))

    jobs = [job_payload(row, modality, brain_regions) for _, row in df_jobs.iterrows()]
    n = len(df_jobs)

    futures = client.map(
        run_job,
        df_jobs["subject"].tolist(),
        df_jobs["experiment"].tolist(),
        [int(s) for s in df_jobs["session"].tolist()],
        jobs,
        [root] * n,
        [overrides] * n,
        [force] * n,
    )

    # Key futures back to their job so a dead worker is attributed correctly.
    future_to_job = dict(zip(futures, zip(
        df_jobs["subject"].tolist(),
        df_jobs["experiment"].tolist(),
        [int(s) for s in df_jobs["session"].tolist()],
    )))

    for future in as_completed(futures):
        try:
            tally.handle(future.result())
        except Exception as e:
            job = future_to_job.get(future)
            if job is None:
                tally.n_fail += 1
                print("✗ failed:", future.key)
                print(e)
            else:
                tally.record_unhandled(*job, e, stages)


def run_jobs(df_jobs, *, modality, root, overrides, force, serial,
             brain_regions=None, dask_opts=None, error_logs=None):
    """Convert every job in ``df_jobs``; return the tally.

    Returns a ``_Tally`` carrying counts, the rows that actually ran (for
    validation) and the per-experiment error logs, already flushed.
    """
    error_logs = error_logs if error_logs is not None else make_error_logs(df_jobs, root)
    tally = _Tally(error_logs)

    if serial:
        print("Running SERIALLY (no Dask)\n")
        _run_serial(df_jobs, modality=modality, root=root, overrides=overrides,
                    force=force, brain_regions=brain_regions, tally=tally)
    else:
        print("Running in PARALLEL via Slurm+Dask\n")
        _run_parallel(df_jobs, modality=modality, root=root, overrides=overrides,
                      force=force, brain_regions=brain_regions, tally=tally,
                      dask_opts=dask_opts or {})

    for log in error_logs.values():
        log.flush()

    print(f"\nDone. ok={tally.n_ok} skipped={tally.n_skip} fail={tally.n_fail}")
    return tally
