"""Per-task conversion error CSV.

Writes `bids_conversion_error_{task}.csv` at the base of each task root,
listing only jobs that had at least one failure. Rows are upserted by
(subject, session): re-running a subject replaces its prior row, and a
subject that now succeeds has its prior row dropped.

The CSV name is also added to `.bidsignore` so BIDS validation ignores it.
"""

import os
import traceback as _tb

import pandas as pd


CSV_COLUMNS = [
    "subject",
    "experiment",
    "session",
    "files_written",
    "files_not_written",
    "error_stage",
    "error_type",
    "error_message",
    "cmlreader_failure",
]

_BIDSIGNORE_PATTERN = "bids_conversion_error_*.csv"


def cmlreader_involved(exc: BaseException) -> bool:
    """True if any frame in the exception's traceback is inside the cmlreaders package."""
    tb = exc.__traceback__
    while tb is not None:
        mod = tb.tb_frame.f_globals.get("__name__", "")
        if mod == "cmlreaders" or mod.startswith("cmlreaders."):
            return True
        tb = tb.tb_next
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return cmlreader_involved(cause)
    return False


def _one_line(msg: str) -> str:
    return " ".join(str(msg).splitlines()).strip()


class ConversionErrorLog:
    """Collects per-job failure rows and flushes to a task CSV on demand.

    One instance per (root, task). The orchestrator calls `record_attempt()`
    for every job it tries (to build the upsert key set), then
    `record_failure()` only for jobs that failed (partial or full).
    `flush()` writes the merged CSV and updates `.bidsignore`.
    """

    def __init__(self, root: str, task: str):
        self.root = root
        self.task = task
        self._attempted: set[tuple[str, int]] = set()
        self._rows: list[dict] = []

    def record_attempt(self, subject: str, session):
        self._attempted.add((str(subject), int(session)))

    def record_result(self, result: dict):
        """Record a job result dict produced by a converter orchestrator.

        Expected keys: subject, experiment, session, files_written,
        files_not_written, any_failure, raised, error_stage, error_type,
        error_message, cmlreader_failure. Only jobs with any_failure or
        raised produce a row; others are recorded as attempts only.
        """
        self.record_attempt(result["subject"], result["session"])
        if not (result.get("any_failure") or result.get("raised")):
            return
        self._rows.append({
            "subject": str(result["subject"]),
            "experiment": result["experiment"],
            "session": int(result["session"]),
            "files_written": ",".join(result.get("files_written") or []),
            "files_not_written": ",".join(result.get("files_not_written") or []),
            "error_stage": result.get("error_stage") or "",
            "error_type": result.get("error_type") or "",
            "error_message": _one_line(result.get("error_message") or ""),
            "cmlreader_failure": bool(result.get("cmlreader_failure")),
        })

    def flush(self):
        os.makedirs(self.root, exist_ok=True)
        csv_path = os.path.join(self.root, f"bids_conversion_error_{self.task}.csv")

        if os.path.exists(csv_path):
            try:
                prior = pd.read_csv(csv_path, dtype={"subject": str})
            except Exception as e:
                print(f"WARNING: could not read prior {csv_path} ({e}); starting fresh")
                prior = pd.DataFrame(columns=CSV_COLUMNS)
        else:
            prior = pd.DataFrame(columns=CSV_COLUMNS)

        for col in CSV_COLUMNS:
            if col not in prior.columns:
                prior[col] = "" if col != "cmlreader_failure" else False
        prior = prior[CSV_COLUMNS]

        if len(prior) and self._attempted:
            prior["session"] = prior["session"].astype(int)
            keys = list(zip(prior["subject"].astype(str), prior["session"]))
            keep = [k not in self._attempted for k in keys]
            prior = prior.loc[keep].reset_index(drop=True)

        new_df = pd.DataFrame(self._rows, columns=CSV_COLUMNS) if self._rows else pd.DataFrame(columns=CSV_COLUMNS)
        merged = pd.concat([prior, new_df], ignore_index=True)

        if len(merged):
            merged["session"] = merged["session"].astype(int)
            merged = merged.sort_values(["subject", "session"]).reset_index(drop=True)

        merged.to_csv(csv_path, index=False)
        self._ensure_bidsignore()
        print(f"Wrote conversion error log: {csv_path} ({len(merged)} row(s))")
        return csv_path

    def _ensure_bidsignore(self):
        path = os.path.join(self.root, ".bidsignore")
        existing = ""
        if os.path.exists(path):
            with open(path) as f:
                existing = f.read()
        if _BIDSIGNORE_PATTERN not in existing.splitlines():
            with open(path, "a") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(_BIDSIGNORE_PATTERN + "\n")
