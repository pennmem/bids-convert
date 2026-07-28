"""Stage-gating plumbing shared by the scalp and intracranial converters.

Both converters run a fixed list of stages, each of which is skipped when its
outputs are already on disk unless the user asked to overwrite it. That
bookkeeping — outcome tracking, the report the orchestrator turns into a CSV
row, the failure policy, and the root-level BIDS files — is identical for both
and lives here. Only ``_stage_outputs_exist`` stays modality-specific, since
the two write genuinely different filenames.
"""

import json
import os

import mne_bids
import pandas as pd

_MNE_BIDS_CITATION = (
    "Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., "
    "Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., "
    "Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and "
    "Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into "
    "the BIDS format and facilitating their analysis. Journal of Open "
    "Source Software 4: (1896). https://doi.org/10.21105/joss.01896\n"
)

IEEG_BIDS_CITATION = (
    "Holdgraf, C., Appelhoff, S., Bickel, S., Bouchard, K., D'Ambrosio, S., "
    "David, O., ... & Hermes, D. (2019). iEEG-BIDS, extending the Brain "
    "Imaging Data Structure specification to human intracranial "
    "electrophysiology. Scientific Data, 6(1), 102. "
    "https://doi.org/10.1038/s41597-019-0105-7\n"
)

EEG_BIDS_CITATION = (
    "Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., "
    "Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, "
    "an extension to the brain imaging data structure for "
    "electroencephalography. Scientific Data, 6, 103. "
    "https://doi.org/10.1038/s41597-019-0104-8\n"
)


class StageGatedConverter:
    """Mixin providing stage bookkeeping, failure policy and root BIDS files.

    Subclasses provide ``ALL_STAGES``, ``_stage_outputs_exist(stage)``, and the
    ``root`` / ``experiment`` / ``overrides`` attributes.

    Stage outcomes: ``'ok'`` (wrote), ``'skipped'`` (outputs already exist),
    ``'failed'``, ``'not_run'`` (never reached). Files on disk = ok + skipped.
    """

    ALL_STAGES: tuple = ()

    # Filled in by the concrete converters; used for the root-level files.
    MODALITY_LABEL = "EEG"
    MODALITY_CITATION = _MNE_BIDS_CITATION

    # Failure policy. False (the default): any stage failure aborts the
    # session's conversion. True: failures are downgraded to [WARN] and the
    # remaining stages still run. Set by the orchestrator from --force.
    force = False

    # ------------------------------------------------------------------
    # Stage bookkeeping
    # ------------------------------------------------------------------
    def stage_report(self):
        """Summarize stage outcomes for the per-task conversion error CSV.

        Only ``'failed'`` flags the job as a failure — ``'not_run'`` just
        means the stage was never reached.
        """
        outcomes = getattr(self, 'stage_outcomes', {})
        written = [s for s in self.ALL_STAGES if outcomes.get(s) in ('ok', 'skipped')]
        not_written = [s for s in self.ALL_STAGES if outcomes.get(s) in ('failed', 'not_run', None)]
        any_failure = any(outcomes.get(s) == 'failed' for s in self.ALL_STAGES)
        return {
            'files_written': written,
            'files_not_written': not_written,
            'any_failure': any_failure,
            'error_stage': getattr(self, 'first_error_stage', None),
            'exception': getattr(self, 'first_exception', None),
        }

    def _mark_stage(self, stage, outcome, exc=None):
        if not hasattr(self, 'stage_outcomes'):
            self.stage_outcomes = {}
        self.stage_outcomes[stage] = outcome
        if outcome == 'failed' and exc is not None and not hasattr(self, 'first_exception'):
            self.first_exception = exc
            self.first_error_stage = stage

    def _should_run(self, stage):
        if self.overrides.get(stage, False):
            return True
        return not self._stage_outputs_exist(stage)

    def stages_to_run(self):
        """Stages this session would run right now.

        Pure path checks — no data is loaded. The orchestrator calls this
        before ``run()`` so a fully-converted session can be skipped outright.
        """
        return [s for s in self.ALL_STAGES if self._should_run(s)]

    def _report_stage_failure(self, stages, label, exc):
        """Mark ``stages`` failed and surface ``exc``.

        By default a stage failure is fatal: the exception is re-raised so the
        run aborts loudly. When ``force`` is set the failure is downgraded to
        a ``[WARN]`` line and the run continues (best-effort behavior).
        """
        for stage in stages:
            self._mark_stage(stage, 'failed', exc)
        msg = (f"{label} failed for {self.subject}, {self.experiment}, "
               f"session {self.session}: {exc}")
        if not self.force:
            raise RuntimeError(msg) from exc
        print(f"[WARN] {msg}")

    # ------------------------------------------------------------------
    # Root-level BIDS files
    # ------------------------------------------------------------------
    def _ensure_dataset_description(self):
        """Write a minimal BIDS-compliant dataset_description.json at the
        BIDS root if one isn't already there. Idempotent — never overwrites
        a customised version."""
        path = os.path.join(self.root, 'dataset_description.json')
        if os.path.exists(path):
            return
        os.makedirs(self.root, exist_ok=True)
        body = {
            "Name": f"{self.experiment} {self.MODALITY_LABEL} (CML pennmem/bids-convert)",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
            "Authors": ["[Unspecified]"],
        }
        with open(path, 'w') as f:
            json.dump(body, f, indent=4)
            f.write('\n')

    def _ensure_readme(self):
        """Write a minimal BIDS-compliant README at the BIDS root if missing.
        Idempotent — never overwrites a customised README."""
        path = os.path.join(self.root, 'README')
        if os.path.exists(path):
            return
        os.makedirs(self.root, exist_ok=True)
        with open(path, 'w') as f:
            f.write(
                "References\n"
                "----------\n"
                f"{self.MODALITY_CITATION}\n"
                f"{_MNE_BIDS_CITATION}"
            )

    # ------------------------------------------------------------------
    # scans.tsv
    # ------------------------------------------------------------------
    def _update_scans_tsv(self, data_file_path):
        """Append a row for the new recording to ``scans.tsv``.

        BIDS spec: ``scans.tsv`` lists every recording in the session with a
        path relative to the session directory. We append rather than
        overwrite so multiple acquisitions in the same session coexist.
        """
        scans_tsv = mne_bids.BIDSPath(
            subject=self.subject,
            session=str(self.session),
            suffix="scans",
            extension=".tsv",
            root=self.root,
        ).fpath
        # Path relative to the session directory.
        rel_path = os.path.relpath(data_file_path, scans_tsv.parent)
        new_row = pd.DataFrame([{"filename": rel_path}])
        if scans_tsv.exists():
            existing = pd.read_csv(scans_tsv, sep="\t")
            existing = existing[existing["filename"] != rel_path]
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            os.makedirs(scans_tsv.parent, exist_ok=True)
            combined = new_row
        combined.to_csv(scans_tsv, sep="\t", index=False)
