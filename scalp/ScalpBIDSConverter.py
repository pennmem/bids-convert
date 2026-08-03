import cmlreaders as cml
import mne
import numpy as np
import pandas as pd
import os
import sys
import json
import inspect
from glob import glob, escape as glob_escape
import shutil
import mne_bids
import cmlreaders as cml
import mne
import time
import pyedflib

# edf_digital_writer lives under intracranial/ — share it without a move.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intracranial")
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The repo root, for the shared `cli` engine package.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
)
from edf_digital_writer import (  # noqa: E402
    write_digital, resolve_edf_units, encode_egi_to_bdf,
)
from cli.stages import EEG_BIDS_CITATION, StageGatedConverter  # noqa: E402

# Montage cap files ship next to this module. Anchor on __file__ rather than
# the cwd — the entry point lives at the repo root, not in scalp/.
_SCALP_DIR = os.path.dirname(os.path.abspath(__file__))
MONTAGE_DIR = os.path.join(_SCALP_DIR, "montage_files")


class UnknownElectrodeCapError(Exception):
    pass
class MultiplePathsError(FileExistsError):
    pass

class ScalpBIDSConverter(StageGatedConverter):
    # EGI output container: "bdf" (per-channel int24, see _write_eeg_from_egi)
    # or "brainvision" (IEEE float32 .vhdr/.eeg/.vmrk). Change here in code.
    egi_output_format = "brainvision"

    # Manual raw-file pins, for sessions the automatic resolution in
    # locate_raw_files() cannot settle. Keyed by (subject_raw, session) ->
    # chosen basename (or absolute path); a list/tuple of names pins a
    # multi-part session in run order. Consulted first, ahead of
    # ``events.eegfile``, so it can override a bad alignment.
    #
    # This used to carry a long commented-out block of ambiguous sessions
    # waiting on a lab decision. Every one of them is now resolved
    # automatically from ``events.eegfile`` — the alignment pipeline already
    # recorded which recording(s) each session was synced to — so the block
    # is gone. Add an entry here only when eegfile is absent or known wrong.
    MANUAL_EEG_FILE = {}

    event_column_dict = {
        # Five columns present in every CourierReinstate1 session are omitted
        # deliberately: they are NiCLS/pointing-task leftovers that carry no
        # information here. Checked across all 203 sessions — 'classifier' is
        # always 'X'; 'correctPointingDirection' and 'submittedPointingDirection'
        # are always -999 (written as "n/a"); 'efr_mark' and 'finalrecalled'
        # only ever take -999 or 0.
        "CourierReinstate1": [
            'eegfile',
            'eogArtifact',
            'experiment',
            'intruded',
            'intrusion',
            'item',
            'itemno',
            'montage',
            'msoffset',
            'phase',
            'presX',
            'presZ',
            'protocol',
            'recalled',
            'rectime',
            'serialpos',
            'session',
            'store',
            'storeX',
            'storeZ',
            'subject',
            'trial',
        ],
        "ltpFR": ['subject', 'experiment', 'session', 'trial', 'task', 'item_name', 'item_num', 'recog_resp', 'recog_conf', 
                  'resp', 'answer', 'test_x', 'test_y', 'test_z', 'color_r', 'color_g', 'color_b', 'case', 'font'],
        "ltpFR2": ['subject', 'experiment', 'session', 'trial', 'item_name', 'item_num',
                   'list', 'answer', 'test_x', 'test_y', 'test_z'],
        "VFFR": ['subject', 'experiment', 'session', 'trial', 'item_name', 'item_num', 'too_fast'],
        "ValueCourier": [
            'actualvalue',
            'compensation',
            'eegfile',
            'eogArtifact',
            'experiment',
            'intruded',
            'intrusion',
            'item',
            'itemno',
            'itemvalue',
            'montage',
            'msoffset',
            'multiplier',
            'numingroupchosen',
            'phase',
            'playerrotY',
            'presX',
            'presZ',
            'primacybuf',
            'protocol',
            'recalled',
            'rectime',
            'recencybuf',
            'serialpos',
            'session',
            'store',
            'storeX',
            'storeZ',
            'storepointtype',
            'subject',
            'task',
            'trial',
            'valuerecall',
        ],
        "VCBehOnly": [
            'actualvalue',
            'compensation',
            'eegfile',
            'eogArtifact',
            'experiment',
            'intruded',
            'intrusion',
            'item',
            'itemno',
            'itemvalue',
            'montage',
            'msoffset',
            'multiplier',
            'numingroupchosen',
            'phase',
            'playerrotY',
            'presX',
            'presZ',
            'primacybuf',
            'protocol',
            'recalled',
            'rectime',
            'recencybuf',
            'serialpos',
            'session',
            'store',
            'storeX',
            'storeZ',
            'storepointtype',
            'subject',
            'task',
            'trial',
            'valuerecall',
        ],
         "VCFROP": [
            'avgvaluecorrect',
            'avgvalueguess',
            'eegfile',
            'eogArtifact',
            'experiment',
            'intruded',
            'intrusion',
            'item',
            'itemno',
            'itemvaluecorrect',
            'itemvalueguess',
            'montage',
            'msoffset',
            'multiplier',
            'numingroupchosen',
            'phase',
            'playerrotY',
            'presX',
            'presZ',
            'primacybuf',
            'protocol',
            'recalled',
            'rectime',
            'recencybuf',
            'serialpos',
            'session',
            'store',
            'storeX',
            'storeZ',
            'storepointtype',
            'subject',
            'task',
            'trial',
            'valuerecall',
        ],
    }
    # eegoffset,correctPointingDirection, eegfile, eogArtifact, finalrecalled, montage, msoffset, mstime, phase, protocol, submittedPointingDirection, type
    @staticmethod
    def _sanitize_bids_label(label):
        """Sanitize a label for use as a BIDS entity value.

        BIDS only allows alphanumerics in subject/session/task labels;
        underscores, dashes and slashes are reserved as field/file
        separators. Some on-disk subject IDs (e.g. ``LTP220_03``)
        contain underscores. We replace ``_`` with ``v`` ("visit") so
        the boundary between the base subject ID and the retest
        suffix is preserved (``LTP220_03`` → ``LTP220v03``), and drop
        any other non-alphanumeric characters. The original ID is kept
        as ``self.subject_raw`` for filesystem and CMLReader lookups.
        """
        out = []
        for ch in str(label):
            if ch.isalnum():
                out.append(ch)
            elif ch == "_":
                out.append("v")
        return "".join(out)

    # Stage outcomes: 'ok' (wrote), 'skipped' (outputs exist and not
    # overridden), 'failed', 'not_run'. Files-on-disk = ok + skipped.
    # Bookkeeping lives in cli.stages.StageGatedConverter.
    ALL_STAGES = ('behavioral', 'eeg', 'montage')
    MODALITY_LABEL = 'scalp EEG'
    MODALITY_CITATION = EEG_BIDS_CITATION

    def __init__(self, subject, experiment, session, root="/scratch/PEERS_BIDS/",
                 overrides=None, force=False):
        """Set up the conversion. Deliberately cheap and I/O-free — call
        ``run()`` to actually convert.

        force=False (default): any stage failure raises, aborting the run.
        force=True: failures are logged as warnings and the run continues.
        """
        self.force = force
        self.root = root
        # The on-disk / CMLReader subject label may contain characters
        # BIDS forbids in entity values (e.g. 'LTP220_03'). Keep the
        # original for data lookups, but expose a sanitized form to
        # everything that builds a BIDSPath.
        self.subject_raw = subject
        self.subject = self._sanitize_bids_label(subject)
        self.experiment = experiment
        self.session = session
        self.overrides = overrides or {}
        self.stage_outcomes = {s: 'not_run' for s in self.ALL_STAGES}

    def run(self):
        """Convert this session. Each stage runs only when ``_should_run``
        says so — i.e. its outputs are missing, or --overwrite named it."""
        self.stage_outcomes = {s: 'not_run' for s in self.ALL_STAGES}

        # Root-level required/recommended BIDS files. Idempotent — won't
        # overwrite a customised dataset_description.json or README.
        self._ensure_dataset_description()
        self._ensure_readme()

        self.load_subject_info()
        self.set_wordpool()

        # ---------- Behavioral ----------
        # When _should_run picks a stage (because outputs are missing OR
        # --overwrite named it), rewriting its files is the whole point, so
        # the writers always run with overwrite=True.
        if self._should_run('behavioral'):
            try:
                self.events = self.load_events(beh_only=True)
            except FileNotFoundError as exc:
                self._mark_stage('behavioral', 'failed', exc)
                print(f"[SKIP] No events found for {self.subject}, {self.experiment}, "
                      f"session {self.session}: {exc}")
                return
            try:
                self.make_event_descriptors()
                self.write_bids_beh(overwrite=True)
                self._mark_stage('behavioral', 'ok')
            except Exception as exc:
                self._report_stage_failure(
                    ['behavioral'], 'Behavioral conversion', exc)
                return
        else:
            self._mark_stage('behavioral', 'skipped')

        # ---------- Common EEG load (needed by both eeg + montage stages) ----------
        run_eeg = self._should_run('eeg')
        run_montage = self._should_run('montage')
        if not (run_eeg or run_montage):
            self._mark_stage('eeg', 'skipped')
            self._mark_stage('montage', 'skipped')
            return

        try:
            raw_filepaths = self.locate_raw_files()
        except Exception as exc:
            self._report_stage_failure(['eeg', 'montage'], 'EEG load', exc)
            return

        # No recording holds samples, and nothing claimed one should. That's a
        # real property of some sessions — the recording was aborted before the
        # first data block, or never started — not a conversion failure. The
        # behavioral output above stands; the orchestrator logs these to
        # bids_conversion_noeeg_*.csv rather than the error CSV.
        if not raw_filepaths:
            self.no_eeg_reason = (
                "no readable recording on disk and events are not aligned to "
                "one (events.eegfile blank)")
            print(f"[NO EEG] {self.subject}, {self.experiment}, session "
                  f"{self.session}: {self.no_eeg_reason}")
            self._mark_stage('eeg', 'no_eeg')
            self._mark_stage('montage', 'no_eeg')
            return

        # A session recorded in more than one part gets one BIDS run per part;
        # single-part sessions keep their unsuffixed filenames.
        multi_run = len(raw_filepaths) > 1
        eeg_ok, montage_ok = True, True

        for index, raw_filepath in enumerate(raw_filepaths, start=1):
            run = str(index) if multi_run else None
            try:
                self.raw_filepath = raw_filepath
                if self.raw_filepath.endswith(".bz2"):
                    self.unzip_raw_files()
                self.file_type = os.path.splitext(self.raw_filepath)[1]
                self.raw_file = self.load_scalp_eeg()
                self.set_montage()
                self.events = self.load_events(
                    eegfile=self.raw_filepath if multi_run else None,
                    sfreq=self.sfreq,
                )
                # events_descriptor is otherwise only built in the behavioral
                # stage; build it here too so the eeg/montage stages are
                # self-sufficient when behavioral is skipped (e.g. a re-run with
                # --overwrite eeg / --overwrite montage but existing behavioral output).
                self.make_event_descriptors()
            except Exception as exc:
                # Both downstream stages depend on the source EEG; fail together.
                eeg_ok = montage_ok = False
                self._report_stage_failure(
                    ['eeg', 'montage'],
                    f'EEG load{f" (run {run})" if run else ""}', exc)
                return

            # ---------- EEG (direct pyedflib write, no MNE round-trip) ----------
            if run_eeg:
                try:
                    self.write_bids_eeg(overwrite=True, run=run)
                except Exception as exc:
                    eeg_ok = False
                    self._report_stage_failure(
                        ['eeg'],
                        f'EEG conversion{f" (run {run})" if run else ""}', exc)

            # ---------- Montage (channels.tsv + electrodes.tsv only) ----------
            if run_montage:
                try:
                    self.write_bids_montage(overwrite=True, run=run)
                except Exception as exc:
                    montage_ok = False
                    self._report_stage_failure(
                        ['montage'],
                        f'Montage write{f" (run {run})" if run else ""}', exc)

        # A stage counts as 'ok' only when every run wrote. Failures were
        # already marked (and, unless --force, raised) by _report_stage_failure.
        if not run_eeg:
            self._mark_stage('eeg', 'skipped')
        elif eeg_ok:
            self._mark_stage('eeg', 'ok')
        if not run_montage:
            self._mark_stage('montage', 'skipped')
        elif montage_ok:
            self._mark_stage('montage', 'ok')

    # ------------------------------------------------------------------
    # Stage gating helpers. The shared half (_should_run, _mark_stage,
    # stage_report, _report_stage_failure) lives in StageGatedConverter;
    # only the filename layout below is scalp-specific.
    # ------------------------------------------------------------------
    def _bids_prefix(self):
        task = self.experiment.lower()
        return f'sub-{self.subject}_ses-{self.session}_task-{task}'

    def _session_eeg_dir(self):
        return os.path.join(self.root, f'sub-{self.subject}',
                            f'ses-{self.session}', 'eeg')

    def _session_beh_dir(self):
        return os.path.join(self.root, f'sub-{self.subject}',
                            f'ses-{self.session}', 'beh')

    def _stage_outputs_exist(self, stage):
        """True iff every expected file for `stage` already exists on disk."""
        prefix = self._bids_prefix()
        eeg_dir = self._session_eeg_dir()
        beh_dir = self._session_beh_dir()

        if stage == 'behavioral':
            # Behavioral lives under either beh/ (no EEG) or eeg/ (events.tsv).
            return any(os.path.exists(p) for p in (
                os.path.join(beh_dir, f'{prefix}_beh.tsv'),
                os.path.join(eeg_dir, f'{prefix}_events.tsv'),
            )) or bool(glob(os.path.join(eeg_dir, f'{prefix}_run-*_events.tsv')))
        if stage == 'eeg':
            # Glob rather than test one fixed name: a session recorded in two
            # parts writes {prefix}_run-1_eeg.* / _run-2_eeg.*, and each data
            # file needs its own sidecar.
            data_files = [p for ext in ('.edf', '.bdf', '.vhdr')
                          for p in glob(os.path.join(eeg_dir, f'{prefix}*_eeg{ext}'))]
            if not data_files:
                return False
            return all(
                os.path.exists(os.path.splitext(p)[0] + '.json')
                for p in data_files
            )
        if stage == 'montage':
            data_files = [p for ext in ('.edf', '.bdf', '.vhdr')
                          for p in glob(os.path.join(eeg_dir, f'{prefix}*_eeg{ext}'))]
            if data_files:
                # One channels.tsv per recording.
                channels_ok = all(
                    os.path.exists(
                        os.path.splitext(p)[0].rsplit('_eeg', 1)[0] + '_channels.tsv')
                    for p in data_files
                )
            else:
                channels_ok = os.path.exists(
                    os.path.join(eeg_dir, f'{prefix}_channels.tsv'))
            # space-* prefix on the electrodes file: any matching tsv satisfies.
            sub_ses_prefix = f'sub-{self.subject}_ses-{self.session}'
            electrodes_ok = bool(glob(os.path.join(
                eeg_dir, f'{sub_ses_prefix}_space-*_electrodes.tsv'
            )))
            return channels_ok and electrodes_ok
        raise ValueError(f"unknown stage: {stage!r}")

    # ------------------------------------------------------------------
    # Locating the source recording(s)
    # ------------------------------------------------------------------
    RECORDING_EXTS = (".raw", ".bdf", ".mff")

    def _source_eeg_dir(self):
        return (f"/data/eeg/scalp/ltp/{self.experiment}/"
                f"{self.subject_raw}/session_{self.session}/eeg")

    def _raw_events(self):
        """This session's CMLReader events, loaded once and cached.

        Both the eegfile lookup and ``load_events`` want the same frame, and
        load_events runs twice per session (behavioral + eeg stage); reading
        it once keeps the eegfile-based resolution free.
        """
        if not hasattr(self, '_raw_events_cache'):
            self._raw_events_cache = cml.CMLReader(
                self.subject_raw, self.experiment, self.session
            ).load('events')
        return self._raw_events_cache

    def eegfile_targets(self):
        """Basenames of the recording(s) this session's events are aligned to.

        ``events.eegfile`` is written by the lab's event-alignment pipeline and
        names the raw file it actually synced against, which makes it
        authoritative where a filename heuristic can only guess. It identifies
        recordings whose filename was mistyped at acquisition ('LP106 ...',
        'LTP254 ...' sitting under LTP258, 'Change This Filename.bdf'), picks
        the intended one out of a directory holding several real recordings,
        and points past a corrupt .mff to the intact .raw beside it.

        Returns [] when the events can't be loaded or the session was never
        aligned (eegfile blank throughout) — callers fall back to the glob
        heuristic. Order is first appearance, i.e. chronological, i.e. run
        order for a session recorded in more than one part.
        """
        try:
            events = self._raw_events()
        except Exception as exc:
            print(f"  eegfile unavailable for {self.subject_raw} "
                  f"session {self.session} ({type(exc).__name__}: {exc})")
            return []
        if 'eegfile' not in events.columns:
            return []
        seen, targets = set(), []
        for value in events['eegfile']:
            name = os.path.basename(str(value).strip())
            if name and name not in seen:
                seen.add(name)
                targets.append(name)
        return targets

    @classmethod
    def _recording_stem(cls, name):
        """Strip the container extension and any trailing part number.

        'LTP154_20150827_021348.2.raw' -> 'LTP154_20150827_021348', so a name
        that differs from what's on disk only by container still matches.
        """
        stem = name
        for _ in range(2):
            base, ext = os.path.splitext(stem)
            if ext.lower() in cls.RECORDING_EXTS or ext[1:].isdigit():
                stem = base
            else:
                break
        return stem

    @classmethod
    def _is_usable_recording(cls, path):
        """True if `path` actually holds samples.

        These session directories are full of husks that look like recordings:
        header-only BDFs (exactly 35328 bytes, zero data records, from
        recordings that were aborted before the first data block), zero-byte
        and partially-decompressed .raw files whose EGI header advertises more
        samples than the file contains, NetStation session files misnamed
        .raw, and .mff bundles exported without a signal bin. Selecting one of
        those is what turned a handful of empty sessions into IndexError /
        broadcast-shape crashes deep inside the readers.
        """
        path = path.rstrip("/")
        if path.endswith(".mff"):
            return bool(glob(os.path.join(path, "signal*.bin")))
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size == 0:
            return False
        if path.endswith(".bdf"):
            try:
                reader = pyedflib.EdfReader(path)
            except Exception:
                return False
            try:
                return reader.datarecords_in_file > 0
            finally:
                reader.close()
        # EGI simple binary: the header must parse, and the file must be long
        # enough to hold the n_samples it advertises.
        try:
            from mne.io.egi.egi import _read_header
            with open(path, "rb") as fid:
                header = _read_header(fid)
            row_bytes = ((header["n_channels"] + header["n_events"])
                         * header["dtype"].itemsize)
            needed = 36 + header["n_events"] * 4 + header["n_samples"] * row_bytes
        except Exception:
            return False
        return header["samp_rate"] > 0 and size >= needed

    def _resolve_eegfile(self, name, eeg_dir):
        """Map one events.eegfile basename onto a real path in `eeg_dir`.

        Exact match, then case-insensitive (plenty of recordings were saved as
        'ltp106 ...' / 'LTp296_...' / 'Ltp329_...'), then a same-stem glob so a
        name differing only in container resolves to its sibling.
        """
        exact = os.path.join(eeg_dir, name)
        if os.path.exists(exact):
            return exact
        try:
            entries = os.listdir(eeg_dir)
        except OSError:
            return None
        lowered = name.lower()
        for entry in entries:
            if entry.lower() == lowered:
                return os.path.join(eeg_dir, entry)

        stem = self._recording_stem(name)
        if not stem:
            return None
        siblings = [p for p in glob(os.path.join(eeg_dir, glob_escape(stem) + "*"))
                    if p.rstrip("/").endswith(self.RECORDING_EXTS)]
        if not siblings:
            return None
        # Prefer one that still holds data, and among those the container the
        # events named; otherwise fall through to whatever matched.
        usable = [p for p in siblings if self._is_usable_recording(p)] or siblings
        wanted = os.path.splitext(name)[1].lower()
        for path in sorted(usable):
            if path.lower().endswith(wanted):
                return path
        return sorted(usable)[0]

    def _heuristic_candidates(self, eeg_dir):
        """Legacy glob + subject-prefix selection, used only when eegfile is
        unavailable (no events, or a session that was never aligned).

        Filters out non-EEG and junk paths:
          * extension whitelist: keep only names ending exactly in .raw / .bdf
            / .mff (drops .raw.txt, .raw.txt.bz2, *_GAIN.txt, *_IMP*.txt);
          * drop empty .mff placeholder dirs (e.g. *_NEVER_EXPORTED.mff and
            empty wrong-subject stubs);
          * drop files whose basename doesn't start with the subject code
            (wrong-subject / prefixless exports) — matched case-insensitively,
            since the subject code was often typed in the wrong case;
          * drop malformed basenames containing a backslash (corrupt stubs).
        When a single real .mff coexists with .raw file(s) of the same session,
        the native .mff wins.
        """
        candidates = (glob(os.path.join(eeg_dir, "*.raw*"))
                      + glob(os.path.join(eeg_dir, "*.bdf*"))
                      + glob(os.path.join(eeg_dir, "*.mff*")))

        prefix = self.subject_raw.lower()

        def _keep(p):
            base = os.path.basename(p)
            if "\\" in base:                       # malformed / corrupt stub
                return False
            if not base.lower().startswith(prefix):  # wrong-subject / prefixless
                return False
            if not base.endswith(self.RECORDING_EXTS):  # sidecars
                return False
            if base.endswith(".mff") and os.path.isdir(p) and not os.listdir(p):
                return False                       # empty .mff placeholder
            return True

        real = [p for p in candidates if _keep(p)]

        # Dedupe by realpath, preserving order.
        seen, deduped = set(), []
        for p in real:
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                deduped.append(p)

        if len(deduped) <= 1:
            return deduped, candidates

        # >1 survivor: prefer a single native .mff over .raw file(s).
        mffs = [p for p in deduped if p.endswith(".mff")]
        non_mff = [p for p in deduped if not p.endswith(".mff")]
        if len(mffs) == 1 and all(p.endswith(".raw") for p in non_mff):
            return [mffs[0]], candidates
        return deduped, candidates

    def locate_raw_files(self):
        """Ordered list of the raw recording file(s) backing this session.

        Normally one path. A session whose recording was stopped and restarted
        mid-way returns one path per part in run order — those are real
        two-part sessions (events split cleanly between the parts, each with
        its own eegoffset origin), and ``run()`` writes them as separate BIDS
        runs rather than picking one and silently dropping half the session.

        Precedence:
          1. ``MANUAL_EEG_FILE``, for anything the lab has pinned by hand.
          2. ``events.eegfile`` — see ``eegfile_targets``. The container it
             names is honoured as-is: several sessions carry a corrupt .mff
             beside an intact .raw, and eegfile names the .raw.
          3. The legacy glob heuristic, for sessions with no usable events.

        Returns [] when nothing on disk holds samples. That is a real property
        of some sessions (aborted recordings), not an error — ``run()`` turns
        it into a 'no_eeg' outcome and still writes behavioral output.
        """
        eeg_dir = self._source_eeg_dir()

        pin = ScalpBIDSConverter.MANUAL_EEG_FILE.get(
            (self.subject_raw, int(self.session))
        )
        if pin is not None:
            names = [pin] if isinstance(pin, str) else list(pin)
            pinned = []
            for name in names:
                path = name if os.path.isabs(name) else os.path.join(eeg_dir, name)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"MANUAL_EEG_FILE pin does not exist: {path}")
                pinned.append(path)
            print(f"Raw File (manual pin): {pinned}")
            return pinned

        targets = self.eegfile_targets()
        if targets:
            resolved, missing = [], []
            for name in targets:
                path = self._resolve_eegfile(name, eeg_dir)
                if path is None:
                    missing.append(name)
                elif self._is_usable_recording(path):
                    resolved.append(path)
                else:
                    missing.append(f"{name} (no samples)")
            if resolved:
                note = f" [unresolved: {missing}]" if missing else ""
                print(f"Raw File(s) from events.eegfile: {resolved}{note}")
                return resolved
            # eegfile named files we can't use. Don't silently guess something
            # else — that risks converting the wrong recording.
            raise FileNotFoundError(
                f"events.eegfile for {self.subject_raw} session {self.session} "
                f"names {targets}, none of which resolve to a readable "
                f"recording in {eeg_dir} ({missing})")

        deduped, candidates = self._heuristic_candidates(eeg_dir)
        usable = [p for p in deduped if self._is_usable_recording(p)]
        if not usable:
            print(f"No usable raw EEG for {self.subject_raw} session "
                  f"{self.session} in {eeg_dir} "
                  f"(candidates before filtering: {candidates})")
            return []
        if len(usable) > 1:
            raise MultiplePathsError(
                f"Multiple real EEG files for {self.subject_raw} "
                f"session {self.session} and no events.eegfile to choose "
                f"between them; pin one in MANUAL_EEG_FILE: {usable}")
        print(f"Raw File Found (heuristic): {usable[0]}")
        return usable

    def locate_raw_file(self):
        """The first (usually only) recording for this session.

        Kept for callers that predate multi-part sessions; prefer
        ``locate_raw_files``.
        """
        paths = self.locate_raw_files()
        if not paths:
            raise FileNotFoundError(
                f"No usable raw EEG file for {self.subject_raw} "
                f"session {self.session}")
        return paths[0]


    def load_scalp_eeg(self):
        if self.file_type == ".bdf":
            raw = mne.io.read_raw_bdf(self.raw_filepath, stim_channel='Status', preload=False)
        elif self.file_type in (".raw", ".mff"):
            raw = mne.io.read_raw_egi(self.raw_filepath, preload=False)
        else:
            raise ValueError("Unknown File Extension:", self.file_type)
        self.sfreq = raw.info['sfreq']
        self.recording_start = raw.info['meas_date']
        return raw
    
    def unzip_raw_files(self):
        output_path = os.path.splitext(self.raw_filepath.replace(' ', '_'))[0]
        if not os.path.exists(output_path):
            success = os.system(f"bunzip2 -k '{self.raw_filepath}'") == 0
            if success:
                shutil.move(self.raw_filepath, output_path)
                self.raw_filepath = output_path
    
    def set_montage(self):
        self.eeg_sidecar = {"PowerLineFrequency":60.0}
        if self.file_type == ".bdf":
            self.raw_file.set_channel_types({'EXG1':'eog', 'EXG2':'eog', 'EXG3':'eog', 'EXG4':'eog',
                                             'EXG5':'misc', 'EXG6':'misc', 'EXG7':'misc', 'EXG8':'misc'})
            self.raw_file.set_montage('biosemi128', on_missing="warn")
            self.eeg_sidecar["Manufacturer"] = "BioSemi"
            self.eeg_sidecar["CapManufacturer"] = "BioSemi"
        elif self.file_type in (".raw", ".mff", ".edf"):
            self.eeg_sidecar["Manufacturer"] = "EGI"
            self.eeg_sidecar["CapManufacturer"] = "EGI"
            self.eeg_sidecar["EEGReference"] = "Cz"
            # Some EGI nets don't expose the vertex reference as an explicit
            # E129 channel; only rename when present (avoids a hard ValueError).
            if 'E129' in self.raw_file.ch_names:
                self.raw_file.rename_channels({'E129': 'Cz'})
            if "sync" in self.raw_file.ch_names:
                # GSN 200 v2.1 caps
                self.eeg_sidecar["CapManufacturersModelName"] = "Geodisic Sensor Net 200 v2.1"
                mon = mne.channels.read_custom_montage(
                    os.path.join(MONTAGE_DIR, "egi128_GSN_200.sfp"))
                self.raw_file.set_montage(mon, on_missing="warn")
                self.raw_file.set_channel_types({'E8': 'eog', 'E26': 'eog', 'E126': 'eog', 
                                            'E127': 'eog'})
                ## peripheral electrodes tend to flip up during the session, and get poor signal
                ## peripheral [127 126 17 128 125 120 44 49 56 63 69 74 82 89 95 100 108 114]
            else:
#             elif "DI15" in self.raw_file.ch_names:
                # GSN HydroCel caps
                self.eeg_sidecar["CapManufacturersModelName"] = "HydroCel Geodisic Sensor Net"
                mon = mne.channels.read_custom_montage(
                    os.path.join(MONTAGE_DIR, "egi128_GSN_HydroCel.sfp"))
                self.raw_file.set_montage(mon, on_missing="warn")
                self.raw_file.set_channel_types({'E8': 'eog', 'E25': 'eog', 'E126': 'eog',
                           'E127': 'eog'})
                ## peripheral [127 126]
#             else:
#                 raise UnknownElectrodeCapError
        else:
            raise UnknownElectrodeCapError
        self.montage = self.raw_file.get_montage()
            
    def set_wordpool(self):
        if self.experiment=='ltpFR':
            if self.subject_raw <= 'LTP159':
                self.wordpool_file = "wordpools/wasnorm_wordpool.txt"
            else:
                self.wordpool_file = "wordpools/wasnorm_wordpool_less_exclusions.txt"
        elif np.isin(self.experiment, ["ltpFR2", "VFFR"]):
            self.wordpool_file = "wordpools/wasnorm_wordpool_576.txt"
        elif np.isin(self.experiment, ["ValueCourier"]):
            self.wordpool_file = "wordpools/valuecourier_wordpool.txt"
        elif np.isin(self.experiment, ["VCBehOnly"]):
            self.wordpool_file = f"/data/eeg/scalp/ltp/VCBehOnly/{self.subject_raw}/wordpool.txt"
        elif np.isin(self.experiment, ["VCFROP"]):
            self.wordpool_file = f"/data/eeg/scalp/ltp/VCFROP/{self.subject_raw}/wordpool.txt"
        elif np.isin(self.experiment, ["CourierReinstate1"]):
            # Courier delivers objects rather than words; the per-subject item
            # pool is all_items.txt (companion all_stores.txt lists the stores).
            self.wordpool_file = f"/data/eeg/scalp/ltp/CourierReinstate1/{self.subject_raw}/all_items.txt"
        else:
            raise Exception("Wordpool not known for this experiment.")
    
    def load_events(self, beh_only=False, eegfile=None, sfreq=None):
        """Build the BIDS events table for this session.

        ``eegfile`` restricts the table to the events aligned to one
        recording, for sessions recorded in more than one part. Each part
        carries its own eegoffset origin, so onsets are computed against that
        part's own ``sfreq`` — pass it in rather than relying on
        ``self.sfreq``, which tracks whichever part was loaded last. Events
        with a blank eegfile were never aligned to any recording and belong in
        the behavioral table only, so they are dropped here.
        """
        events = self._raw_events().copy()
        if eegfile is not None:
            aligned = events['eegfile'].map(lambda p: os.path.basename(str(p).strip()))
            events = events[aligned == os.path.basename(eegfile)]
        events = events.rename(columns={"eegoffset":"sample", "type":"trial_type"})
        ## math distractor
        if "test" in events.columns:
            events[["test_x", "test_y", "test_z"]] = events['test'].apply(pd.Series)
            events = events.drop(columns=["test"])
        if "font" in events.columns:
            events["font"] = events['font'].apply(os.path.basename)
        if beh_only:
            standard_cols = ["mstime", "trial_type", 'stim_file']
            events["mstime"] = events["mstime"] - events["mstime"].iloc[0]
        else:
            events['onset'] = events['sample'] / (sfreq or self.sfreq)
            events['duration'] = "n/a"
            standard_cols = ['onset', 'duration', "trial_type", "sample", 'stim_file']
        events['stim_file'] = np.where(events.trial_type.str.contains("WORD"), self.wordpool_file, "n/a")
        events = events.fillna("n/a")
        events = events.replace("", "n/a")
        events = events.replace("-999", "n/a")
        events = events.replace(-999, "n/a")
        cols_to_include = ScalpBIDSConverter.event_column_dict[self.experiment]
        cols_to_include = [col for col in cols_to_include if col in events.columns]
        events = events[standard_cols + cols_to_include]
        return events
    
    def make_event_descriptors(self):
        descriptions = {
            "SESS_START": "Beginning of session.",
            "SESS_END": "End of session.",
            "WORD": "Word presentation onset.",
            "WORD_ON": "Word presentation onset.",
            "WORD_OFF": "Word presentation offset.",
            "REC_START": "Recall phase begins.",
            "REC_END": "Recall phase ends.",
            "REC_STOP": "Recall phase ends.",
            "REST_REWET": "Mid-session break to rewet scalp cap.",
            "REC_WORD": "Recalled word, onset of speech (during free recall).",
            "REC_WORD_VV": "Vocalization (during free recall).",
            "FFR_REC_WORD": "Recalled word, onset of speech (during final free recall).",
            "FFR_REC_WORD_VV": "Vocalization (during final free recall).",
            "RECOG_CONF": "Confidence judgement for recognition.",
            "KEY_MSG": "Warning message telling the subject they pressed one of the keys corresponding to the wrong judgment.",
            "RECOG_LURE": "Recognition item that is a lure.",
            "RECOG_RESP": "Recognition response ('pess', 'po').",
            "RECOG_RESP_VV": "Vocalization (during recognition).",
            "RECOG_TARGET": "Recognition item that is a target.",
            "SLOW_MSG": "Warning message telling the subject that they took too long to make their judgment about a word.",
            "START": "Beginning of math distractor phase.",
            "STOP": "End of math distractor phase.",
            "PROB": "Math problem presentation onset.",
            "PRACTICE_WORD": "Word presentation onset (in a practice list)",
            "PRACTICE_WORD_OFF": "Word presentation offset (in a practice list)",
            "FFR_START": "Beginning of final free recall phase.",
            "FFR_END": "End of final free recall phase.",
            "FFR_STOP": "End of final free recall phase.",
            "DISTRACTOR": "Beginning of math distractoor phase.",
            'PRACTICE_REC_START': "Recall phase begins (in a practice list).", 
            'PRACTICE_REC_STOP': "Recall phase begins (in a practice list).",
            'PRACTICE_REC_WORD': "Recalled word, onset of speech (during practice list free recall).", 
            'PRACTICE_REC_WORD_VV': "Vocalization (during practice list free recall).", 
            "COUNTDOWN": "Initiate countdown before encoding.",
            "BREAK_START": "Start mid-session break to rewet scalp cap.",
            "BREAK_STOP": "Stop mid-session break to rewet scalp cap, same as BREAK_END.",
            "BREAK_END": "Stop mid-session break to rewet scalp cap, same as BREAK_STOP.",
                # ---------------- ValueCourier-specific trial types ----------------
            "store mappings": "Display or logging of store-to-location/value mappings in the task.",
            "VIDEO_START": "Start of instructional or task-related video.",
            "VIDEO_STOP": "End of instructional or task-related video.",
            "TL_START": "Start of a temporal-learning or timeline block.",
            "TL_END": "End of a temporal-learning or timeline block.",
            "PRACTICE_DELIVERY_START": "Start of a practice delivery trial in the task.",
            "PRACTICE_DELIVERY_END": "End of a practice delivery trial in the task.",
            "PRACTICE_VALUE_RECALL": "Value recall response during a practice trial.",
            "POINTER_ON": "Onset of the pointing/selection cursor during navigation or choice.",
            "TRIAL_START": "Start of a trial.",
            "TRIAL_END": "End of a trial.",
            "VALUE_RECALL": "Subject’s value recall response for the current list or item.",
            "ITEM_VALUE_RECALL": "Subject’s value recall response for an individual item (VCFROP).",
            "PRACTICE_ITEM_VALUE_RECALL": "Subject’s value recall response for an individual item (in a practice list).",
            "AVG_VALUE_RECALL": "Subject’s recalled estimate of the average value for the current list/delivery day (VCFROP).",
            "PRACTICE_AVG_VALUE_RECALL": "Subject’s recalled estimate of the average value for the current list/delivery day (in a practice list).",
            "FINAL_COMPENSATION": "Event marking display and computation of final monetary compensation.",
            "EFR_MARK": "Externalized free recall: participants say the items the remember outloud.",
            # ---------------- CourierReinstate1-specific trial types ----------------
            "REINSTATEMENT": "Object reinstatement: the store where a previously delivered item was "
                             "presented comes into view while the courier navigates to the next "
                             "delivery. Logged repeatedly (~6 Hz) for as long as the store is in "
                             "view; the 'item' column names the previously delivered item seen.",
            "MUSIC_VIDEOS_STOP": "End of the music-video filler block presented between trials.",
        }
        HED = {
            "onset": {
                "Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file. ",
            },
            "subject": {
                "LongName": "Subject ID",
                "Description": "The string identifier of the subject, e.g. LTP123",

            },
            "session": {
                "Description": "The session number (1 - 24)."
            },
            "trial": {
                "LongName": "Trial Number",
                "Description": "Word list (1-24) during which the event occurred. Trial <= 0 indicates practice list.",
            },
            "trial_type": {
                "LongName": "Event category",
                "Description": "Indicator of type of task action that occurs at the marked time",
                "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()},
            },
            "item_name": {
                "Description": "The word being presented or recalled in a WORD or REC_WORD event."
            },
            "item_num": {
                "LongName": "Item number",
                "Description": "The ID number of the presented or recalled word in the word pool. -1 represents an intrusion or vocalization."
            },
            'task': {
                "LongName": "Task type",
                "Description": "Type of judgment made on a presented word",
                "Levels": {
                    -1: "Control, no task, just read the word",
                    0: "Size",
                    1: "Animacy"
                }
            },
            'recog_conf': {
                "LongName": "Confidence rating",
                "Description": "Confidence rating of recognition response, 1 (very low confidence) - 5 (complete confidence)"
            }, 
            'recog_resp': {
                "LongName": "Recognition response",
                "Description": "1 (old) or 0 (new) response in the recognition period.",
                "Levels": {0: "new", 1: "old"}
            },
            'resp': {
                "LongName": "Judgement response",
                "Description": "Judgement response during the study period for size/animacy. Responses >2 are wrong key presses.",
                "Levels": {-1: "control", 0: "small/nonliving", 1: "big/living"}
            },
            'answer': {
                "LongName": "Math problem response",
                "Description": "Answer to problem with form X + Y + Z = ?",
            },
            'test_x': {
                "LongName": "Math problem X",
                "Description": "X component of problem with form X + Y + Z = ?",
            },
            'test_y': {
                "LongName": "Math problem Y",
                "Description": "Y component of problem with form X + Y + Z = ?",
            },
            'test_z': {
                "LongName": "Math problem Z",
                "Description": "Z component of problem with form X + Y + Z = ?",
            }, 
            'color_b': {
                "LongName": "Blue",
                "Description": "Blue RGB value in [0, 255]",
            },
            'color_g': {
                "LongName": "Green",
                "Description": "Green RGB value in [0, 255]",
            },
            'color_r': {
                "LongName": "Red",
                "Description": "Red RGB value in [0, 255]",
            },
            'case': {
                "LongName": "Letter case",
                "Description": "Case of text presented on screen",
                "Levels": {'upper': 'upper case', 'lower': 'lower case'},
            },
            'font': {
                "LongName": "Word font",
                "Description": "File name for font of text presented on screen, found in stimuli/fonts.",
            },
            'too_fast':{
                "LongName": "'Too fast' message displayed",
                "Description": "Subject recalled word too quickly, warning displayed on screen."
            },
            
            # ---- ValueCourier-specific fields ----
            'item': {
                "LongName": "Item string",
                "Description": "The item identifier (often the string name of the object/value being judged)."
            },
            'itemno': {
                "LongName": "Item number",
                "Description": "Numeric ID for the item within the ValueCourier item set."
            },
            'itemvalue': {
                "LongName": "Displayed item value",
                "Description": "Point or monetary value displayed for the item on that trial."
            },
            'actualvalue': {
                "LongName": "True average tip value",
                "Description": "True average tip value of items in a delivery day"
            },
            # ---- VCFROP-specific value fields (correct vs. guessed) ----
            'itemvaluecorrect': {
                "LongName": "True item value",
                "Description": "True (correct) point or monetary value of the item on that trial, against which the subject's guess is scored."
            },
            'itemvalueguess': {
                "LongName": "Guessed item value",
                "Description": "Subject's guessed point or monetary value for the item on that trial."
            },
            'avgvaluecorrect': {
                "LongName": "True average value",
                "Description": "True (correct) average tip value of items in a delivery day, against which the subject's average-value guess is scored."
            },
            'avgvalueguess': {
                "LongName": "Guessed average value",
                "Description": "Subject's guessed average tip value of items in a delivery day."
            },
            'compensation': {
                "LongName": "Compensation",
                "Description": "Compensation to participant at the end of the session."
            },
            'intruded': {
                "LongName": "Intruded flag",
                "Description": "1 if the response was an intrusion relative to the current list/context, 0 otherwise."
            },
            'intrusion': {
                "LongName": "Intrusion item number",
                "Description": "Identifier of the intruded item when an intrusion occurs, -1 otherwise."
            },
            'multiplier': {
                "LongName": "Value multiplier",
                "Description": "Multiplicative factor applied to the max tip value (10$) which is used to calculate compensation."
            },
            'numingroupchosen': {
                "LongName": "Number in group chosen",
                "Description": "In temporal value association condition, the list of items is split in half and the each half has a high or low value. Number in group chosen describes the number of high items are in the high side of the list and vice versa for the low side."
            },
            'playerrotY': {
                "LongName": "Player rotation (Y axis)",
                "Description": "Rotation angle of the player avatar around the Y axis in the virtual environment."
            },
            'presX': {
                "LongName": "Presentation X coordinate",
                "Description": "X coordinate of the item at presentation in the virtual environment."
            },
            'presZ': {
                "LongName": "Presentation Z coordinate",
                "Description": "Z coordinate of the item at presentation in the virtual environment."
            },
            'primacybuf': {
                "LongName": "Primacy buffer length",
                "Description": "Length of the buffer in the beginning of the item list during the temporal association condition which is not split into halves."
            },
            'recencybuf': {
                "LongName": "Recency buffer length",
                "Description": "Length of the buffer at the end of the item list during the temporal association condition which is not split into halves."
            },
            'recalled': {
                "LongName": "Recalled flag",
                "Description": "1 if the item was successfully recalled, 0 otherwise."
            },
            'serialpos': {
                "LongName": "Serial position",
                "Description": "Serial position of the item within the list."
            },
            'store': {
                "LongName": "Store identifier",
                "Description": "Identifier of the store/location in the ValueCourier environment where the item is presented."
            },
            'storeX': {
                "LongName": "Store X coordinate",
                "Description": "X coordinate of the store in the virtual environment."
            },
            'storeZ': {
                "LongName": "Store Z coordinate",
                "Description": "Z coordinate of the store in the virtual environment."
            },
            'storepointtype': {
                "LongName": "Store point type",
                "Description": "Type of point or reward structure associated with the store."
            },
            'valuerecall': {
                "LongName": "Value recall response",
                "Description": "Subject’s recalled value or judgment about the average list value."
            },
            "msoffset": {
                "LongName": "Event offset (ms)",
                "Description": "Event time offset in milliseconds relative to a task-specific reference (typically the start of the trial or session).",
                "Units": "ms",
            },
            # "eegoffset": {
            #     "LongName": "EEG sample offset",
            #     "Description": "Sample index of the event relative to the start of the EEG recording. For convenience, this is often duplicated in the 'sample' column.",
            # },
            "rectime": {
                "LongName": "Recall Time",
                "Description": "Time when item was recalled.",
            },
            "eegfile": {
                "LongName": "EEG file",
                "Description": "Filename of the original EEG recording from which this BIDS dataset was derived.",
            },
            "eogArtifact": {
                "LongName": "EOG artifact flag",
                "Description": "Indicator of whether the event or trial was contaminated by EOG artifact (1 = artifact, 0 = clean, or 'n/a' if not assessed).",
            },
            "montage": {
                "LongName": "EEG montage name",
                "Description": "Name or identifier of the electrode montage or sensor net used for the recording (e.g., BioSemi 128, GSN-HydroCel-128).",
            },
            "experiment": {
                "LongName": "Experiment name",
                "Description": "Name of the experiment/task this session belongs to (e.g. 'VCFROP', 'ValueCourier', 'CourierReinstate1').",
            },
            "phase": {
                "LongName": "Task phase",
                "Description": "Phase of the trial during which the event occurred (e.g. encoding/delivery, recall, practice).",
            },
            "protocol": {
                "LongName": "CML protocol name",
                "Description": "High-level protocol identifier from the CML system (e.g., 'ltp', 'r1').",
            },
        }
        self.events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        
    
    def load_subject_info(self):
        #TODO
        pass
    
    def write_bids_beh(self, overwrite=True):
        task_name = self.experiment.lower() 
        # events = self.load_events(beh_only=True)
        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                          session=str(self.session),
                                          task=task_name,
                                          datatype="beh",
                                          suffix="beh",
                                          extension=".tsv",
                                          root=self.root)
        os.makedirs(bids_path.directory, exist_ok=True)
        self.events.to_csv(bids_path.fpath, sep="\t", index=False)
        with open(bids_path.update(suffix="beh", extension=".json").fpath, "w") as f:
            json.dump(fp=f, obj = self.events_descriptor)
    
    def write_bids_eeg(self, overwrite=True, run=None):
        """Write the EEG file as a bit-exact digital copy of the source.

        BDF inputs are read with ``pyedflib`` and written back via
        ``write_digital(container="BDF")`` — the on-disk digital int24
        samples and per-channel ``(pmin, pmax, dmin, dmax, dim)`` headers
        match the source byte-for-byte. EGI ``.raw`` / ``.mff`` inputs
        must still be decoded by MNE (pyedflib cannot read EGI), but the
        EDF write goes straight through pyedflib — no
        ``mne.export.export_raw`` round-trip, no ``mne_bids.write_raw_bids``.

        ``run`` adds a ``run-<n>`` entity, for a session recorded in more than
        one part. Left None for the single-recording case so the vast majority
        of filenames stay unsuffixed.
        """
        task_name = self.experiment.lower()
        bids_path = mne_bids.BIDSPath(
            subject=self.subject, session=str(self.session),
            task=task_name, run=run, datatype="eeg", root=self.root,
        )

        if self.file_type == ".bdf":
            out_path = self._write_eeg_from_bdf(bids_path)
        elif self.egi_output_format == "brainvision":
            out_path = self._write_eeg_from_egi_brainvision(bids_path)
        else:
            out_path = self._write_eeg_from_egi(bids_path)

        # Channels / electrodes / coordsystem — unchanged.
        self.write_bids_montage(overwrite=overwrite, run=run)

        # Sidecar JSON — write directly (no mne_bids.update_sidecar_json).
        sidecar_path = bids_path.copy().update(
            suffix="eeg", extension=".json",
        ).fpath
        with open(sidecar_path, "w") as f:
            json.dump(self.eeg_sidecar, f, indent=2)

        # Events.
        events_tsv = bids_path.copy().update(
            suffix="events", extension=".tsv",
        ).fpath
        self.events.to_csv(events_tsv, sep="\t", index=False)
        events_json = bids_path.copy().update(
            suffix="events", extension=".json",
        ).fpath
        with open(events_json, "w") as f:
            json.dump(self.events_descriptor, f)

        # scans.tsv (shared helper, see cli.stages).
        self._update_scans_tsv(out_path)

    def _write_eeg_from_bdf(self, bids_path):
        """True bit-exact copy: pyedflib → pyedflib, no MNE."""
        src_bdf = self.raw_filepath
        out_path = bids_path.copy().update(
            suffix="eeg", extension=".bdf",
        ).fpath
        os.makedirs(out_path.parent, exist_ok=True)

        f = pyedflib.EdfReader(str(src_bdf))
        try:
            labels = list(f.getSignalLabels())
            sfreq = float(f.getSampleFrequency(0))
            n_samp = f.getNSamples()[0]
            data_int = np.empty((len(labels), n_samp), dtype=np.int32)
            for i in range(len(labels)):
                data_int[i] = f.readSignal(i, digital=True).astype(np.int32)
        finally:
            f.close()

        signal_units, _ = resolve_edf_units(
            labels,
            source_edf_path=str(src_bdf),
            conversion_to_V=None,
            container="BDF",
            data_for_fallback=data_int,
        )
        write_digital(
            str(out_path), labels, data_int, sfreq, signal_units,
            container="BDF",
        )
        return out_path

    def _scrub_nonfinite(self, raw):
        """Zero out non-finite samples, recording which channels were hit.

        A run of EGI sessions from 2013 decodes with a handful of ±inf samples
        on one channel in the first few seconds — a saturated electrode at
        recording onset, not file corruption (LTP244 ses-18: 49 samples on
        E124 between samples 1038 and 2190; LTP247 ses-14: 76 samples, same
        channel, same window). pybv refuses to write them at all, so the whole
        session used to fail. Zero-filling the affected samples lets the
        session convert; ``write_bids_montage`` then marks those channels
        ``status=bad`` in channels.tsv so the substitution stays visible
        downstream rather than passing as clean signal.

        Populates ``self.nonfinite_channels`` (used by write_bids_montage) and
        returns the raw, modified in place.
        """
        self.nonfinite_channels = {}
        data = raw.get_data()
        bad = ~np.isfinite(data)
        if not bad.any():
            return raw
        for ch_index in np.unique(np.nonzero(bad)[0]):
            samples = np.nonzero(bad[ch_index])[0]
            self.nonfinite_channels[raw.ch_names[ch_index]] = (
                len(samples), int(samples[0]), int(samples[-1]))
        # get_data() may hand back a view or a copy depending on preload;
        # write through _data so the change lands on the object we export.
        raw.load_data()
        raw._data[bad] = 0.0
        for name, (count, first, last) in self.nonfinite_channels.items():
            print(f"[WARN] {self.subject} {self.experiment} ses-{self.session}: "
                  f"{count} non-finite samples on {name} (samples {first}-{last}) "
                  f"zero-filled; channel marked bad")
        return raw

    def _write_eeg_from_egi(self, bids_path):
        """EGI .raw / .mff → BDF.

        MNE decodes the source to Volts and we requantize to int32 over
        BDF's full 24-bit range.

        Stim/sync channels (sync, D255, DIN1, ...) are dropped —
        channels.tsv already lists only the eeg+eog subset.
        """
        out_path = bids_path.copy().update(
            suffix="eeg", extension=".bdf",
        ).fpath
        os.makedirs(out_path.parent, exist_ok=True)

        raw = self._scrub_nonfinite(self.raw_file.copy().pick(['eeg', 'eog']))
        labels = list(raw.ch_names)
        sfreq = float(raw.info['sfreq'])

        data_v = raw.get_data()
        data_int, phys_min, phys_max, signal_units = encode_egi_to_bdf(
            data_v, labels=labels, dim="uV", container="BDF",
        )
        peak = float(np.max(np.abs(data_v))) or 1e-6
        print(
            f"  EGI requantize path: peak={peak:.3e} V, "
            f"per-channel min quantization over 24-bit BDF range "
            f"({self.subject} {self.experiment} ses-{self.session})"
        )
        write_digital(
            str(out_path), labels, data_int, sfreq, signal_units,
            container="BDF",
        )

        return out_path

    def _write_eeg_from_egi_brainvision(self, bids_path):
        """EGI .raw / .mff → BrainVision (.vhdr/.eeg/.vmrk), IEEE float32.

        MNE decodes the source to Volts and pybv writes them straight to
        float32 — no integer requantization and no 8-char EDF/BDF header
        gain truncation, so the round-trip is at the float32 floor
        (~1e-7 relative).

        Same channel subset as the BDF path: stim/sync channels are
        dropped so channels.tsv (also keyed off eeg+eog) stays consistent.
        """
        out_path = bids_path.copy().update(
            suffix="eeg", extension=".vhdr",
        ).fpath
        os.makedirs(out_path.parent, exist_ok=True)

        raw = self._scrub_nonfinite(self.raw_file.copy().pick(['eeg', 'eog']))
        peak = float(np.max(np.abs(raw.get_data()))) or 1e-6
        print(
            f"  EGI BrainVision path: peak={peak:.3e} V, float32 "
            f"({self.subject} {self.experiment} ses-{self.session})"
        )
        mne.export.export_raw(
            str(out_path), raw, fmt="brainvision", overwrite=True,
        )

        return out_path

    def write_bids_montage(self, overwrite=True, run=None):
        """Write only ``*_channels.tsv``, ``*_electrodes.tsv`` and
        ``*_coordsystem.json`` for this session — without re-encoding the
        EEG. Mirrors the intracranial converter's
        ``write_BIDS_channels`` / ``write_BIDS_electrodes`` flow so that
        a ``--overwrite montage`` rerun can fix sidecar files in place.

        Note: the on-disk EDF must already match the source recording's
        bare channel names (i.e. it must have been written with the
        ``add_ch_type=False`` fix). Older EDFs from the buggy converter
        carry ``"EEG E1"`` etc.; running this method against those will
        produce a bare-name channels.tsv that no longer matches the EDF.
        Pair with ``--overwrite eeg`` for affected sessions.
        """
        from mne_bids.write import _channels_tsv, _write_dig_bids

        task_name = self.experiment.lower()
        bids_path = mne_bids.BIDSPath(
            subject=self.subject, session=str(self.session),
            task=task_name, run=run, datatype="eeg", root=self.root,
        )
        os.makedirs(bids_path.directory, exist_ok=True)

        # For EGI (.raw / .mff) inputs the eeg stage round-trips through
        # mne.export.export_raw → EDF, which drops stim/sync channels
        # (EDF can't represent them). The resulting on-disk EDF and
        # channels.tsv contain only EEG + EOG. Pick the same subset here
        # so our channels.tsv matches the EDF content exactly. BDF
        # inputs are copied straight through write_raw_bids, so all
        # source channels (EEG + EOG + MISC + TRIG) land on disk and
        # we use the raw as-is.
        if self.file_type == ".bdf":
            raw_for_tsv = self.raw_file
        else:
            raw_for_tsv = self.raw_file.copy().pick(['eeg', 'eog'])

        channels_path = bids_path.copy().update(
            suffix="channels", extension=".tsv",
        )
        # mne-bids dropped the ``convert_fmt`` kwarg from ``_channels_tsv`` in
        # newer releases; pass it only when the installed version accepts it.
        channels_kwargs = {"overwrite": overwrite}
        if "convert_fmt" in inspect.signature(_channels_tsv).parameters:
            channels_kwargs["convert_fmt"] = None
        _channels_tsv(raw_for_tsv, channels_path.fpath, **channels_kwargs)
        self._flag_nonfinite_channels(channels_path.fpath)
        _write_dig_bids(bids_path, raw_for_tsv,
                        montage=self.montage, overwrite=overwrite)

    def _flag_nonfinite_channels(self, channels_tsv):
        """Mark channels whose non-finite samples were zero-filled as bad.

        Runs after ``_channels_tsv`` has written the file, so the substitution
        made in ``_scrub_nonfinite`` is recorded where a reader will find it
        instead of silently passing as clean signal.
        """
        flagged = getattr(self, 'nonfinite_channels', None)
        if not flagged:
            return
        channels = pd.read_csv(channels_tsv, sep="\t")
        if 'status' not in channels.columns:
            channels['status'] = 'good'
        if 'status_description' not in channels.columns:
            channels['status_description'] = 'n/a'
        for name, (count, first, last) in flagged.items():
            row = channels['name'] == name
            channels.loc[row, 'status'] = 'bad'
            channels.loc[row, 'status_description'] = (
                f"{count} non-finite samples (samples {first}-{last}) "
                f"zero-filled during conversion")
        channels.to_csv(channels_tsv, sep="\t", index=False)
