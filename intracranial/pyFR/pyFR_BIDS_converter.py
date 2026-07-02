# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
import scipy
from pathlib import Path
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

_HERE = Path(__file__).parent

# Default raw EEG sample format for pyFR split-channel recordings whose
# params.txt omits the `dataformat` line. Verified int16: reading the raw
# eeg.noreref/*.NNN files as int16 yields clean EEG ranges, whereas float32
# yields denormal garbage; intact same-era subjects all declare 'int16'.
_PYFR_DEFAULT_DATAFORMAT = "int16"


def _patch_cmlreaders_params_reader():
    """Make cmlreaders tolerant of old pyFR split-EEG quirks.

    Two staging issues in old pyFR subjects break cmlreaders' raw EEG load:

    1. Incomplete ``params.txt``: many subjects have only ``samplerate <rate>``
       (a few wrote ``samplerate=<rate>``), so ``EEGMetaReader._read_params_txt``
       raised ``KeyError: 'dataformat'`` / ``'samplerate'``. We replace it with a
       line-based parser that accepts both ``key value`` and ``key=value`` and
       defaults a missing ``dataformat`` to int16. ``gain`` is intentionally
       ignored — cmlreaders never uses it; physical-µV scaling is carried by
       ``conversion_to_V`` in system_1_unit_conversions.csv.

    2. Auxiliary files in eeg.noreref: ``SplitEEGReader.read`` globs
       ``<basename>.*`` and does ``int(ext)`` on every match, so a stray
       ``.sync`` or ``.params.txt`` file raised ``ValueError: invalid literal
       for int() ... 'sync'`` / ``'txt'``. We filter the glob to files whose
       extension is a numeric channel index.

    3. Matlab event files with no ``session`` column: cmlreaders'
       ``EventReader._read_matlab_events`` does ``df[df["session"] == ...]`` and
       raised ``KeyError: 'session'`` for old single-session subjects (FZ*,
       FR091). We add the column (= the requested session) before filtering, for
       both the main and math event frames.

    Only the params.txt / split-EEG / matlab-events paths are affected
    (pyFR-type data); r1/ltp experiments read sources.json and HDF5. Idempotent.
    """
    from cmlreaders.readers.eeg import EEGMetaReader, SplitEEGReader, EEGReader
    from cmlreaders.readers.readers import EventReader
    from cmlreaders.path_finder import PathFinder
    from pathlib import Path as _Path
    import glob as _glob

    def _eegfile_to_str(v):
        """Coerce one `eegfile` cell to a clean scalar string.

        Old/System-1 pyFR events store `eegfile` per row as a numpy array
        (often the empty ``array([], dtype='<U1')`` for non-EEG events) or a
        float/NaN scalar. cmlreaders' ``to_absolute`` does
        ``row["eegfile"].startswith("/")`` and crashes on any non-str.
        """
        if isinstance(v, str):
            return v
        if isinstance(v, np.ndarray):
            size = getattr(v, "size", None)
            if size == 0:
                return ""
            if size == 1:
                return _eegfile_to_str(v.item())
            return _eegfile_to_str(v.flat[0])
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return str(v)
    if getattr(EEGMetaReader, "_pyfr_params_patched", False):
        return

    def _read_params_txt(self):
        params = {}
        with open(self.file_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"(\w+)\s*[=\s]\s*(.+)", line)
                if m:
                    params[m.group(1)] = m.group(2).strip()
        if "samplerate" not in params:
            raise KeyError("samplerate")        # genuinely unusable; keep failing loudly
        return {
            "sample_rate": float(params["samplerate"]),
            "data_format": params.get("dataformat", _PYFR_DEFAULT_DATAFORMAT).replace("'", ""),
            "n_samples": None,
            "path": self.file_path,
        }

    def _get_files(self, glob_pattern):
        files = sorted(_Path(self.filename).parent.glob(glob_pattern + ".*"))
        # Keep only per-channel data files (numeric extension); drop auxiliary
        # files like .sync and .params.txt that otherwise break int(ext).
        return [f for f in files if f.name.split(".")[-1].isdigit()]

    def _read_matlab_events(self):
        import scipy.io as sio
        df = pd.DataFrame(sio.loadmat(self.file_path, squeeze_me=True)["events"])
        # Old single-session event files may lack a `session` column; every row
        # belongs to this session, so add it before the filter below.
        if "session" not in df.columns:
            df["session"] = self.session
        if self.session is not None:
            df = df[df["session"] == self.session]
        # Some System-1 events store `eegfile` as a numpy array, and some old
        # sessions store it as a float/NaN scalar; both break cmlreaders'
        # to_absolute. Coerce here (and again at _eegfile_absolute, which also
        # catches events loaded via other reader paths).
        if "eegfile" in df.columns:
            df["eegfile"] = df["eegfile"].map(_eegfile_to_str)
        # Re-implant event files store `subject` as `<subject>_<montage>` (e.g.
        # TJ035_1); cmlreaders' load_eeg rejects events whose subject != the
        # reader's. Normalize to the reader's base subject.
        if "subject" in df.columns:
            df["subject"] = self.subject
        # `df.loc[:, col] = scalar` raises on an empty (no-index) frame; plain
        # assignment is safe whether or not the frame has rows.
        if "experiment" not in df.columns:
            df["experiment"] = self.experiment
        if self.experiment == 'pyFR':
            math_toggle = False
            if self.montage != 0:
                fpath = f'/data/events/pyFR/{self.subject}_{self.montage}_math.mat'
            else:
                fpath = f'/data/events/pyFR/{self.subject}_math.mat'
            if os.path.exists(fpath):
                math_toggle = True
                math_df = pd.DataFrame(sio.loadmat(fpath, squeeze_me=True)['events'])
            if math_toggle:
                if "session" not in math_df.columns:
                    math_df["session"] = self.session
                math_df = math_df[math_df['session'] == self.session]
                math_df = math_df[(math_df['type'] != 'B') & (math_df['type'] != 'E')]
                math_df['list'] = math_df['list'] - 1    # math events given list + 1
                df = pd.concat([math_df, df], ignore_index=True)
                df = df.sort_values(by='mstime', ascending=True, ignore_index=True)
        return df

    _orig_find = PathFinder.find

    def _find(self, data_type, *args, **kwargs):
        try:
            return _orig_find(self, data_type, *args, **kwargs)
        except FileNotFoundError:
            # cmlreaders' EEGReader.load builds a PathFinder WITHOUT montage, so
            # for re-implant (montage>0) pyFR subjects the task_events lookup
            # resolves `{subject}_events.mat` and misses the real
            # `{subject}_{montage}_events.mat`. Fall back to the montage-suffixed
            # events file on a miss.
            if data_type not in ("task_events", "all_events"):
                raise
            if getattr(self, "experiment", None) != "pyFR":
                raise
            root = getattr(self, "rootdir", "") or "/"
            cands = sorted(_glob.glob(os.path.join(
                root, "data", "events", "pyFR", f"{self.subject}_*_events.mat")))
            cands = [c for c in cands
                     if re.fullmatch(rf"{self.subject}_\d+_events\.mat", os.path.basename(c))]
            if len(cands) == 1:
                return cands[0]
            if len(cands) > 1:
                # Disambiguate multi-montage subjects via the data index. Use
                # boolean masking with explicit int coercion (session dtype
                # varies across the index / PathFinder), not .query with @-vars.
                idx = cml.get_data_index()
                try:
                    sess = int(self.session)
                    m = idx[(idx["subject"] == self.subject)
                            & (idx["experiment"] == "pyFR")
                            & (idx["session"].astype(int) == sess)]
                except (TypeError, ValueError):
                    m = idx.iloc[0:0]
                if len(m):
                    p = os.path.join(root, "data", "events", "pyFR",
                                     f"{self.subject}_{int(m.iloc[0]['montage'])}_events.mat")
                    if os.path.exists(p):
                        return p
            raise

    # EEGReader.load builds the events frame via whichever event reader the
    # source declares (not always the patched _read_matlab_events), so empty
    # ndarray / float eegfile cells can still reach to_absolute. Normalize the
    # eegfile column at the entry to _eegfile_absolute as a catch-all.
    _orig_eegfile_absolute = EEGReader._eegfile_absolute

    def _eegfile_absolute(self, events):
        if "eegfile" in getattr(events, "columns", []):
            events = events.copy()
            events["eegfile"] = events["eegfile"].map(_eegfile_to_str)
        return _orig_eegfile_absolute(self, events)

    EEGMetaReader._read_params_txt = _read_params_txt
    SplitEEGReader._get_files = _get_files
    EventReader._read_matlab_events = _read_matlab_events
    EEGReader._eegfile_absolute = _eegfile_absolute
    PathFinder.find = _find
    EEGMetaReader._pyfr_params_patched = True


_patch_cmlreaders_params_reader()

class pyFR_BIDS_converter(intracranial_BIDS_converter):
    wordpool_EN = np.loadtxt(_HERE / 'wordpools' / 'wordpool_EN.txt', dtype=str)
    CH_TYPES = {'TJ027': 'ECOG', 'TJ029': 'SEEG', 'TJ030': 'SEEG', 'TJ032': 'ECOG', 'TJ061': 'ECOG', 'TJ083':'ECOG',
                'UP004': 'SEEG', 'UP008':'ECOG', 'UP011':'ECOG', 'UP037': 'ECOG'}

    # initialize
    # just hand empty dictionary for brain_regions
    def __init__(self, subject, experiment, session, montage, math_events, system_version, unit_scale, monopolar, bipolar, mni, tal, overrides=None, root='/scratch/hherrema/BIDS/pyFR/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.montage = montage
        self.math_events = math_events
        self.system_version = system_version
        self.unit_scale = unit_scale
        self.monopolar = monopolar
        self.bipolar = bipolar
        self.mni = mni
        self.tal = tal
        self.overrides = overrides or {}
        self.root = root

    # ---------- BIDS Utility ---------
    # return a base BIDS_path object to update
    def _BIDS_path(self):
        # update session if re-implant
        new_session = self.reassign_session()
        if new_session:
            bids_path = mne_bids.BIDSPath(subject=self.subject, task=self.experiment, session=str(new_session),
                                            root=self.root)
        else:
            bids_path = mne_bids.BIDSPath(subject=self.subject, task=self.experiment, session=str(self.session),
                                            root=self.root)
        return bids_path
    # instantiate CMLRead object, save as attribute
    def cml_reader(self):
        df = cml.get_data_index()
        q = "subject==@self.subject & experiment==@self.experiment & session==@self.session"
        matches = df.query(q + " & montage==@self.montage")
        if len(matches) == 0:
            # The caller defaults montage to 0, but some subjects/sessions are
            # re-implants indexed under a different montage. Fall back to the
            # index's montage for this (subject, experiment, session).
            matches = df.query(q)
        if len(matches) == 0:
            raise ValueError(
                f"no data-index entry for {self.subject}/{self.experiment}/ses-{self.session}")
        sel = matches.iloc[0]
        # Keep self.montage consistent with what we actually loaded so
        # downstream paths (re-implant remap, MNI coords) use the right montage.
        self.montage = int(sel.montage)
        reader = cml.CMLReader(subject=sel.subject, experiment=sel.experiment, session=sel.session,
                               localization=sel.localization, montage=sel.montage)
        return reader

    def reassign_session(self):
        re_implants = pd.read_csv(_HERE / 're_implants.csv')
        ri = re_implants[(re_implants.subject == self.subject) & (re_implants.montage == self.montage) &
                         (re_implants.session == self.session)]
        if len(ri) == 1:
            return ri.iloc[0].new_session
        else:
            return None

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']
        if np.all([x in self.wordpool_EN for x in word_evs.item]):
            wordpool_file = 'wordpools/wordpool_EN.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file

    def events_to_BIDS(self):
        events = self._load_events()     # cmlreaders now loads in math events automatically

        # An empty events frame (e.g. TJ074: corrupt/empty event .mat) has no
        # mstime[0] to anchor onsets; fail cleanly rather than IndexError on
        # `.iloc[0]` so the stage is attributed correctly.
        if events is None or len(events) == 0:
            raise ValueError(
                f"Events DataFrame is empty for {self.subject} ses-{self.session}; "
                "event .mat likely empty or not uploaded.")

        # transformations
        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})                  # rename columns
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0        # event duration [s]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                  # replace events with negative duration with 0.0 s
        events = self.apply_event_durations(events)                                                  # apply well-defined durations [s]
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                           # onset from first event [s]
        events['response_time'] = 'n/a'                                                              # response time [s]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') |
                   (events.trial_type=='PROB'), 'response_time'] = events['rectime'] / 1000.0
        events['stim_file'] = np.where(events.trial_type=='WORD', self.wordpool_file, 'n/a')              # add wordpool to word events
        events['item_name'] = events['item'].replace('X', 'n/a')
        events = self.assign_serial_positions(events)                                                # assign serail positions to recalls
        # Some columns (e.g. the math `test` problem = array([X, Y, Z])) hold
        # array/list cell values. pandas' elementwise fillna/replace below does
        # `cell == ''`, which raises "truth value of an array ... is ambiguous"
        # on those cells. Serialize any array/list cells to scalar strings first
        # (also makes them TSV-writable).
        for col in events.columns:
            if events[col].map(lambda v: isinstance(v, (list, np.ndarray))).any():
                events[col] = events[col].map(
                    lambda v: str(list(v)) if isinstance(v, (list, np.ndarray)) else v)
        events = events.fillna('n/a')                                                                # chnage NaN to 'n/a'
        events = events.replace('', 'n/a')                                                           # no empty cells

        # update session if re-implant
        new_session = self.reassign_session()
        if new_session:
            events['session'] = new_session
            events['subject'] = self.subject   # remove _montage from subject

        if self.math_events:
            events = events[self._append_uncorrected_cols(events, ['onset', 'duration', 'sample', 'trial_type', 'response_time',
                            'stim_file', 'item_name', 'serialpos', 'list', 'test', 'answer',
                            'experiment', 'session', 'subject'])]                                        # re-order columns + drop unneeded fields
        else:
            events = events[self._append_uncorrected_cols(events, ['onset', 'duration', 'sample', 'trial_type', 'response_time',
                             'stim_file', 'item_name', 'serialpos', 'list',
                             'experiment', 'session', 'subject'])]
        return events

    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # fixation events = 1600 ms
            if row.trial_type == 'ORIENT':
                durations.append(1.6)

            # word events = 1600 ms
            elif row.trial_type == 'WORD':
                durations.append(1.6)

            # keep current duration
            else:
                durations.append(row.duration)

        events['duration'] = durations        # preserves column order
        return events

    # assign serial positions to recall events (all given serial position = -999)
    def assign_serial_positions(self, events):
        serialpos = []
        for l, l_evs in events.groupby('list', sort=False):         # preserve order
            w_evs = l_evs.query("trial_type == 'WORD'")
            r_evs = l_evs.query("trial_type == 'REC_WORD'")

            # recall events all given default serial position == -999 or 0

            # Coerce to strings so the membership/argwhere comparisons stay
            # scalar — a stray array/empty-array cell otherwise triggers a
            # numpy broadcast error ("operands could not be broadcast ...").
            words = np.array([str(x) for x in w_evs.item_name])
            recs = np.array([str(x) for x in r_evs.item_name])
            sp = []
            for r in recs:
                match = np.argwhere(words == r)
                sp.append(int(match[0][0]) + 1 if len(match) else -999)

            serialpos.extend(sp)

        # Guard against a length mismatch (e.g. no REC_WORD rows but serialpos
        # populated, or vice versa), which pandas reports as a broadcast error.
        n_rec = int((events['trial_type'] == 'REC_WORD').sum())
        if len(serialpos) == n_rec:
            events.loc[events['trial_type'] == 'REC_WORD', 'serialpos'] = serialpos
        else:
            print(f"WARNING: serialpos length {len(serialpos)} != REC_WORD count {n_rec} "
                  f"for {self.subject}/{self.experiment}/ses-{self.session}; leaving serialpos unset")
        return events

    def make_events_descriptor(self):
        descriptions = {
            'B': 'Beginning of session.',
            'SESS_START': 'Beginning of session.',
            'E': 'End of session.',
            'START': "Beginning of math distractor phase.",
            'PROB': "Math problem presentation onset.",
            'STOP': "End of math distractor phase.",
            'SESS_END': 'End of session.',
            'ORIENT': 'Fixation preceding presentation of word list.',
            'REC_START': 'Start of recall phase.',
            'TRIAL': 'Denotes new trial (list).',
            'WORD': 'Word presentation (onset).',
            'REC_WORD': 'Recalled word, onset of speech (during free recall).',
            'REC_WORD_VV': 'Vocalization, onset of speech (during free recall).',
            'RECOG_START': 'Start of recognition phase.',
            'RECOG_END': 'End of recognition phase.',
            'RECOG_PRES': 'Word presentation (onset) during recognition.',
            'RECOG_RESP': 'Recognition response (old/new judgment).',
            'RECOG_FEEDBACK': 'Feedback during recognition.',
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category",
                           "Description": "Indicator of type of task action that occurs at the marked time",
                           "Levels": {k:descriptions.get(k, k) for k in self.events["trial_type"].unique()}},
            "response_time": {"Description": "Time (in seconds) between onset of recall phase and recall (for recalls and vocalizations), or between onset of problem on screen and response (for math problems)."},
            "stim_file": {"LongName": "Stimulus File",
                          "Description": "Location of wordpool file containing words presented in WORD events."},
            "subject": {"LongName": "Subject ID",
                        "Description": "The string identifier of the subject, e.g. R1001P."},
            'experiment': {'Description': 'The experimental paradigm completed.'},
            "session": {"Description": "The session number."},
            "list": {"LongName": "List Number",
                     "Description": "Word list (1-24) during which the event occurred. Trial <= 0 indicates practice list."},
            "item_name": {"Description": "The word being presented or recalled in a WORD or REC_WORD event."},
            'serialpos': {'LongName': 'Serial Position',
                          'Description': 'The order position (at encoding) of a word presented in an WORD event or a recall in a REC_WORD event.'},
            'test': {"LongName": "Math problem",
                     "Description": "Math problem with form X + Y + Z = ?  Stored in list [X, Y, Z]."},
            'answer': {"LongName": "Math problem response",
                       "Description": "Participant answer to problem with form X + Y + Z = ?  Note this is not necessarily the correct answer."}
        }
        HED.update(self.UNCORRECTED_HED)
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor

    # ---------- Electrodes ----------
    def load_contacts(self):
        contacts = self.reader.load('contacts')
        # Some (older) subjects' contacts have no `type` column at all, which
        # made `contacts.type` raise AttributeError. Guarantee the column and
        # normalize missing/non-string entries to 'n/a'.
        if 'type' not in contacts.columns:
            contacts['type'] = 'n/a'
        else:
            contacts['type'] = [x if isinstance(x, str) else 'n/a' for x in contacts['type']]
        return contacts

    def _load_mni_coords(self):
        """Load the per-subject MNI coords text file. Returns Nx4 array
        ([contact, x, y, z]) or None if the file isn't available."""
        try:
            if self.montage != 0:
                path = f'/data/eeg/{self.subject}_{self.montage}/tal/RAW_coords.txt.mni'
            else:
                path = f'/data/eeg/{self.subject}/tal/RAW_coords.txt.mni'
            return np.loadtxt(path)
        except (OSError, ValueError):
            return None

    def _available_cml_spaces(self):
        available = []
        cols = set(self.contacts.columns)
        if {'x', 'y', 'z'}.issubset(cols):
            available.append('tal')
        if self._load_mni_coords() is not None:
            available.append('mni')
        return available

    def contacts_to_electrodes(self, cml_space):
        # Import here to avoid a cycle with the package __init__.
        from ..intracranial_BIDS_converter import CML_TO_BIDS_SPACE  # noqa: F401

        electrodes = pd.DataFrame({'name': np.array(self.contacts.label)})

        if cml_space == 'tal':
            electrodes['x'] = self.contacts['x'].astype(float)
            electrodes['y'] = self.contacts['y'].astype(float)
            electrodes['z'] = self.contacts['z'].astype(float)
        elif cml_space == 'mni':
            # Map MNI coords to each contact by contact number, filling NaN for
            # contacts absent from the coords file. (Aligning by position fails
            # when the coords file has fewer contacts than the contacts table —
            # e.g. CH066: 150 coords vs 156 contacts.)
            mni_coords = self._load_mni_coords()
            coord_map = {int(row[0]): row[1:4] for row in mni_coords}
            xyz = np.array([
                coord_map.get(int(c), (np.nan, np.nan, np.nan))
                for c in np.array(self.contacts.contact)
            ], dtype=float)
            electrodes['x'] = xyz[:, 0]
            electrodes['y'] = xyz[:, 1]
            electrodes['z'] = xyz[:, 2]
        else:
            electrodes['x'] = np.nan
            electrodes['y'] = np.nan
            electrodes['z'] = np.nan

        electrodes['size'] = -999
        # Older subjects' contacts may lack grpName / Loc* columns; guard each
        # so a missing column degrades to 'n/a' instead of raising AttributeError.
        if 'grpName' in self.contacts.columns:
            electrodes['group'] = np.array(self.contacts.grpName)
        else:
            electrodes['group'] = [re.sub(r'\d+$', '', str(lbl)) for lbl in self.contacts.label]
        loc1 = self.contacts['Loc1'] if 'Loc1' in self.contacts.columns else ['n/a'] * len(self.contacts)
        electrodes['hemisphere'] = ['L' if isinstance(x, str) and 'Left' in x
                                    else 'R' if isinstance(x, str) and 'Right' in x
                                    else 'n/a' for x in loc1]
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) if isinstance(x, str) else None
                              for x in self.contacts['type']]
        electrodes['lobe'] = np.array(self.contacts['Loc2']) if 'Loc2' in self.contacts.columns else 'n/a'
        electrodes['region1'] = np.array(self.contacts['Loc3']) if 'Loc3' in self.contacts.columns else 'n/a'
        electrodes['region2'] = np.array(self.contacts['Loc5']) if 'Loc5' in self.contacts.columns else 'n/a'
        electrodes['gray_white'] = np.array(self.contacts['Loc4']) if 'Loc4' in self.contacts.columns else 'n/a'
        electrodes = electrodes.fillna('n/a')
        electrodes = electrodes.replace('', 'n/a')
        electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type',
                                 'lobe', 'region1', 'region2', 'gray_white']]
        return electrodes

    def make_electrodes_sidecar(self, cml_space):
        from ..intracranial_BIDS_converter import CML_TO_BIDS_SPACE
        bids_space = CML_TO_BIDS_SPACE[cml_space]
        sidecar = {
            'name': 'Label of electrode',
            'x': f'x-axis position in {bids_space} coordinates',
            'y': f'y-axis position in {bids_space} coordinates',
            'z': f'z-axis position in {bids_space} coordinates',
            'size': 'Surface area of electrode.',
            'group': 'Group of channels electrode belongs to (same shank).',
            'hemisphere': 'Hemisphere of electrode location.',
            'type': 'Type of electrode.',
            'lobe': 'Brain lobe of electrode location.',
            'region1': 'Brain region of electrode location.',
            'region2': 'Brain region of electrode location.',
            'gray_white': 'Denotes gray or white matter.',
        }
        return sidecar

    # ---------- Channels ----------
    def load_pairs(self):
        pairs = self.reader.load('pairs')
        # Guarantee a `type` column (some subjects' pairs lack one) so the
        # inherited base channel/sidecar logic never hits AttributeError.
        if 'type' not in pairs.columns:
            pairs['type'] = 'n/a'
        else:
            pairs['type'] = [x if isinstance(x, str) else 'n/a' for x in pairs['type']]
        return pairs

    # NOTE: contacts_to_channels / pairs_to_channels / eeg_metadata /
    # eeg_bi_to_BIDS / eeg_mono_to_BIDS / run() are intentionally NOT overridden
    # here. They are inherited from intracranial_BIDS_converter, whose modern
    # implementations provide per-stage resilience, phantom-contact filtering,
    # channel reconciliation, and the EDF/BDF digital-writer EEG path. Only the
    # genuinely pyFR-specific pieces (events, wordpool, re-implant sessions,
    # coordinate spaces, and the eeg_sidecar channel-count fields) are kept.

    # ---------- EEG ----------
    def eeg_sidecar(self, ref):                 # overwrite for different 'type' field
        sidecar = {'TaskName': self.experiment}
        sidecar['TaskDescription'] = 'delayed free recall of word lists'
        sidecar['SamplingFrequency'] = float(self.sfreq)
        sidecar['PowerLineFrequency'] = 60.0    # check for German subjects
        sidecar['SoftwareFilters'] = 'n/a'
        sidecar['HardwareFilters'] = 'n/a'
        sidecar['RecordingDuration'] = float(self.recording_duration)
        sidecar['RecordingType'] = 'continuous'
        sidecar['ElectricalStimulation'] = False
        if ref == 'bipolar':
            sidecar['iEEGReference'] = 'bipolar'
            sidecar['ECOGChannelCount'] = len(self.pairs[(self.pairs.type=='S') | (self.pairs.type=='G')].index)
            sidecar['SEEGChannelCount'] = len(self.pairs[(self.pairs.type=='D') | (self.pairs.type=='uD')].index)
            sidecar['MiscChannelCount'] = len(self.pairs[self.pairs.type=='n/a'].index)
            sidecar['EEGChannelCount'] = 0
        elif ref == 'monopolar':
            sidecar['iEEGReference'] = 'monopolar'
            sidecar['ECOGChannelCount'] = len(self.contacts[(self.contacts.type=='S') | (self.contacts.type=='G')].index)
            sidecar['SEEGChannelCount'] = len(self.contacts[(self.contacts.type=='D') | (self.contacts.type=='uD')].index)
            sidecar['MiscChannelCount'] = len(self.contacts[self.contacts.type=='n/a'].index)
            sidecar['EEGChannelCount'] = 0
        return sidecar
