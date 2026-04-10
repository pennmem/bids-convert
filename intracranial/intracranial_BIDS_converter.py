# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
from glob import glob
import mne_bids

from .edf_digital_writer import resolve_edf_units, write_digital


class intracranial_BIDS_converter:
    ELEC_TYPES_DESCRIPTION = {'S': 'strip', 'G': 'grid', 'D': 'depth', 'uD': 'micro'}
    ELEC_TYPES_BIDS = {'S': 'ECOG', 'G': 'ECOG', 'D': 'SEEG', 'uD': 'SEEG'}
    BRAIN_REGIONS = ['wb.region', 'ind.region', 'das.region', 'stein.region']

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale=1e6, monopolar=True, bipolar=True, mni=True, tal=False, area=False, brain_regions=None, root='/scratch/hherrema/BIDS/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.system_version = system_version
        self.unit_scale = unit_scale
        self.monopolar = monopolar # bool
        self.bipolar = bipolar # bool
        self.mni = mni # bool
        self.tal = tal # bool
        self.area = area # bool
        if self.area:
            self.area_map = self.generate_area_map()
        self.brain_regions = brain_regions   # dictionary mapping target regions to number of non-NaN contacts
        self.root = root

    @property
    def task_label(self):
        """BIDS-compliant task label: alphanumeric only (e.g. 'PS21' for 'PS2.1')."""
        return re.sub(r'[^a-zA-Z0-9]', '', self.experiment)

    # ---------- BIDS Utility ----------
    # return a base BIDS_path object to update
    def _BIDS_path(self):
        bids_path = mne_bids.BIDSPath(subject=self.subject, task=self.task_label, session=str(self.session),
                                        root=self.root)
        return bids_path

    # write pandas dataframe to tsv file
    def _to_tsv(self, dframe, fpath):
        dframe.to_csv(fpath, sep='\t', index=False)
    
        
    # instantiate CMLReader object, save as attribute\
    def cml_reader(self):
        df = cml.get_data_index('r1', '/')
        sel = df.query("subject==@self.subject & experiment==@self.experiment & session==@self.session").iloc[0]
        reader = cml.CMLReader(subject=sel.subject, experiment=sel.experiment, session=sel.session, 
                               localization=sel.localization, montage=sel.montage)
        return reader
    
    # ---------- Events ----------
    def set_wordpool(self):
        raise NotImplementedError       # override in subclass
    
    def events_to_BIDS(self):
        raise NotImplementedError       # override in subclass
    
    def make_events_descriptor(self):
        raise NotImplementedError       # override in subclass
    
    def write_BIDS_beh(self):
        bids_path = self._BIDS_path().update(suffix='beh', extension='.tsv', datatype='beh')
        os.makedirs(bids_path.directory, exist_ok=True)
        
        # write events to tsv
        self._to_tsv(self.events, bids_path.fpath)

        # write sidecar json
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)

    # ---------- Electrodes ----------
    def load_contacts(self):
        try:
            return self.reader.load('contacts')
        except KeyError as exc:
            if "contact" in str(exc) and "label" in str(exc):
                raise RuntimeError(
                    f"contacts file for {self.subject}/{self.experiment}/ses-{self.session} "
                    f"is missing required columns (contact, label); skipping session"
                ) from exc
            raise

    def _filter_scheme_to_recording(self, scheme, scheme_name="contacts"):
        """Filter a contacts/pairs scheme to only entries the recording has.

        Discovers phantom contacts by catching ``KeyError`` from
        ``load_eeg(scheme=...)``. Each ``KeyError(<int>)`` identifies a
        missing contact number; we drop every row that references it (for
        contacts that's the ``contact`` column; for pairs it's
        ``contact_1`` or ``contact_2``) and retry until the load succeeds.

        Returns ``(filtered_scheme, dropped_scheme_df)``.
        """
        events = self.reader.load("events").iloc[[0]]
        dropped_ids = set()
        filtered = scheme.copy()

        # Which columns hold the contact numbers we need to filter on?
        if "contact" in filtered.columns:
            id_cols = ["contact"]
        else:
            id_cols = [c for c in ("contact_1", "contact_2") if c in filtered.columns]

        while True:
            try:
                self.reader.load_eeg(
                    events=events, rel_start=0, rel_stop=50, scheme=filtered,
                )
                break
            except KeyError as exc:
                try:
                    missing = int(exc.args[0])
                except (TypeError, ValueError):
                    raise
                n_before = len(filtered)
                # Drop any row where ANY id column references the missing contact
                mask = pd.Series(False, index=filtered.index)
                for col in id_cols:
                    mask |= (filtered[col] == missing)
                filtered = filtered[~mask]
                dropped_ids.add(missing)
                if len(filtered) == n_before or filtered.empty:
                    raise

        # Build the dropped df: rows from original scheme that were removed
        mask = pd.Series(False, index=scheme.index)
        for col in id_cols:
            mask |= scheme[col].isin(dropped_ids)
        dropped = scheme[mask]

        if len(dropped):
            print(
                f"  Dropped {len(dropped)} phantom {scheme_name} "
                f"(missing contact IDs: {sorted(dropped_ids)})"
            )
        return filtered, dropped
    
    def generate_area_map(self):
        area_path = f'/data10/RAM/subjects/{self.subject}/docs/area.txt'
        try:
            area = np.loadtxt(area_path, dtype=str)
            area_map = dict(zip(area[:, 0], area[:, 1].astype(float)))
        except BaseException:
            if self.system_version == 4.0:
                area_map = self.generate_area_map_system_4()
            else:
                area_map = {}

        return area_map
    
    def generate_area_map_system_4(self):
        pf = cml.PathFinder(self.subject, self.experiment, self.session)
        reader = cml.CMLReader(self.subject, self.experiment, self.session)

        try:
            path = f'/data10/RAM/subjects/{self.subject}/behavioral/{self.experiment}/session_{self.session}/elemem/{pf.find("elemem_session_folder")}/'
            config_files = glob(path + f'{self.subject}*.csv')       # maybe should specificy "mono" in pattern match
            if len(config_files) == 1:                               # config files don't have consisten names, but want the monopolar (not bipolar) config
                areas = pd.read_csv(config_files[0], names=['label', 'index', 'area'], header=None, usecols=[0, 1, 2], on_bad_lines='skip')
                contacts = reader.load('contacts')
                area = pd.merge(areas, contacts)[['label', 'area']]

                groups = []
                for _, row in area.iterrows():
                    split_label = list(row.label)
                    alpha_idx = [i for i, c in enumerate(split_label) if c.isalpha()]   # indices of alphabetical characters in label
                    groups.append(row.label[:alpha_idx[-1]+1])
                
                area['group'] = groups
                area_groups = area.groupby(['group'])['area'].agg(pd.Series.mode).reset_index()
                area_map = dict(zip(area_groups.group, area_groups.area.astype(float)))
        except BaseException:
            area_map = {}

        return area_map
    
    # convert CML contacts to BIDS electrodes
    def contacts_to_electrodes(self, atlas, toggle):
        # Use the full contacts set (including phantom contacts) for
        # electrodes — these are physical positions, not data channels.
        contacts = getattr(self, 'contacts_all', self.contacts)
        electrodes = pd.DataFrame({'name': np.array(contacts.label)})      # name = label of contact
        electrodes['x'] = contacts[f'{atlas}.x']
        electrodes['y'] = contacts[f'{atlas}.y']
        electrodes['z'] = contacts[f'{atlas}.z']
        
        # electrode groups (shanks)
        groups = []
        for _, row in contacts.iterrows():
            split_label = list(row.label)
            alpha_idx = [i for i, c in enumerate(split_label) if c.isalpha()]
            groups.append(row.label[:alpha_idx[-1]+1])
        electrodes['group'] = groups

        if self.area:
            electrodes['size'] = [self.area_map.get(x) if x in self.area_map.keys() else -999 for x in electrodes.group]
        else:
            electrodes['size'] = -999

        electrodes['hemisphere'] = ['L' if x < 0 else 'R' if x > 0 else 'n/a' for x in electrodes.x]
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in contacts.type]

        # anatomical regions
        br_cols = []
        for br in self.BRAIN_REGIONS:
            if self.brain_regions[br] > 0:
                if br not in contacts.columns:
                    print(f"WARNING: brain region column '{br}' not found in contacts for {self.subject} — omitting from electrodes TSV.")
                    continue
                electrodes[br] = contacts[br].values
                br_cols.append(br)

        # both MNI and Talairach --> default to MNI first, manually define Tal
        if toggle:
            electrodes['tal.x'] = contacts['tal.x'].values
            electrodes['tal.y'] = contacts['tal.y'].values
            electrodes['tal.z'] = contacts['tal.z'].values
        
        electrodes = electrodes.fillna('n/a')                                   # remove NaN
        electrodes = electrodes.replace('', 'n/a')                              # resolve empty cell issue
        
        if toggle:
            electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type',
                                     'tal.x', 'tal.y', 'tal.z'] + br_cols]
        else:
            electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type'] + br_cols]          # re-order columns
        return electrodes
    
    # create sidecar json for electrodes.tsv
    def make_electrodes_sidecar(self, atlas, toggle):
        sidecar = {'name': 'Label of electrode.'}

        # coordinate system
        if atlas == 'mni':
            sidecar['x'] = 'x-axis position in MNI coordinates'
            sidecar['y'] = 'y-axis position in MNI coordinates'
            sidecar['z'] = 'z-axis position in MNI coordinates'
        elif atlas == 'tal':
            sidecar['x'] = 'x-axis position in Talairach coordinates'
            sidecar['y'] = 'y-axis position in Talairach coordinates'
            sidecar['z'] = 'z-axis position in Talairach coordinates'

        sidecar['size'] = 'Surface area of electrode.'
        sidecar['group'] = 'Group of channels electrode belongs to (same shank).'
        sidecar['hemisphere'] = 'Hemisphere of electrode location.'
        sidecar['type'] = 'Type of electrode.'

        # brain regions
        if self.brain_regions['wb.region'] > 0:
            sidecar['wb.region'] = 'Brain region of electrode location from subcortical neuroradiology pipeline.'
        if self.brain_regions['ind.region'] > 0:
            sidecar['ind.region'] = 'Brain region of electrode location from surface neuroradiology pipeline.'
        if self.brain_regions['das.region'] > 0:
            sidecar['das.region'] = 'Brain region of electrode location from hand annotations by neuroradiologist.  Usually in MTL.'
        if self.brain_regions['stein.region'] > 0:
            sidecar['stein.region'] = 'Brain region of electrode location from hand annotations by neurologist.  Usually in MTL.'

        # both MNI and Talairach
        if toggle:
            sidecar['tal.x'] = 'x-axis position in Talairach coordinates'
            sidecar['tal.y'] = 'x-axis position in Talairach coordinates'
            sidecar['tal.z'] = 'z-axis position in Talairach coordinates'

        return sidecar

    def write_BIDS_electrodes(self, atlas, toggle):
        bids_path = self._BIDS_path().update(suffix='electrodes', extension='.tsv', datatype='ieeg')
        # COULD DO AWAY WITH SEPARATE tal AND mni ATTRIBUTES
        if toggle:
            bids_path.update(space='MNI152NLin6ASym')
            os.makedirs(bids_path.directory, exist_ok=True)
            self._to_tsv(self.electrodes, bids_path.fpath)
        elif atlas == 'tal':
            bids_path.update(space='Talairach')
            os.makedirs(bids_path.directory, exist_ok=True)
            self._to_tsv(self.electrodes_tal, bids_path.fpath)
        elif atlas == 'mni':
            bids_path.update(space='MNI152NLin6ASym')
            os.makedirs(bids_path.directory, exist_ok=True)
            self._to_tsv(self.electrodes_mni, bids_path.fpath)

        # also write sidecar json
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.electrodes_sidecar)

    def _coordinate_system(self, atlas):
        """ not allowed to add fields
        if toggle:
            return {'iEEGCoordinateSystem': 'MNI152NLin6ASym', 'iEEGCoordinateUnits': 'mm',
                    'iEEGCoordinateSystem_': 'Talairach', 'iEEGCoordinateUnits_': 'mm'}
        """
        if atlas == 'tal':
            return {'iEEGCoordinateSystem': 'Talairach', 'iEEGCoordinateUnits': 'mm'}
        elif atlas == 'mni':
            return {'iEEGCoordinateSystem': 'MNI152NLin6ASym', 'iEEGCoordinateUnits': 'mm'}
        
    def write_BIDS_coords(self, atlas):
        bids_path = self._BIDS_path().update(suffix='coordsystem', extension='.json', datatype='ieeg')
        coord_sys = self._coordinate_system(atlas)
        if atlas == 'tal':
            bids_path.update(space='Talairach')
        elif atlas == 'mni':
            bids_path.update(space='MNI152NLin6ASym')
        
        with open(bids_path.fpath, 'w') as f:
            json.dump(fp=f, obj=coord_sys)

    # ---------- Channels ----------
    def load_pairs(self):
        return self.reader.load('pairs')
    
    # convert CML pairs to BIDS channels (bipolar)
    def pairs_to_channels(self):
        labels = np.array(self.pairs.label)
        truncated = [self._truncate_bipolar(n) if len(n) > 16 else n for n in labels]
        channels = pd.DataFrame({'name': truncated})
        type_col = 'type_1' if 'type_1' in self.pairs.columns else 'type'
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.pairs[type_col]]
        channels['units'] = 'uV'
        channels['low_cutoff'] = 'n/a'
        channels['high_cutoff'] = 'n/a'
        channels['reference'] = 'bipolar'
        channels['group'] = [re.sub('\d+', '', x).split('-')[0] for x in self.pairs.label]
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.pairs[type_col]]
        channels['notch'] = 'n/a'
        channels['status'] = 'good'
        channels['status_description'] = 'n/a'

        # Append dropped pairs (phantom contacts) with status = bad
        if hasattr(self, 'pairs_dropped') and len(self.pairs_dropped):
            p = self.pairs_dropped
            p_type_col = 'type_1' if 'type_1' in p.columns else 'type'
            dropped_rows = pd.DataFrame({
                'name': [self._truncate_bipolar(n) if len(n) > 16 else n for n in np.array(p.label)],
                'type': [self.ELEC_TYPES_BIDS.get(x) for x in p[p_type_col]],
                'units': 'uV',
                'low_cutoff': 'n/a',
                'high_cutoff': 'n/a',
                'reference': 'bipolar',
                'group': [re.sub('\d+', '', x).split('-')[0] for x in p.label],
                'sampling_frequency': self.sfreq,
                'description': [self.ELEC_TYPES_DESCRIPTION.get(x) for x in p[p_type_col]],
                'notch': 'n/a',
                'status': 'bad',
                'status_description': 'not_in_recording',
            })
            channels = pd.concat([channels, dropped_rows], ignore_index=True)

        channels = channels.fillna('n/a')
        channels = channels.replace('', 'n/a')
        return channels
    
    # convert CML contacts to BIDS channels (monopolar)
    def contacts_to_channels(self):
        # Recorded contacts → status = good
        channels = pd.DataFrame({'name': np.array(self.contacts.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.contacts.type]
        channels['units'] = 'uV'
        channels['low_cutoff'] = 'n/a'
        channels['high_cutoff'] = 'n/a'
        channels['group'] = [re.sub('\d+', '', x) for x in self.contacts.label]
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]
        channels['notch'] = 'n/a'
        channels['status'] = 'good'
        channels['status_description'] = 'n/a'

        # Append phantom contacts (in wiring doc but not in recording) as
        # status = bad so every contact is documented in channels.tsv.
        if hasattr(self, 'contacts_dropped') and len(self.contacts_dropped):
            dropped_rows = pd.DataFrame({
                'name': np.array(self.contacts_dropped.label),
                'type': [self.ELEC_TYPES_BIDS.get(x) for x in self.contacts_dropped.type],
                'units': 'uV',
                'low_cutoff': 'n/a',
                'high_cutoff': 'n/a',
                'group': [re.sub('\d+', '', x) for x in self.contacts_dropped.label],
                'sampling_frequency': self.sfreq,
                'description': [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts_dropped.type],
                'notch': 'n/a',
                'status': 'bad',
                'status_description': 'not_in_recording',
            })
            channels = pd.concat([channels, dropped_rows], ignore_index=True)

        channels = channels.fillna('n/a')
        channels = channels.replace('', 'n/a')
        return channels
    
    def write_BIDS_channelmap(self, ref):
        """Write a mapping of original to truncated channel names if any were shortened.

        Older versions of mne-bids accepted ``suffix='channelmap'``;
        newer ones (≥0.13) restrict suffixes to a fixed list and raise
        ``ValueError``. Probe the installed version's allowed list and
        fall back to ``suffix='channels'`` (which is always allowed)
        when ``channelmap`` isn't permitted. The output filename ends
        in ``..._channelmap.tsv`` either way so downstream tooling
        keeps working, and we add a ``.bidsignore`` entry so the BIDS
        validator skips it.
        """
        original_labels = np.array(self.pairs.label)
        truncated_labels = [self._truncate_bipolar(n) if len(n) > 16 else n for n in original_labels]

        renamed = [(orig, trunc) for orig, trunc in zip(original_labels, truncated_labels) if orig != trunc]
        if not renamed:
            return

        mapping_df = pd.DataFrame(renamed, columns=['original_name', 'truncated_name'])

        # Pick a suffix this mne-bids version will accept.
        try:
            from mne_bids.path import ALLOWED_FILENAME_SUFFIX
            suffix = 'channelmap' if 'channelmap' in ALLOWED_FILENAME_SUFFIX else 'channels'
        except ImportError:
            suffix = 'channelmap'

        bids_path = self._BIDS_path().update(
            suffix=suffix, extension='.tsv', datatype='ieeg',
            acquisition=ref,
        )
        # If we had to fall back to 'channels', rewrite the basename so
        # the file is still distinguishable from the real channels.tsv.
        out_path = bids_path.fpath
        if suffix != 'channelmap':
            out_dir = os.path.dirname(out_path)
            out_name = os.path.basename(out_path).replace('_channels.tsv', '_channelmap.tsv')
            out_path = os.path.join(out_dir, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._to_tsv(mapping_df, out_path)

        # Ensure .bidsignore includes channelmap files
        bidsignore = os.path.join(self.root, '.bidsignore')
        pattern = '**/*_channelmap.tsv'
        existing = set()
        if os.path.exists(bidsignore):
            existing = set(open(bidsignore).read().splitlines())
        if pattern not in existing:
            with open(bidsignore, 'a') as f:
                f.write(pattern + '\n')

    def write_BIDS_channels(self, ref):
        bids_path = self._BIDS_path().update(suffix='channels', extension='.tsv', datatype='ieeg')

        if ref == 'bipolar':
            bids_path.update(acquisition='bipolar')
            self._to_tsv(self.channels_bi, bids_path.fpath)
        elif ref == 'monopolar':
            bids_path.update(acquisition='monopolar')
            self._to_tsv(self.channels_mono, bids_path.fpath)

    # ---------- EEG ----------
    # set sfreq and recording_duration attributes
    def eeg_metadata(self):
        try:
            eeg = self.reader.load_eeg()
        except Exception as exc:
            # `load_eeg()` with no events asks for the whole file at once,
            # which fails on sessions with index/file length mismatches
            # (MissingDataError) or split-EEG bookkeeping issues
            # ("split EEG filenames don't seem to match"). Probe a small
            # window around the first event instead.
            from cmlreaders.exc import MissingDataError
            if not (isinstance(exc, MissingDataError) or "split EEG filenames" in str(exc)):
                raise
            events = self.reader.load("events")
            eeg = self.reader.load_eeg(events=events.iloc[[0]], rel_start=0, rel_stop=50)
        sfreq = eeg.samplerate
        recording_duration = eeg.data.shape[-1] / sfreq
        return sfreq, recording_duration
    # blob:vscode-webview://1nao9csoaulpj1p7mubgic192ukh97s57s733a1227t712dsg7ba/bca3f58f-d070-4788-b29f-d787e3189f36
    def eeg_sidecar(self, ref):
        sidecar = {'TaskName': self.experiment}
        # call super and then add TaskDescription for each experiment
        if self.system_version == 2.0 or self.system_version == 4.0:
            sidecar['Manufacturer'] = 'Blackrock'
        elif self.system_version >= 3.0 and self.system_version < 4.0:
            sidecar['Manufacturer'] = 'Medtronic'
        sidecar['SamplingFrequency'] = float(self.sfreq)
        sidecar['PowerLineFrequency'] = 60.0
        sidecar['SoftwareFilters'] = 'n/a'
        sidecar['HardwareFilters'] = 'n/a'
        sidecar['RecordingDuration'] = float(self.recording_duration)
        sidecar['RecordingType'] = 'continuous'
        #sidecar['ElectricalStimulation'] = False     # experiemtn dependent
        if ref == 'bipolar':
            sidecar['iEEGReference'] = 'bipolar'
            sidecar['ECOGChannelCount'] = len(self.pairs[(self.pairs.type_1=='S') | (self.pairs.type_1=='G')].index)
            sidecar['SEEGChannelCount'] = len(self.pairs[(self.pairs.type_1=='D') | (self.pairs.type_1=='uD')].index)
        elif ref == 'monopolar':
            sidecar['iEEGReference'] = 'monopolar'
            sidecar['ECOGChannelCount'] = len(self.contacts[(self.contacts.type=='S') | (self.contacts.type=='G')].index)
            sidecar['SEEGChannelCount'] = len(self.contacts[(self.contacts.type=='D') | (self.contacts.type=='uD')].index)
        return sidecar
    
    def _source_recording_path(self):
        """Return the path to the original recording file, or None.

        For system 1 sessions this is the source EDF under
        ``current_source/raw_eeg/``; for other systems the file at the
        same location may be NSx / HDF5 / etc. and the EDF priority
        branch in ``resolve_edf_units`` will simply skip it (pyedflib
        will fail to open it and return None).
        """
        index_path = os.path.join(
            "/protocols/r1/subjects",
            self.subject,
            "experiments",
            self.experiment,
            "sessions",
            str(self.session),
            "ephys",
            "current_source",
            "index.json",
        )
        if not os.path.exists(index_path):
            return None
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        files = (index.get("raw_eeg") or {}).get("files") or []
        if not files:
            return None
        candidate = os.path.join(os.path.dirname(index_path), files[0])
        return candidate if os.path.exists(candidate) else None

    def write_BIDS_ieeg(self, ref):
        """Write the iEEG file directly via pyedflib (no MNE round-trip).

        Monopolar sessions always write EDF int16. Bipolar sessions
        write EDF int16 when the per-pair int subtraction stays within
        ±32767, and BDF int24 otherwise. The EEG file write goes
        straight to disk in the recording's native LSB units; analyst-
        side MNE recovers Volts via the standard EDF/BDF gain formula
        plus the dimension multiplier.
        """
        if ref == "bipolar":
            data_int, labels, sfreq, container = self.eeg_bi
            sidecar_dict = self.eeg_sidecar_bi
        elif ref == "monopolar":
            data_int, labels, sfreq = self.eeg_mono
            container = "EDF"
            sidecar_dict = self.eeg_sidecar_mono
        else:
            raise ValueError(f"unknown ref {ref!r}")

        ext = ".edf" if container == "EDF" else ".bdf"
        bids_path = self._BIDS_path().update(
            suffix="ieeg", extension=ext, datatype="ieeg", acquisition=ref,
        )
        out_path = bids_path.fpath
        os.makedirs(out_path.parent, exist_ok=True)

        # Resolve per-channel units via the priority cascade.
        source_edf = self._source_recording_path() if ref == "monopolar" else None
        signal_units, units_status = resolve_edf_units(
            labels,
            source_edf_path=source_edf,
            conversion_to_V=float(self.unit_scale) if self.unit_scale else None,
            container=container,
            data_for_fallback=data_int,
        )
        if units_status == "derived":
            print(
                f"  WARN: derived units from data for {self.subject} "
                f"{self.experiment} ses-{self.session} ({ref})"
            )

        write_digital(
            str(out_path),
            labels,
            data_int,
            sfreq,
            signal_units,
            container=container,
        )

        # Sidecar JSON.
        with open(bids_path.update(extension=".json").fpath, "w") as f:
            json.dump(sidecar_dict, f, indent=2)

        # Append to scans.tsv (replacing any prior entry for this acquisition).
        self._update_scans_tsv(out_path)

        # Events sidecar (only on the first acquisition write — same as before).
        bids_path = self._BIDS_path().update(
            suffix="events", extension=".tsv", datatype="ieeg",
        )
        self._to_tsv(self.events, bids_path.fpath)
        with open(bids_path.update(extension=".json").fpath, "w") as f:
            json.dump(fp=f, obj=self.events_descriptor)

    def _update_scans_tsv(self, ieeg_file_path):
        """Append a row for the new iEEG file to ``scans.tsv``.

        BIDS spec: ``scans.tsv`` lists every recording in the session
        with a path relative to the session directory. We append rather
        than overwrite so multiple acquisitions in the same session
        coexist.
        """
        scans_tsv = mne_bids.BIDSPath(
            subject=self.subject,
            session=str(self.session),
            suffix="scans",
            extension=".tsv",
            root=self.root,
        ).fpath
        # Path relative to the session directory.
        rel_path = os.path.relpath(ieeg_file_path, scans_tsv.parent)
        new_row = pd.DataFrame([{"filename": rel_path}])
        if scans_tsv.exists():
            existing = pd.read_csv(scans_tsv, sep="\t")
            existing = existing[existing["filename"] != rel_path]
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            os.makedirs(scans_tsv.parent, exist_ok=True)
            combined = new_row
        combined.to_csv(scans_tsv, sep="\t", index=False)

    # ---------- EEG (monopolar) ----------
    def eeg_mono_to_BIDS(self):
        """Load monopolar digital ints via cmlreaders. No MNE, no Volts.

        Returns
        -------
        data_int16 : np.ndarray, shape (n_channels, n_samples)
            Raw integer samples in the source recording's native LSB.
        labels : list[str]
            Per-channel labels in the same order as ``data_int16`` rows.
        sfreq : float
            Sampling frequency in Hz.
        """
        # Pass scheme=self.contacts so the loaded EEG is aligned to the
        # filtered contacts (phantom contacts have already been removed
        # in run() via _filter_contacts_to_recording).
        eeg = self.reader.load_eeg(scheme=self.contacts)
        sfreq = float(eeg.samplerate)
        # cmlreaders' EEGContainer.data is shape (n_events, n_channels, n_samples)
        # for continuous loads → squeeze the singleton event dim.
        arr = np.asarray(eeg.data)
        if arr.ndim == 3:
            arr = np.squeeze(arr, axis=0)
        if arr.ndim != 2:
            raise ValueError(
                f"unexpected EEG shape from cmlreaders: {eeg.data.shape}"
            )
        # cmlreaders returns the raw LSB values as float64; cast back to int16.
        data_int16 = arr.astype(np.int16)
        labels = list(self.contacts.label)
        return data_int16, labels, sfreq

    # ---------- EEG (bipolar) ----------
    @staticmethod
    def _shorten_label(label, max_len):
        """Shorten a contact label to max_len, preserving trailing digits."""
        if len(label) <= max_len:
            return label
        m = re.match(r'^(.*?)(\d+)$', label)
        if m:
            prefix, digits = m.group(1), m.group(2)
            return prefix[:max_len - len(digits)] + digits
        return label[:max_len]

    @classmethod
    def _split_bipolar(cls, name):
        """Split a bipolar label into (contact1, separator, contact2).

        Handles edge cases:
          - underscore separators:  B'micro1_B'micro2
          - hyphens inside labels: RP-THAL1-RP-THAL2
        """
        # Try underscore separator first (e.g. B'micro1_B'micro2)
        if '_' in name:
            idx = name.index('_')
            return name[:idx], '_', name[idx + 1:]

        # For hyphens, find the split where both halves end with a digit
        positions = [i for i, c in enumerate(name) if c == '-']
        for pos in sorted(positions, key=lambda p: abs(p - len(name) / 2)):
            left, right = name[:pos], name[pos + 1:]
            if left and right and left[-1].isdigit() and right[-1].isdigit():
                return left, '-', right

        # Fallback: split at first hyphen
        if '-' in name:
            idx = name.index('-')
            return name[:idx], '-', name[idx + 1:]

        return name, '', ''

    @classmethod
    def _truncate_bipolar(cls, name):
        """Truncate a bipolar channel name to 16 chars, preserving trailing digits."""
        left, sep, right = cls._split_bipolar(name)
        if not sep:
            return cls._shorten_label(name, 16)
        max_half = (16 - len(sep)) // 2  # 7 for '-'/'_'
        return cls._shorten_label(left, max_half) + sep + cls._shorten_label(right, max_half)

    @classmethod
    def _truncate_channel_names(cls, raw_mne):
        """Shorten channel names that exceed the 16-char EDF/BDF limit."""
        renames = {}
        for name in raw_mne.ch_names:
            if len(name) > 16:
                renames[name] = cls._truncate_bipolar(name)
        if renames:
            raw_mne.rename_channels(renames)

    def eeg_bi_to_BIDS(self):
        """Load bipolar digital ints via cmlreaders. No MNE, no Volts.

        cmlreaders' ``load_eeg(scheme=self.pairs)`` already does the
        per-pair subtraction in the source-int domain, so the returned
        ``eeg.data`` is the bipolar LSB values (just stored as float64).
        Most pairs fit comfortably in int16, but a worst-case
        opposing-rail pair *could* exceed ±32767. We detect that here
        and report which container the writer should use.

        Returns
        -------
        data_int : np.ndarray, shape (n_channels, n_samples)
            Bipolar integer samples. ``int16`` if every value fits in
            int16, else ``int32`` (writer will narrow to int24 BDF).
        labels : list[str]
            Per-pair labels (truncated to 16 chars to satisfy the
            EDF/BDF spec — same truncation as
            ``pairs_to_channels``).
        sfreq : float
            Sampling frequency in Hz.
        container : str
            ``"EDF"`` or ``"BDF"``. The caller writes the file with the
            corresponding extension.
        """
        eeg = self.reader.load_eeg(scheme=self.pairs)
        sfreq = float(eeg.samplerate)
        arr = np.asarray(eeg.data)
        if arr.ndim == 3:
            arr = np.squeeze(arr, axis=0)
        if arr.ndim != 2:
            raise ValueError(
                f"unexpected bipolar EEG shape from cmlreaders: {eeg.data.shape}"
            )
        # Range check before narrowing.
        obs_min = float(arr.min())
        obs_max = float(arr.max())
        if obs_min >= -32768 and obs_max <= 32767:
            data_int = arr.astype(np.int16)
            container = "EDF"
        else:
            data_int = arr.astype(np.int32)
            container = "BDF"
            print(
                f"  bipolar overflow for {self.subject} {self.experiment} ses-{self.session}: "
                f"[{obs_min:.0f}, {obs_max:.0f}] → promoting to BDF int24"
            )

        labels = [
            self._truncate_bipolar(n) if len(n) > 16 else n
            for n in np.array(self.pairs.label)
        ]
        return data_int, labels, sfreq, container
    
    # ----------------------------------------
    # run conversion
    def run(self):
        # ---------- Events ----------
        self.reader = self.cml_reader()
        self.wordpool_file = self.set_wordpool()
        self.events = self.events_to_BIDS()
        self.events_descriptor = self.make_events_descriptor()
        self.write_BIDS_beh()

        # terminate if no monopolar or bipolar EEG
        if not self.monopolar and not self.bipolar:
            return True

        # ---------- EEG ----------
        self.sfreq, self.recording_duration = self.eeg_metadata()

        # ---------- Electrodes ----------
        self.contacts = self.load_contacts()
        # Filter contacts to only those in the recording. Keep the full
        # set for electrodes.tsv (physical positions); use the filtered
        # set for channels.tsv and EEG labels.
        self.contacts_all = self.contacts.copy()
        self.contacts, self.contacts_dropped = self._filter_scheme_to_recording(self.contacts, "contacts")

        # Check which coordinate systems are actually available
        has_mni = self.mni and 'mni.x' in self.contacts_all.columns
        has_tal = self.tal and 'tal.x' in self.contacts_all.columns
        if self.mni and not has_mni:
            print(f"WARNING: MNI coordinates missing for {self.subject} — skipping MNI output")
        if self.tal and not has_tal:
            print(f"WARNING: Talairach coordinates missing for {self.subject} — skipping Tal output")

        # both MNI and Talairach
        if has_mni and has_tal:
            self.electrodes = self.contacts_to_electrodes('mni', True)
            self.electrodes_sidecar = self.make_electrodes_sidecar('mni', True)
            self.write_BIDS_electrodes('mni', True)
            self.write_BIDS_coords('mni')
        elif has_mni:
            self.electrodes_mni = self.contacts_to_electrodes('mni', False)
            self.electrodes_sidecar = self.make_electrodes_sidecar('mni', False)
            self.write_BIDS_electrodes('mni', False)
            self.write_BIDS_coords('mni')
        elif has_tal:
            self.electrodes_tal = self.contacts_to_electrodes('tal', False)
            self.electrodes_sidecar = self.make_electrodes_sidecar('tal', False)
            self.write_BIDS_electrodes('tal', False)
            self.write_BIDS_coords('tal')

        # ---------- Channels ----------
        self.pairs = self.load_pairs()
        if self.bipolar:
            self.pairs_all = self.pairs.copy()
            self.pairs, self.pairs_dropped = self._filter_scheme_to_recording(self.pairs, "pairs")
            self.channels_bi = self.pairs_to_channels()
        if self.monopolar:
            self.channels_mono = self.contacts_to_channels()

        # ---------- EEG ----------
        if self.bipolar:
            self.eeg_sidecar_bi = self.eeg_sidecar('bipolar')
            self.eeg_bi = self.eeg_bi_to_BIDS()
            self.write_BIDS_ieeg('bipolar')
            self.write_BIDS_channels('bipolar')          # write bipolar channels to BIDS format (overwrite automatic)
            self.write_BIDS_channelmap('bipolar')        # log any truncated channel names
        if self.monopolar:
            self.eeg_sidecar_mono = self.eeg_sidecar('monopolar')
            self.eeg_mono = self.eeg_mono_to_BIDS()
            self.write_BIDS_ieeg('monopolar')
            self.write_BIDS_channels('monopolar')        # write monopolar channels to BIDS format (overwrite automatic)

        return True
