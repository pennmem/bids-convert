# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
from glob import glob
import mne_bids


class intracranial_BIDS_converter:
    ELEC_TYPES_DESCRIPTION = {'S': 'strip', 'G': 'grid', 'D': 'depth', 'uD': 'micro'}
    ELEC_TYPES_BIDS = {'S': 'ECOG', 'G': 'ECOG', 'D': 'SEEG', 'uD': 'SEEG'}
    BRAIN_REGIONS = ['wb.region', 'ind.region', 'das.region', 'stein.region']

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/'):
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
        return self.reader.load('contacts')
    
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
                areas = pd.read_csv(config_files[0], names=['label', 'index', 'area'], header=None)
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
        # MAY NEED EDGE CASES
        electrodes = pd.DataFrame({'name': np.array(self.contacts.label)})      # name = label of contact
        electrodes['x'] = self.contacts[f'{atlas}.x']
        electrodes['y'] = self.contacts[f'{atlas}.y']
        electrodes['z'] = self.contacts[f'{atlas}.z']
        
        # electrode groups (shanks)
        groups = []
        for _, row in self.contacts.iterrows():
            split_label = list(row.label)
            #dig_idx = [i for i, c in enumerate(split_label) if c.isdigit()]     # indicies of digits in label
            alpha_idx = [i for i, c in enumerate(split_label) if c.isalpha()]   # indices of alphabetical characters in label
            groups.append(row.label[:alpha_idx[-1]+1])                          # select values before final digits
        electrodes['group'] = groups
        #electrodes['group'] = [re.sub('\d+', '', x) for x in self.contacts.label]
        
        if self.area:       # use area data if available
            electrodes['size'] = [self.area_map.get(x) if x in self.area_map.keys() else -999 for x in electrodes.group]
        else:
            electrodes['size'] = -999
        
        electrodes['hemisphere'] = ['L' if x < 0 else 'R' if x > 0 else 'n/a' for x in electrodes.x]        # use coordinates for hemisphere
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]
        
        # anatomical regions
        br_cols = []
        for br in self.BRAIN_REGIONS:
            if self.brain_regions[br] > 0:
                if br not in self.contacts.columns:
                    print(f"WARNING: brain region column '{br}' not found in contacts for {self.subject} — omitting from electrodes TSV.")
                    continue
                electrodes[br] = self.contacts[br]
                br_cols.append(br)
        
        # both MNI and Talairach --> default to MNI first, manually define Tal
        if toggle:
            electrodes['tal.x'] = self.contacts['tal.x']
            electrodes['tal.y'] = self.contacts['tal.y']
            electrodes['tal.z'] = self.contacts['tal.z']
        
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
        # MAY NEED EDGE CASES
        channels = pd.DataFrame({'name': np.array(self.pairs.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.pairs.type_1]      # all pairs have same type
        channels['units'] = 'uV'
        channels['low_cutoff'] = 'n/a'                 # highpass filter
        channels['high_cutoff'] = 'n/a'                # lowpass filter (mne adds Nyquist frequency = 2 x sampling rate)
        channels['reference'] = 'bipolar'
        channels['group'] = [re.sub('\d+', '', x).split('-')[0] for x in self.pairs.label]
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.pairs.type_1]    # all pairs have same type
        channels['notch'] = 'n/a'
        channels = channels.fillna('n/a')              # remove NaN
        channels = channels.replace('', 'n/a')         # resolve empty cell issue
        return channels
    
    # convert CML contacts to BIDS channels (monopolar)
    def contacts_to_channels(self):
        # MAY NEED EDGE CASES
        channels = pd.DataFrame({'name': np.array(self.contacts.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.contacts.type]
        channels['units'] = 'uV'
        channels['low_cutoff'] = 'n/a'
        channels['high_cutoff'] = 'n/a'
        channels['group'] = [re.sub('\d+', '', x) for x in self.contacts.label]
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]
        channels['notch'] = 'n/a'
        channels = channels.fillna('n/a')              # remove NaN
        channels = channels.replace('', 'n/a')         # resolve empty cell issue
        return channels
    
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
        eeg = self.reader.load_eeg()
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
    
    @staticmethod
    def _export_bdf(raw, bdf_path):
        """Export MNE Raw to BDF (24-bit) for higher precision than EDF (16-bit).

        BDF uses 24-bit integers (16,777,214 levels vs EDF's 65,534),
        giving 256x more precision and eliminating quantization errors
        for subjects recorded at fine resolution (0.1 µV, 250 nV).
        """
        from EDFlib.edfwriter import EDFwriter
        from mne.export._edf import _auto_close, _try_to_set_value

        units = dict(
            eeg="uV", ecog="uV", seeg="uV", eog="uV",
            ecg="uV", emg="uV", bio="uV", dbs="uV",
        )
        digital_min, digital_max = -8388607, 8388607

        raw.load_data()
        ch_names = raw.ch_names
        ch_types = np.array(raw.get_channel_types(picks=ch_names))
        n_channels = len(ch_names)
        sfreq = raw.info["sfreq"]
        out_sfreq = int(sfreq) if float(sfreq).is_integer() else int(np.floor(sfreq))
        data = raw.get_data(units=units, picks=ch_names)

        # physical range per channel type
        ch_types_phys_max, ch_types_phys_min = {}, {}
        for _type in np.unique(ch_types):
            _picks = np.nonzero(ch_types == _type)[0]
            ch_types_phys_max[_type] = data[_picks].max()
            ch_types_phys_min[_type] = data[_picks].min()

        file_type = EDFwriter.EDFLIB_FILETYPE_BDFPLUS
        with _auto_close(EDFwriter(str(bdf_path), file_type, n_channels)) as hdl:
            for idx, ch in enumerate(ch_names):
                ch_type = ch_types[idx]
                for key, val in [
                    ("PhysicalMaximum", ch_types_phys_max[ch_type]),
                    ("PhysicalMinimum", ch_types_phys_min[ch_type]),
                    ("DigitalMaximum", digital_max),
                    ("DigitalMinimum", digital_min),
                    ("PhysicalDimension", "uV"),
                    ("SampleFrequency", out_sfreq),
                    ("SignalLabel", ch),
                ]:
                    _try_to_set_value(hdl, key, val, channel_index=idx)

            meas_date = raw.info["meas_date"]
            if meas_date is not None:
                hdl.setStartDateTime(
                    meas_date.year, meas_date.month, meas_date.day,
                    meas_date.hour, meas_date.minute, meas_date.second,
                )

            for idx in range(n_channels):
                hdl.writeSamples(data[idx])

    def write_BIDS_ieeg(self, ref):
        # Write EDF first to generate all BIDS sidecars (channels.tsv, scans.tsv, etc.)
        bids_path = self._BIDS_path().update(suffix='ieeg', extension='.edf', datatype='ieeg')

        if ref == 'bipolar':
            bids_path = bids_path.update(acquisition='bipolar')
            mne_bids.write_raw_bids(self.eeg_bi, bids_path=bids_path, events=None, allow_preload=True, format='EDF', overwrite=True)
            mne_bids.update_sidecar_json(bids_path.update(extension='.json'), self.eeg_sidecar_bi)
            raw = self.eeg_bi
        elif ref == 'monopolar':
            bids_path = bids_path.update(acquisition='monopolar')
            mne_bids.write_raw_bids(self.eeg_mono, bids_path=bids_path, events=None, allow_preload=True, format='EDF', overwrite=True)
            mne_bids.update_sidecar_json(bids_path.update(extension='.json'), self.eeg_sidecar_mono)
            raw = self.eeg_mono

        # Replace EDF with BDF for 24-bit precision
        edf_path = bids_path.fpath
        bdf_path = edf_path.with_suffix('.bdf')
        self._export_bdf(raw, bdf_path)
        edf_path.unlink()

        # Update scans.tsv to reference .bdf instead of .edf
        scans_tsv = mne_bids.BIDSPath(
            subject=self.subject, session=str(self.session),
            suffix='scans', extension='.tsv', root=self.root,
        ).fpath
        if scans_tsv.exists():
            scans = pd.read_csv(scans_tsv, sep='\t')
            scans['filename'] = scans['filename'].str.replace('.edf', '.bdf', regex=False)
            scans.to_csv(scans_tsv, sep='\t', index=False)
        
        # also write events
        bids_path = self._BIDS_path().update(suffix='events', extension='.tsv', datatype='ieeg')
        self._to_tsv(self.events, bids_path.fpath)
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)

    # ---------- EEG (monopolar) ----------
    def eeg_mono_to_BIDS(self):
        eeg = self.reader.load_eeg(scheme=self.contacts)
        # Two-step conversion to avoid int16 precision loss from a single large divisor:
        #   Step 1: normalize raw units to 1 µV (e.g. divide by 10 for 0.1µV, by 4 for 250nV)
        #   Step 2: convert µV to V for MNE's internal representation
        # Two-step conversion to preserve precision:
        #   Step 1: normalize raw units to 1 µV (small divisor: 10 for 0.1µV, 4 for 250nV)
        #   Step 2: convert µV to V for MNE (MNE's EDF exporter converts V->µV when writing)
        eeg.data = eeg.data / int(self.unit_scale / 1_000_000)               # raw units -> µV
        eeg.data = eeg.data / 1_000_000                                      # µV -> V

        eeg_mne = eeg.to_mne()
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_mono.type]))    # ecog or seeg
        eeg_mne.set_channel_types(mapping)                                                     # set channel types

        return eeg_mne

    # ---------- EEG (bipolar) ----------
    def eeg_bi_to_BIDS(self):
        eeg = self.reader.load_eeg(scheme=self.pairs)
        eeg.data = eeg.data / int(self.unit_scale / 1_000_000)               # raw units -> µV
        eeg.data = eeg.data / 1_000_000                                      # µV -> V

        eeg_mne = eeg.to_mne()
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_bi.type]))
        eeg_mne.set_channel_types(mapping)

        return eeg_mne
    
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

        # both MNI and Talairach
        if self.mni and self.tal:
            self.electrodes = self.contacts_to_electrodes('mni', True)
            self.electrodes_sidecar = self.make_electrodes_sidecar('mni', True)
            self.write_BIDS_electrodes('mni', True)
            self.write_BIDS_coords('mni')
        elif self.mni:
            self.electrodes_mni = self.contacts_to_electrodes('mni', False)
            self.electrodes_sidecar = self.make_electrodes_sidecar('mni', False)
            self.write_BIDS_electrodes('mni', False)
            self.write_BIDS_coords('mni')
        elif self.tal:
            self.electrodes_tal = self.contacts_to_electrodes('tal', False)
            self.electrodes_sidecar = self.make_electrodes_sidecar('tal', False)
            self.write_BIDS_electrodes('tal', False)
            self.write_BIDS_coords('tal')

        # ---------- Channels ----------
        self.pairs = self.load_pairs()
        if self.bipolar:
            self.channels_bi = self.pairs_to_channels()
        if self.monopolar:
            self.channels_mono = self.contacts_to_channels()

        # ---------- EEG ----------
        if self.bipolar:
            self.eeg_sidecar_bi = self.eeg_sidecar('bipolar')
            self.eeg_bi = self.eeg_bi_to_BIDS()
            self.write_BIDS_ieeg('bipolar')
            self.write_BIDS_channels('bipolar')          # write bipolar channels to BIDS format (overwrite automatic)
        if self.monopolar:
            self.eeg_sidecar_mono = self.eeg_sidecar('monopolar')
            self.eeg_mono = self.eeg_mono_to_BIDS()
            self.write_BIDS_ieeg('monopolar')
            self.write_BIDS_channels('monopolar')        # write monopolar channels to BIDS format (overwrite automatic)

        return True
