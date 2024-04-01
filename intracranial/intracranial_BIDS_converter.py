# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids


class intracranial_BIDS_converter:
    ELEC_TYPES_DESCRIPTION = {'S': 'strip', 'G': 'grid', 'D': 'depth', 'uD': 'micro'}
    ELEC_TYPES_BIDS = {'S': 'ECOG', 'G': 'ECOG', 'D': 'SEEG', 'uD': 'SEEG'}
    BRAIN_REGIONS = ['wb.region', 'ind.region', 'das.region', 'stein.region']

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS_storage/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.system_version = system_version
        self.unit_scale = unit_scale
        self.monopolar = monopolar
        self.bipolar = bipolar
        self.mni = mni
        self.tal = tal
        self.area = area
        if self.area:
            self.area_map = self.generate_area_map()
        self.brain_regions = brain_regions   # dictionary mapping target regions to number of non-NaN contacts
        self.root = root

    # ---------- BIDS Utility ----------
    # return a base BIDS_path object to update
    def _BIDS_path(self):
        bids_path = mne_bids.BIDSPath(subject=self.subject, task=self.experiment, session=str(self.session),
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
            self.area_map = dict(zip(area[:, 0], area[:, 1].astype(float)))
        except BaseException:
            self.area_map = {}
    
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
        for br in self.BRAIN_REGIONS:
            if self.brain_regions[br] > 0:
                electrodes[br] = self.contacts[br]
        
        # both MNI and Talairach --> default to MNI first, manually define Tal
        if toggle:
            electrodes['tal.x'] = self.contacts['tal.x']
            electrodes['tal.y'] = self.contacts['tal.y']
            electrodes['tal.z'] = self.contacts['tal.z']
        
        electrodes = electrodes.fillna('n/a')                                   # remove NaN
        electrodes = electrodes.replace('', 'n/a')                              # resolve empty cell issue
        
        if toggle:
            electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type',
                                     'tal.x', 'tal.y', 'tal.z']]
        else:
            electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type']]          # re-order columns
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

    def _coordinate_system(self, atlas, toggle):
        if toggle:
            return {'iEEGCoordinateSystem': 'MNI152NLin6ASym', 'iEEGCoordinateUnits': 'mm',
                    'iEEGCoordinateSystem_': 'Talairach', 'iEEGCoordinateUnits_': 'mm'}
        elif atlas == 'tal':
            return {'iEEGCoordinateSystem': 'Talairach', 'iEEGCoordinateUnits': 'mm'}
        elif atlas == 'mni':
            return {'iEEGCoordinateSystem': 'MNI152NLin6ASym', 'iEEGCoordinateUnits': 'mm'}
        
    def write_BIDS_coords(self, atlas, toggle):
        bids_path = self._BIDS_path().update(suffix='coordsystem', extension='.json', datatype='ieeg')
        coord_sys = self._coordinate_system(atlas, toggle)
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
        channels['units'] = 'V'
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
        channels['units'] = 'V'
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
    
    def write_BIDS_ieeg(self, ref):
        bids_path = self._BIDS_path().update(suffix='ieeg', extension='.edf', datatype='ieeg')

        if ref == 'bipolar':
            bids_path = bids_path.update(acquisition='bipolar')
            mne_bids.write_raw_bids(self.eeg_bi, bids_path=bids_path, events=None, allow_preload=True, format='EDF', overwrite=True)
            mne_bids.update_sidecar_json(bids_path.update(extension='.json'), self.eeg_sidecar_bi)
        elif ref == 'monopolar':
            bids_path = bids_path.update(acquisition='monopolar')
            mne_bids.write_raw_bids(self.eeg_mono, bids_path=bids_path, events=None, allow_preload=True, format='EDF', overwrite=True)
            mne_bids.update_sidecar_json(bids_path.update(extension='.json'), self.eeg_sidecar_mono)
        
        # also write events
        bids_path = self._BIDS_path().update(suffix='events', extension='.tsv', datatype='ieeg')
        self._to_tsv(self.events, bids_path.fpath)
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)

    # ---------- EEG (monopolar) ----------
    def eeg_mono_to_BIDS(self):
        eeg = self.reader.load_eeg(scheme=self.contacts)
        eeg.data = eeg.data / self.unit_scale              # convert to V before instantiating raw object
        eeg_mne = eeg.to_mne()
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_mono.type]))    # ecog or seeg
        eeg_mne.set_channel_types(mapping)                                                     # set channel types

        return eeg_mne
    
    # ---------- EEG (bipolar) ----------
    def eeg_bi_to_BIDS(self):
        eeg = self.reader.load_eeg(scheme=self.pairs)
        eeg.data = eeg.data / self.unit_scale               # convert to V before instantiating raw object
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

        # ---------- EEG ----------
        self.sfreq, self.recording_duration = self.eeg_metadata()

        # ---------- Electrodes ----------
        self.contacts = self.load_contacts()

        # both MNI and Talairach
        if self.mni and self.tal:
            self.electrodes = self.contacts_to_electrodes('mni', True)
            self.electrodes_sidecar = self.make_electrodes_sidecar('mni', True)
            self.write_BIDS_electrodes('mni', True)
            self.write_BIDS_coords('mni', True)
        elif self.mni:
            self.electrodes_mni = self.contacts_to_electrodes('mni', False)
            self.electrodes_sidecar = self.make_electrodes_sidecar('mni', False)
            self.write_BIDS_electrodes('mni', False)
            self.write_BIDS_coords('mni', False)
        elif self.tal:
            self.electrodes_tal = self.contacts_to_electrodes('tal', False)
            self.electrodes_sidecar = self.make_electrodes_sidecar('tal', False)
            self.write_BIDS_electrodes('tal', False)
            self.write_BIDS_coords('tal', False)

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
