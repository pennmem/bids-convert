# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from BIDS_utils import BIDS_path


class intracranial_BIDS_converter:
    # class 
    ELEC_TYPES_DESCRIPTION = {'S': 'strip', 'G': 'grid', 'D': 'depth', 'uD': 'micro'}
    ELEC_TYPES_BIDS = {'S': 'ECOG', 'G': 'ECOG', 'D': 'SEEG', 'uD': 'SEEG'}

    # initialize
    def __init__(self, subject, experiment, session, root='/scratch/hherrema/BIDS_storage/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.root = root
    
        
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
        bids_path = BIDS_path().update(suffix='beh', extension='.tsv', datatype='beh')
        os.makedirs(bids_path.directory, exist_ok=True)
        
        # write events to tsv
        self.events.to_csv(bids_path.fpath, sep='\t', index=False)

        # write sidecar json
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)

    # ---------- Electrodes ---------- 
    def load_contacts(self):
        return self.reader.load('contacts')
    
    def generate_area_map(self):
        raise NotImplementedError       # implement here
    
    # convert CML contacts to BIDS electrodes
    def contacts_to_electrodes(self, atlas):
        # MAY NEED EDGE CASES
        electrodes = pd.DataFrame({'name': np.array(self.contacts.label)})      # name = label of contact
        electrodes['x'] = self.contacts[f'{atlas}.x']
        electrodes['y'] = self.contacts[f'{atlas}.y']
        electrodes['z'] = self.contacts[f'{atlas}.z']
        electrodes['group'] = [re.sub('\d+', '', x) for x in self.contacts.label]
        if self.area:       # use area data if available
            electrodes['size'] = [self.area_map.get(x) if x in self.area_map.keys() else -999 for x in electrodes.group]
        else:
            electrodes['size'] = -999
        electrodes['hemisphere'] = ['L' if x < 0 else 'R' if x > 0 else 'n/a' for x in electrodes.x]        # use coordinates for hemisphere
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]
        electrodes = electrodes.fillna('n/a')                                   # remove NaN
        electrodes = electrodes.replace('', 'n/a')                              # resolve empty cell issue
        electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type']]          # re-order columns
        return electrodes
    
    def write_BIDS_electrodes(self, atlas):
        bids_path = BIDS_path().update(suffix='electrodes', extension='.tsv', datatype='ieeg')
        # COULD DO AWAY WITH SEPARATE tal AND mni ATTRIBUTES
        if atlas == 'tal':
            bids_path.update(space='Talairach')
            os.makedirs(bids_path.directory, exist_ok=True)
            self.electrodes_tal.to_csv(bids_path.fpath, sep='\t', index=False)
        elif atlas == 'mni':
            bids_path.update(space='MNI152NLin6ASym')
            os.makedirs(bids_path.directory, exist_ok=True)
            self.electrodes_mni.to_csv(bids_path.fpath, sep='\t', index=False)

    def _coordinate_system(self, atlas):
        if atlas == 'tal':
            return {'iEEGCoordinateSystem': 'Talairach', 'iEEGCoordinateUnits': 'mm'}
        elif atlas == 'mni':
            return {'iEEGCoordinateSystem': 'MNI152NLin6ASym', 'iEEGCoordinateUnits': 'mm'}
        
    def write_BIDS_coords(self, atlas):
        bids_path = BIDS_path().update(suffix='coordsystem', extension='.json', datatype='ieeg')
        coord_sys = _coordinate_system(self, atlas)
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
        bids_path = BIDS_path().update(suffix='channels', extension='.tsv', datatype='ieeg')

        if ref == 'bipolar':
            bids_path.update(acquisition='bipolar')
            self.channels_bi.to_csv(bids_path.fpath, sep='\t', index=False)
        elif ref == 'monopolar':
            bids_path.update(acquisition='monopolar')
            self.channels_mono.to_csv(bids_path.fpath, sep='\t', index=False)

    # ---------- EEG ----------
    # set sfreq and recording_duration attributes
    def eeg_metadata(self):
        eeg = self.reader.load_eeg()
        sfreq = eeg.samplerate
        recording_duration = eeg.data.shape[-1] / self.sfreq
        return sfreq, recording_duration
    
    def eeg_sidecar(self, ref):
        sidecar = {'TaskName': self.experiment}
        # call super and then add TaskDescription for each experiment
        #sidecar['TaskDescription'] = 'delayed free recall of categorized word lists'
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
        bids_path = BIDS_path().update(suffix='ieeg', extension='.edf', datatype='ieeg')

        if ref == 'bipolar':
            raise NotImplementedError
        elif ref == 'monopolar':
            raise NotImplementedError
        
        # also write events
        bids_path = BIDS_path().update(suffix='events', extension='.tsv', datatype='ieeg')
        self.events.to_csv(bids_path.fpath, sep='\t', index=False)
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)

    # ---------- EEG (monopolar) ----------
    def eeg_mono_to_BIDS(self):
        raise NotImplementedError
    
    # ---------- EEG (bipolar) ----------
    def eeg_bi_to_BIDS(self):
        raise NotImplementedError