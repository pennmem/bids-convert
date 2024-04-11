# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
import scipy
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class pyFR_BIDS_converter(intracranial_BIDS_converter):
    wordpool_EN = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/pyFR/wordpools/wordpool_EN.txt', dtype=str)

    # initialize
    # just hand empty dictionary for brain_regions
    def __init__(self, subject, experiment, session, montage, math_evs, system_version, unit_scale, monopolar, bipolar, mni, tal, root='/scratch/hherrema/BIDS/pyFR/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.montage = montage
        self.math_evs = math_evs
        self.system_version = system_version
        self.unit_scale = unit_scale
        self.monopolar = monopolar
        self.bipolar = bipolar
        self.mni = mni
        self.tal = tal
        self.root = root

        # update session if re-implant
        new_session = self.reassign_session()
        if new_session:
            self.session = new_session

    # ---------- Events ----------
    def reassign_session(self):
        re_implants = pd.read_csv('pyFR/metadata/re_implants.csv')
        ri = re_implants[(re_implants.subject == self.subject) & (re_implants.montage == self.montage) &
                         (re_implants.session == self.session)]
        if len(ri) == 1:
            return ri.iloc[0].new_session
        else:
            return None

    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']
        if np.all([x in self.wordpool_EN for x in word_evs.item]):
            wordpool_file = 'wordpools/wordpool_EN.txt'
        else:
            wordpool_file = 'n/a'
        
        return wordpool_file
    
    def events_to_BIDS(self):
        evs = self.reader.load('events')
        # load in math events
        if self.math_evs:
            if self.montage != 0:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{self.subject}_{self.montage}_math.mat', squeeze_me=True)['events'])
            else:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{self.subject}_math.mat', squeeze_me=True)['events'])
            math_evs = math_evs[math_evs.session == self.session]                                        # select out session
            math_evs = math_evs[(math_evs.type != 'B') & (math_evs.type != 'E')]                         # remove the B and E events from math evs
            math_evs['list'] = math_evs['list'] - 1                                                      # math events given list + 1
            events = pd.concat([math_evs, evs], ignore_index=True)
            events = events.sort_values(by='mstime', ascending=True, ignore_index=True)                  # sort in chronological order
        events['experiment'] = self.experiment                                                       # math events don't have experiment field
        # transformations
        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})                  # rename columns
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0        # event duration [ms]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                  # replace events with negative duration with 0.0 s
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                           # onset from first event [ms]
        events['response_time'] = 'n/a'                                                              # response time [ms]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') |
                   (events.trial_type=='PROB'), 'response_time'] = events['rectime'] / 1000.0
        events['stim_file'] = np.where(events.trial_type=='WORD', self.wordpool_file, 'n/a')              # add wordpool to word events
        events['item_name'] = events.item.replace('X', 'n/a')
        events = events.fillna('n/a')                                                                # chnage NaN to 'n/a'
        events = events.replace('', 'n/a')                                                           # no empty cells

        # update session if re-implant
        re_implants = pd.read_csv('pyFR/metadata/re_implants.csv')
        ri = re_implants[(re_implants.subject == self.subject) & (re_implants.montage == self.montage) &
                         (re_implants.session == self.session)]
        if len(ri) == 1:
            events['session'] = ri.iloc[0].new_session
        
        if self.math_evs:
            events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 
                            'stim_file', 'item_name', 'serialpos', 'list', 'test', 'answer', 
                            'experiment', 'session', 'subject']]                                        # re-order columns + drop unneeded fields
        else:
            events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time',
                             'stim_file', 'item_name', 'serialpos', 'list',
                             'experiment', 'session', 'subject']]
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
            'REC_WORD_VV': 'Vocalization, onset of speech (during free recall).'
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category", 
                           "Description": "Indicator of type of task action that occurs at the marked time", 
                           "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()}},
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
                          'Description': 'The order position of a word presented in an WORD event.'},
            'test': {"LongName": "Math problem", 
                     "Description": "Math problem with form X + Y + Z = ?  Stored in list [X, Y, Z]."},
            'answer': {"LongName": "Math problem response", 
                       "Description": "Participant answer to problem with form X + Y + Z = ?  Note this is not necessarily the correct answer."}
        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    
    # ---------- Electrodes ----------
    def load_contacts(self):
        contacts = self.reader.load('events')
        contacts['type'] = [x if type(x)==str else 'n/a' for x in contacts.type]      # replace missing types with 'n/a'
        return contacts
    
    def contacts_to_electrodes(self, atlas, toggle):
        electrodes = pd.DataFrame({'name': np.array(self.contacts.label)})             # name = label of contact
        if toggle:    # both tal and mni
            electrodes['x'] = self.contacts['x'].astype(float)
            electrodes['y'] = self.contacts['y'].astype(float)
            electrodes['z'] = self.contacts['z'].astype(float)
            if self.montage != 0:
                mni_coords = np.loadtxt(f'/data/eeg/{self.subject}_{self.montage}/tal/RAW_coords.txt.mni')
            else:
                mni_coords = np.loadtxt(f'/data/eeg/{self.subject}/tal/RAW_coords.txt.mni')
            contacts_mask = []
            for i, c in enumerate(mni_coords[:,0]):
                if int(c) in np.array(self.contacts.contact):
                    contacts_mask.append(i)
            mni_contacts = mni_coords[contacts_mask, :]
            electrodes['mni.x'] = mni_contacts[:,1]
            electrodes['mni.y'] = mni_contacts[:, 2]
            electrodes['mni.z'] = mni_contacts[:, 3]
        elif atlas == 'tal':                                                           # tal coordinates from dataframe
            electrodes['x'] = self.contacts['x'].astype(float)
            electrodes['y'] = self.contacts['y'].astype(float)
            electrodes['z'] = self.contacts['z'].astype(float)
        elif atlas == 'mni':                                                           # manually read in mni coordinates
            if self.montage != 0:
                mni_coords = np.loadtxt(f'/data/eeg/{self.subject}_{self.montage}/tal/RAW_coords.txt.mni')
            else:
                mni_coords = np.loadtxt(f'/data/eeg/{self.subject}/tal/RAW_coords.txt.mni')
            contacts_mask = []
            for i, c in enumerate(mni_coords[:,0]):
                if int(c) in np.array(self.contacts.contact):
                    contacts_mask.append(i)
            mni_contacts = mni_coords[contacts_mask, :]
            electrodes['x'] = mni_contacts[:,1]
            electrodes['y'] = mni_contacts[:,2]
            electrodes['z'] = mni_contacts[:,3]

        electrodes['size'] = -999
        electrodes['group'] = np.array(self.contacts.grpName)
        electrodes['hemisphere'] = ['L' if 'Left' in x else 'R' if 'Right' in x else 'n/a' for x in self.contacts.Loc1]
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]
        electrodes['lobe'] = np.array(self.contacts.Loc2)
        electrodes['region1'] = np.array(self.contacts.Loc3)
        electrodes['region2'] = np.array(self.contacts.Loc5)
        electrodes['gray_white'] = np.array(self.contacts.Loc4)
        electrodes = electrodes.fillna('n/a')                                          # remove NaN
        electrodes = electrodes.replace('', 'n/a')                                     # no empty cells
        if toggle:
            electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type', 
                                     'lobe', 'region1', 'region2', 'gray_white', 'mni.x', 'mni.y', 'mni.z',]]
        else:
            electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type', 
                                    'lobe', 'region1', 'region2', 'gray_white']]      # enforce column order
        return electrodes
    
    def make_electrodes_sidecar(self, atlas, toggle):
        sidecar = {'name': 'Label of electrode'}
        if atlas == 'tal':
            sidecar['x'] = 'x-axis position in Talairach coordinates'
            sidecar['y'] = 'y-axis position in Talairach coordinates'
            sidecar['z'] = 'z-axis position in Talairach coordinates'
        elif atlas == 'mni':
            sidecar['x'] = 'x-axis position in MNI coordinates'
            sidecar['y'] = 'y-axis position in MNI coordinates'
            sidecar['z'] = 'z-axis position in MNI coordinates'
        sidecar['size'] = 'Surface area of electrode.'
        sidecar['group'] = 'Group of channels electrode belongs to (same shank).'
        sidecar['hemisphere'] = 'Hemisphere of electrode location.'
        sidecar['type'] = 'Type of electrode.'
        sidecar['lobe'] = 'Brain lobe of electrode location.'
        sidecar['region1'] = 'Brain region of electrode location.'
        sidecar['region2'] = 'Brain region of electrode location.'
        sidecar['gray_white'] = 'Denotes gray or white matter.'

        # both Tal and MNI
        if toggle:
            sidecar['mni.x'] = 'x-axis position in MNI coordinates'
            sidecar['mni.y'] = 'y-axis position in MNI coordinates'
            sidecar['mni.z'] = 'z-axis position in MNI coordinates'

        return sidecar
    
    def write_BIDS_electrodes(self, atlas, toggle):
        bids_path = self._BIDS_path().update(suffix='electrodes', extension='.tsv', datatype='ieeg')
        # both Tal and MNI, default to Tal (mirror cmlreaders)
        if toggle:
            bids_path.update(space='Talairach')
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
    
    # ---------- Channels ----------
    def load_pairs(self):
        pairs = self.reader.load('pairs')
        pairs['type'] = [x if type(x)==str else 'n/a' for x in pairs.type]        # replace missing types with 'n/a'
        return pairs
    
    def pairs_to_channels(self):
        channels = pd.DataFrame({'name': np.array(self.pairs.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.pairs.type]
        channels['units'] = 'V'                                                    # convert EEG to V
        channels['low_cutoff'] = 'n/a'                                             # highpass filter (don't actually know this for clinical eeg)
        channels['high_cutoff'] = 'n/a'                                            # lowpass filter (mne adds Nyquist frequency = 2 x sampling rate)
        channels['reference'] = 'bipolar'
        channels['group'] = np.array(self.pairs.grpName)
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.pairs.type]
        channels['notch'] = 'n/a'
        channels = channels.fillna('n/a')                                          # remove NaN
        channels = channels.replace('', 'n/a')                                     # no empty cells
        return channels
    
    def contacts_to_channels(self):
        channels = pd.DataFrame({'name': np.array(self.contacts.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.contacts.type]
        channels['units'] = 'V'                                                    # convert EEG to V
        channels['low_cutoff'] = 'n/a'                                             # highpass filter (don't actually know this for clinical eeg)
        channels['high_cutoff'] = 'n/a'                                            # lowpass filter (mne adds Nyquist frequency = 2 x sampling rate)
        channels['group'] = np.array(self.contacts.grpName)
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]
        channels['notch'] = 'n/a'
        channels = channels.fillna('n/a')                                          # remove NaN
        channels = channels.replace('', 'n/a')                                     # no empty cells
        return channels
    
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
    
    # ---------------------------------
    # run conversion
    def run(self):
        # ---------- Events ----------
        self.reader = self.cml_reader()
        self.wordpool_file = self.set_wordpool()
        self.events_descriptor = self.make_events_descriptor()
        self.write_BIDS_beh()

        # terminate if no monopolar or bipolar EEG
        if not self.monopolar and not self.bipolar:
            return True
        
        # ---------- EEG ----------
        self.sfreq, self.recording_duration = self.eeg_metadata()

        # ---------- Electrodes ----------
        self.contacts = self.load_contacts()

        # both Talairach and MNI coordinates (default to Tal)
        if self.tal and self.mni:
            self.electrodes = self.contacts_to_electrodes('tal', True)
            self.electrodes_sidecar = self.make_electrodes_sidecar('tal', True)
            self.write_BIDS_electrodes('tal', True)
            self.write_BIDS_coords('tal')
        elif self.tal:
            self.electrodes_tal = self.contacts_to_electrodes('tal', False)
            self.electrodes_sidecar = self.make_electrodes_sidecar('tal', False)
            self.write_BIDS_electrodes('tal', False)
            self.write_BIDS_coords('tal')
        elif self.mni:
            self.electrodes_mni = self.contacts_to_electrodes('mni', False)
            self.electrodes_sidecar = self.make_electrodes_sidecar('mni', False)
            self.write_BIDS_electrodes('mni', False)
            self.write_BIDS_coords('mni')

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
            self.write_BIDS_channels('bipolar')
        if self.monopolar:
            self.eeg_sidecar_mono = self.eeg_sidecar('monopolar')
            self.eeg_mono = self.eeg_mono_to_BIDS()
            self.write_BIDS_ieeg('monopolar')
            self.write_BIDS_channels('monopolar')

        return True
