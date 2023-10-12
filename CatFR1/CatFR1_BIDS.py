# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids

class CatFR1_bids:
    ELEC_TYPES_DESCRIPTION = {'S': 'strip', 'G': 'grid', 'D': 'depth', 'uD': 'micro'}
    ELEC_TYPES_BIDS = {'S': 'ECOG', 'G': 'ECOG', 'D': 'SEEG', 'uD': 'SEEG'}
    # wordpools

    # initialize
    def __init__(self, subject, experiment, session, system_version, monopolar, mni, tal, area, area_data, root='/scratch/hherrema/BIDS_storage/CatFR1/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.system_version = system_version
        self.monopolar = monopolar
        self.mni = mni
        self.tal = tal
        self.area = area
        self.area_data = area_data
        self.root = root

    # instantiate a reader, save as attribute
    def cml_reader(self):
        df = cml.get_data_index('r1', '/')
        df_sess = df.query("subject==@self.subject & experiment=@self.experiment & session=@self.session").iloc[0]
        self.reader = cml.CMLReader(subject=df_sess.subject, experiment=df_sess.experiment, session=df_sess.session, 
                                    localization=df_sess.localization, montage=df_sess.montage)
    
    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']

    # TO-DO: make CatFR1 specific changes
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                             # onset from first event [ms]
        events['duration'] = np.concatebate((np.diff(events.mstime), np.array([0]))) / 1000.0          # event duration [ms]
        events['response_time'] = 'n/a'                                                                # response time [ms]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') | 
                   (events.trial_type=='PROB'), 'response_time'] = events['duration']           # slightly superfluous
        #events['stim_file'] = np.where((events.trial_type=='WORD') & (events.list!=-1), self.wordpool_file, 'n/a')    # add wordpool to word events
        events.loc[events.answer==-999, 'answer'] = 'n/a'
        events['item_name'] = events.item_name.replace('X', 'n/a')
        events = events.drop(columns=['is_stim', 'stim_list', 'stim_params', 'mstime', 'protocol', 'item_num', 
                                      'iscorrect', 'eegfile', 'exp_version', 'montage', 'msoffset'])
        events = events.drop(columns=['intrusion', 'recalled'])
        #if 'PRACTICE_WORD' in events.trial_type.unique():
        #    events.loc[events.trial_type=='PRACTICE_WORD', 'serialpos'] = events['serialpos' + 1]

        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'item_name',
                         'serialpos', 'list', 'test', 'answer', 'experiment', 'session', 'subject']]
        
        return events
    
    def make_events_descriptor(self):
        descriptions = {

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
        self.events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}

    def write_BIDS_beh(self):
        bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                      suffix='beh', extension='.tsv', datatype='beh', root=self.root)
        os.makedirs(bids_path.directory, exist_ok=True)
        self.events.to_csv(bids_path.fpath, sep='\t', index=False)
        with open(bids_path.update(extenstion='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)

    
    # ---------- Electrodes ----------
    def load_contacts(self):
        self.contacts = self.reader.load('contacts')

    def contacts_to_electrodes(self, atlas):
        electrodes = pd.DataFrame({'name': np.array(self.contacts.label)})
        electrodes['x'] = self.contacts[f'{atlas}.x']
        electrodes['y'] = self.contacts[f'{atlas}.y']
        electrodes['z'] = self.contacts[f'{atlas}.z']
        if self.area:
            electrodes['size'] = [self.area_data.get(re.sub('\d+', '', x)) for x in self.contacts.label]    # add contact area if we have the info
        else:
            electrodes['size'] = -999
        electrodes['group'] = [re.sub('\d+', '', x) for x in self.contacts.label]
        electrodes['hemisphere'] = [x[0] for x in self.contacts.label]
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]

        return electrodes
    
    def write_BIDS_electrodes(self, atlas):
        if atlas == 'tal':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          suffix='electrodes', extension='.tsv', datatype='ieeg', space='Talairach', 
                                          root=self.root)
            os.makedirs(bids_path.directory, exist_ok=True)
            self.electrodes_tal.to_csv(bids_path.fpath, sep='\t', index=False)
        elif atlas == 'mni':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          suffix='electrodes', extenstion='.tsv', datatype='ieeg', space='MNI152NLin6ASym', 
                                          root=self.root)
            os.makedirs(bids_path.directory, exist_ok=True)
            self.electrodes_mni.to_csv(bids_path.fpath, sep='\t', index=False)

    def coordinate_system(self, atlas):
        if atlas == 'tal':
            return {'iEEGCoordinateSystem': 'Talairach', 'iEEGCoordinateUnits': 'mm'}
        elif atlas == 'mni':
            return {'iEEGCoordinateSystem': 'MNI152NLin6ASym', 'iEEGCoordinateUnits': 'mm'}

    def write_BIDS_coords(self, atlas):
        if atlas == 'tal':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          suffix='coordsystem', extension='.json', datatype='ieeg', space='Talairach', 
                                          root=self.root)
            with open(bids_path.fpath, 'w') as f:
                json.dump(fp=f, obj=self.coord_sys_tal)
        elif atlas == 'mni':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          suffix='coordsystem', extension='.json', datatype='ieeg', space='MNI152NLin6ASym', 
                                          root=self.root)
            with open(bids_path.fpath, 'w') as f:
                json.dump(fp=f, obj=self.coord_sys_mni)

    # ---------- Channels ----------
    def load_pairs(self):
        self.pairs = self.reader.load('pairs')

    def pairs_to_channels(self):
        channels = pd.DataFrame({'name': np.array(self.pairs.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.pairs.type_1]     # check that all pairs have same type
        if self.system_version == 1.0:
            channels['units'] = 'arbitrary'
        else:
            channels['units'] = 'uV'
        channels['low_cutoff'] = 'n/a'     # highpass filter
        channels['high_cutoff'] = 'n/a'    # lowpass filter (mne adds Nyquist frequency = 2 x sampling rate)
        channels['reference'] = 'bipolar'
        channels['group'] = [re.sub('\d+', '', x).split('-')[0] for x in self.pairs.label]
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.pairs.type_1]
        channels['notch'] = 'n/a'

        return channels
    
    def write_BIDS_channels(self, ref):
        if ref == 'bipolar':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          acquisition='bipolar', suffix='channels', extension='.tsv', datatype='ieeg', 
                                          root=self.root)
            self.channels_bi.to_csv(bids_path.fpath, sep='\t', index=False)
        elif ref == 'monopolar':
            bids__path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                           acquisition='monopolar', suffix='channels', extenstion='.tsv', datatype='ieeg', 
                                           root=self.root)
            self.channels_mono.to_csv(bids_path.fpath, sep='\t', index=False)

    # ---------- EEG ----------
    def eeg_metadata(self):
        eeg = self.reader.load_eeg()
        self.sfreq = eeg.samplerate
        self.recording_duration = eeg.data.shape[-1] / self.sfreq

    def eeg_sidecar(self, ref):
        sidecar = {'TaskName': self.experiment}
        sidecar['TaskDescription'] = 'delayed free recall of categorized word lists'
        if self.system_version == 2.0 or self.system_version == 4.0:
            sidecar['Manufacturer'] = 'Blackrock'
        elif self.system_version >= 3.0 and self.system_version < 4.0:
            sidecar['Manufacturer'] = 'Medtronic'
        sidecar['SamplingFrequency'] = float(self.sfreq)
        sidecar['PowerLineFrequency'] = 60.0
        sidecar['SoftwareFilters'] = 'n/a'
        sidecar['HardwareFilters'] = 'n/a'
        sidecar['RecordingDuration'] = float(self.recording_duration)
        sidecar['RecordingType'] = 'coninuous'
        sidecar['ElecticalStimulation'] = False
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
        if ref == 'bipolar':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          acquisition='bipolar', suffix='ieeg', extension='.edf', datatype='ieeg', 
                                          root=self.root)
            mne_bids.write_raw_bids(self.eeg_bi, bids_path=bids_path, events=None, allow_preload=True, format='EDF', overwrite=True)
            mne_bids.update_sidecar_json(bids_path.update(extension='.json'), self.eeg_sidecar_bi)
        elif ref == 'monopolar':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          acquisition='monopolar', suffix='ieeg', extension='.edf', datatype='ieeg', 
                                          root=self.root)
            mne_bids.write_raw_bids(self.eeg_mono, bids_path=bids_path, events=None, allow_preload=True, format='EDF', overwrite=True)
            mne_bids.update_sidecar_json(bids_path.update(extension='.json'), self.eeg_sidecar_mono)

        # also write events
        bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                      suffix='events', extension='.tsv', datatype='ieeg', 
                                      root=self.root)
        self.events.to_csv(bids_path.fpath, sep='\t', index=False)
        with open(bids)