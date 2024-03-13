# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
import sys

class FR1_Bids:
    ELEC_TYPES_DESCRIPTION = {'S': 'strip', 'G': 'grid', 'D': 'depth', 'uD': 'micro'}
    ELEC_TYPES_BIDS = {'S': 'ECOG', 'G': 'ECOG', 'D': 'SEEG', 'uD': 'SEEG'}
    wordpool_EN = np.loadtxt('wordpools/wordpool_EN.txt', dtype=str)
    wordpool_ES = np.loadtxt('wordpools/wordpool_ES.txt', dtype=str)
    wordpool_short_EN = np.loadtxt('wordpools/wordpool_short_EN.txt', dtype=str)
    wordpool_long_EN = np.loadtxt('wordpools/wordpool_long_EN.txt', dtype=str)
    wordpool_long_ES = np.loadtxt('wordpools/wordpool_long_ES.txt', dtype=str)
    
    # initialize
    #def __init__(self, subject, experiment, session, system_version, monopolar, mni, tal, area, area_data, root='/scratch/hherrema/BIDS_storage/FR1/'):
    def __init__(self, subject, experiment, session, system_version, monopolar, bipolar, mni, tal, root='/scratch/hherrema/BIDS_storage/FR1/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.system_version = system_version
        self.monopolar = monopolar
        self.bipolar = bipolar
        self.mni = mni
        self.tal = tal
        #self.area = area
        #self.area_data = area_data
        self.root = root
        
    # instantiate a reader, save as attribute
    def cml_reader(self):
        df = cml.get_data_index('r1', '/')
        df_sess = df.query("subject==@self.subject & experiment==@self.experiment & session==@self.session").iloc[0]
        self.reader = cml.CMLReader(subject=df_sess.subject, experiment = df_sess.experiment, session=df_sess.session, 
                                    localization=df_sess.localization, montage=df_sess.montage)
        
    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']; word_evs = word_evs[word_evs['list']!=-1]    # remove practice list
        if np.all([1 if x in self.wordpool_EN else 0 for x in word_evs.item_name]):
                self.wordpool_file = 'wordpools/wordpool_EN.txt'
        elif np.all([1 if x in self.wordpool_short_EN else 0 for x in word_evs.item_name]):
            self.wordpool_file = 'wordpools/wordpool_short_EN.txt'
        elif np.all([1 if x in self.wordpool_long_EN else 0 for x in word_evs.item_name]):
            self.wordpool_file = 'wordpools/wordpool_long_EN.txt'
        elif np.all([1 if x in self.wordpool_ES else 0 for x in word_evs.item_name]):
            self.wordpool_file = 'wordpools/wordpool_ES.txt'
        elif np.all([1 if x in self.wordpool_long_ES else 0 for x in word_evs.item_name]):
            self.wordpool_file = 'wordpools/wordpool_long_ES.txt'
        else:
            self.wordpool_file = 'n/a'
    
    def events_to_BIDS(self):                   # can load events for all 589 FR1 sessions
        events = self.reader.load('events')
        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})     # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0        # onset from first event [ms]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0   # event duration [ms]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)             # replace events with negative duration with 0.0 s
        events['response_time'] = 'n/a'                                                            # response time [ms]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') | 
                   (events.trial_type=='PROB'), 'response_time'] = events['duration']           # slightly superfluous
        events['stim_file'] = np.where((events.trial_type=='WORD') & (events.list!=-1), self.wordpool_file, 'n/a')    # add wordpool to word events
        events.loc[events.answer==-999, 'answer'] = 'n/a'                                       # non-math events no answer
        events['item_name'] = events.item_name.replace('X', 'n/a')                              
        events = events.drop(columns=['is_stim', 'stim_list', 'stim_params', 'mstime', 'protocol', 'item_num', 
                                      'iscorrect', 'eegfile', 'exp_version', 'montage', 'msoffset'])   # drop unneeded fields
        events = events.drop(columns=['intrusion', 'recalled'])                    # dropping because confusing
        if 'PRACTICE_WORD' in events.trial_type.unique():                                # practice words wrongly given serial positions 0-11
            events.loc[events.trial_type=='PRACTICE_WORD', 'serialpos'] = events['serialpos'] + 1
        events = events.fillna('n/a')                                             # change NaN to 'n/a'
        events = events.replace('', 'n/a')                                        # try to resolve empty cell issue
        
        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'item_name', 
                        'serialpos', 'list', 'test', 'answer', 'experiment', 'session', 'subject']]       # re-order columns
        
        return events
            
    def make_events_descriptor(self):
        descriptions = {
            "SESS_START": "Beginning of session.",
            "SESS_END": "End of session.",
            "WORD": "Word presentation onset.",
            "REC_START": "Recall phase begins.",
            "REC_END": "Recall phase ends.",
            "REC_WORD": "Recalled word, onset of speech (during free recall).",
            "REC_WORD_VV": "Vocalization (during free recall).",
            'COUNTDOWN_START': 'Beginning of pre-list presentation countdown.',
            'COUNTDOWN_END': 'End of pre-list presentation countdown.', 
            'DISTRACT_START': 'Beginning of math distractor phase.', 
            'DISTRACT_END': 'End of math distractor phase.',
            "START": "Beginning of math distractor phase.",
            "STOP": "End of math distractor phase.",
            "PROB": "Math problem presentation onset.",
            "PRACTICE_WORD": "Word presentation onset (in a practice list).",
            'PRACTICE_REC_START': "Recall phase begins (in a practice list).", 
            'PRACTICE_REC_END': "Recall phase ends (in a practice list).",
            'PRACTICE_DISTRACT_START': 'Beginning of math distractor phase (in a practice list).',
            'PRACTICE_DISTRACT_END': 'End of math distractor phase (in a practice list).',
            'TRIAL': 'Denotes new trial (list).',         # denotes new trial (list)
            'ORIENT': 'Fixation preceding presentation of word list.',  # before presentation or recall?
            'MIC TEST_START': 'Beginning of microphone test.',
            'WAITING_START': 'Beginning of waiting period.',       #??
            'INSTRUCT_START': 'Beginning of instructions.',
            'SESSION_SKIPPED': 'Denotes a skipped session.',       #??
            'RETRIEVAL_ORIENT': 'Fixation pror to recall phase.',
            'ORIENT_START': 'Beginning of fixation.',
            'MIC_TEST': 'Microphone test.',
            'WORD_OFF': 'End of word presentation.',      # Word presentation offset?
            'ORIENT_OFF': 'End of fixation.',
            'ORIENT_END': 'End of fixation.',
            'INSTRUCT_END': 'End of instructions.',
            'TRIAL_END': 'End of trial (list).',
            'Logging': 'Denotes logging.',
            'PRACTICE_ORIENT_OFF': 'End of fixation (in a practice list).',
            'INSTRUCT_VIDEO': 'Instructions video.',
            'PRACTICE_WORD_OFF': 'End of word presentation (in a practice list).',
            'MIC': 'Microphone test.',
            'PRACTICE_ORIENT': 'Fixation (in a practice list).',
            'TRIAL_START': 'Beginning of trial (list).',
            'MIC TEST_END': 'End of microphone test.',
            'RETRIEVAL_ORIENT_START': 'Beginning of fixation preceding recall phase.',
            'PRACTICE_POST_INSTRUCT_START': 'Beginning of practice trial (list).',
            'PRACTICE_TRIAL': 'Denotes practice trial.',
            'ENCODING_END': 'End of word presentation list.',
            'PRACTICE_POST_INSTRUCT_END': 'End of practice trial (list).',
            'ENCODING_START': 'Beginning of word presentation list.',
            'RETRIEVAL_ORIENT_END': 'End of fixation preceding recall phase.',
            'WAITING_END': 'End of waiting period.'
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
        
    def write_BIDS_beh(self):         # test this locally first
        bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                      suffix='beh', extension='.tsv', datatype='beh', root=self.root)
        os.makedirs(bids_path.directory, exist_ok=True)                                 # make beh directory
        self.events.to_csv(bids_path.fpath, sep='\t', index=False)
        with open(bids_path.update(suffix='beh', extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)


    # ---------- Electrodes ----------
    
    def load_contacts(self):
        self.contacts = self.reader.load('contacts')
        
    def contacts_to_electrodes(self, atlas):
        electrodes = pd.DataFrame({'name': np.array(self.contacts.label)})    # name = label of contact
        electrodes['x'] = self.contacts[f'{atlas}.x']
        electrodes['y'] = self.contacts[f'{atlas}.y']
        electrodes['z'] = self.contacts[f'{atlas}.z']
        #if self.area:
        #    electrodes['size'] = [self.area_data.get(re.sub('\d+', '', x)) for x in contacts.label]  # add contact area if we have the info
        #else:
        #    electrodes['size'] = -999
        electrodes['size'] = -999
        electrodes['group'] = [re.sub('\d+', '', x) for x in self.contacts.label]
        electrodes['hemisphere'] = ['L' if x < 0 else 'R' if x > 0 else 'n/a' for x in electrodes.x]   # use coordinates for hemisphere
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]     # not exactly what is meant by type field
        electrodes = electrodes.fillna('n/a')                         # remove NaN
        electrodes = electrodes.replace('', 'n/a')                    # resolve empty cell issue
        
        return electrodes
    
    def write_BIDS_electrodes(self, atlas):
        if atlas == 'tal':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          suffix='electrodes', extension='.tsv', datatype='ieeg', space='Talairach', 
                                          root=self.root)
            os.makedirs(bids_path.directory, exist_ok=True)                                 # make ieeg directory
            self.electrodes_tal.to_csv(bids_path.fpath, sep='\t', index=False)
        elif atlas == 'mni':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          suffix='electrodes', extension='.tsv', datatype='ieeg', space='MNI152NLin6ASym', 
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
        
    def pairs_to_channels(self):              # sessions with bipolar eeg (all sessions)
        channels = pd.DataFrame({'name': np.array(self.pairs.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.pairs.type_1]      # all pairs have same type
        if self.system_version == 1.0:
            channels['units'] = 'arbitrary'        # don't have system 1 units
        else:
            channels['units'] = 'V'
        channels['low_cutoff'] = 'n/a'        # highpass filter
        channels['high_cutoff'] = 'n/a'       # lowpass filter (mne adds Nyquist frequency = 2 x sampling rate)
        channels['reference'] = 'bipolar'
        channels['group'] = [re.sub('\d+', '', x).split('-')[0] for x in self.pairs.label]
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.pairs.type_1]    # all pairs have same type
        channels['notch'] = 'n/a'
        
        return channels
    
    def contacts_to_channels(self):            # sessions with monopolar eeg
        channels = pd.DataFrame({'name': np.array(self.contacts.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.contacts.type]
        if self.system_version == 1.0:
            channels['units'] = 'arbitrary'        # dont' have system 1 units
        else:
            channels['units'] = 'V'
        channels['low_cutoff'] = 'n/a'
        channels['high_cutoff'] = 'n/a'
        # channels['reference'] = 'intracranial' # 'mastoid'         # should explain that the data should be rereferenced 
        channels['group'] = [re.sub('\d+', '', x) for x in self.contacts.label]
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]    # longer description of electrode type
        channels['notch'] = 'n/a'
        
        return channels
    
    def write_BIDS_channels(self, ref):
        if ref == 'bipolar':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          acquisition='bipolar', suffix='channels', extension='.tsv', datatype='ieeg', 
                                          root=self.root)
            self.channels_bi.to_csv(bids_path.fpath, sep='\t', index=False)
        elif ref == 'monopolar':
            bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                          acquisition='monopolar', suffix='channels', extension='.tsv', datatype='ieeg', 
                                          root=self.root)
            self.channels_mono.to_csv(bids_path.fpath, sep='\t', index=False)
    
    
    # -------- EEG ----------
    def eeg_metadata(self):              # set sample rate, system version
        eeg = self.reader.load_eeg()
        self.sfreq = eeg.samplerate
        self.recording_duration = eeg.data.shape[-1] / self.sfreq
        
    def eeg_sidecar(self, ref):
        sidecar = {'TaskName': self.experiment}
        sidecar['TaskDescription'] = 'delayed free recall of word lists'
        if self.system_version == 2.0:
            sidecar['Manufacturer'] = 'Blackrock'
        elif self.system_version >= 3.0 and self.system_version < 4.0:
            sidecar['Manufacturer'] = 'Medtronic'
        sidecar['SamplingFrequency'] = float(self.sfreq)
        sidecar['PowerLineFrequency'] = 60.0
        sidecar['SoftwareFilters'] = 'n/a'
        sidecar['HarwareFilters'] = 'n/a'
        sidecar['RecordingDuration'] = float(self.recording_duration)      # length of recording (s)
        sidecar['RecordingType'] = 'continuous'
        sidecar['ElectricalStimulation'] = False
        if ref == 'bipolar':
            sidecar['iEEGReference'] = 'bipolar'               # provide non-rereferenced monopolar EEG too
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
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)
          
    
    # ---------- EEG (monopolar) ----------
    def eeg_mono_to_BIDS(self):
        eeg = self.reader.load_eeg()           # convert to V before instantiating raw object
        if self.system_version == 2.0:
            eeg.data = eeg.data / 4000000.0    # convert from 250 nV to V
        elif self.system_version >= 3.0 and self.system_version < 4.0:
            eeg.data = eeg.data / 10000000.0    # convert from 0.1 uV to V
        eeg_mne = eeg.to_mne()
        #eeg_mne.metadata = self.events         # mne-bids automatically adds events annotations when reading (from events.tsv)
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_mono.type]))
        eeg_mne.set_channel_types(mapping)     # set channel types
        
        return eeg_mne
        
    # ---------- EEG (bipolar) ----------
    def eeg_bi_to_BIDS(self):
        eeg = self.reader.load_eeg(scheme=self.pairs)   # convert to V before instantiating raw object
        if self.system_version == 2.0:
            #eeg.data = eeg.data / 4.0          # convert from 250 nV to uV
            eeg.data = eeg.data / 4000000.0    # convert from 250 nV to V
        elif self.system_version >= 3.0 and self.system_version < 4.0:
            #eeg.data = eeg.data / 10           # convert from 0.1 uV to uV
            eeg.data = eeg.data / 10000000.0    # convert from 0.1 uV to V
        eeg_mne = eeg.to_mne()
        #eeg_mne.metadata = self.events         # mne-bids automatically adds events annotation when reading (from events.tsv)       
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_bi.type]))
        eeg_mne.set_channel_types(mapping)      # set channel types
            
        return eeg_mne
    
    # run conversion
    def run(self):
        # ---------- Events ----------
        self.cml_reader()                                          # set self.reader
        self.set_wordpool()                                        # set self.wordpool_file
        self.events = self.events_to_BIDS()                        # convert events to BIDS format
        self.make_events_descriptor()                              # set self.events_descriptor
        self.write_BIDS_beh()                                      # write events to BIDS format
        
        # ---------- EEG ----------
        self.eeg_metadata()                                        # set self.sfreq
        
        # ---------- Electrodes ----------
        self.load_contacts()                                       # set self.contacts
        if self.mni:
            self.electrodes_mni = self.contacts_to_electrodes('mni')   # convert contacts to BIDS format
            self.write_BIDS_electrodes('mni')                          # write electrodes to BIDS format
            self.coord_sys_mni = self.coordinate_system('mni')
            self.write_BIDS_coords('mni')
        if self.tal:
            self.electrodes_tal = self.contacts_to_electrodes('tal')
            self.write_BIDS_electrodes('tal')
            self.coord_sys_tal = self.coordinate_system('tal')
            self.write_BIDS_coords('tal')
        
        # ---------- Channels ----------
        self.load_pairs()                                          # set self.pairs
        if self.bipolar:
            self.channels_bi = self.pairs_to_channels()                # convert pairs to BIDS format
        if self.monopolar:
            self.channels_mono = self.contacts_to_channels()       # convert contacts to BIDS format
            
        # ---------- EEG ----------
        if self.bipolar:
            self.eeg_sidecar_bi = self.eeg_sidecar('bipolar')
            self.eeg_bi = self.eeg_bi_to_BIDS()
            self.write_BIDS_ieeg('bipolar')                            # write bipolar iEEG to BIDS format
            self.write_BIDS_channels('bipolar')                        # write channels to BIDS format (overwrite automatic)
        if self.monopolar:
            self.eeg_bi = None                                     # save memory space
            self.eeg_sidecar_mono = self.eeg_sidecar('monopolar')
            self.eeg_mono = self.eeg_mono_to_BIDS()
            self.write_BIDS_ieeg('monopolar')                      # write monopolar iEEG to BIDS format
            self.write_BIDS_channels('monopolar')