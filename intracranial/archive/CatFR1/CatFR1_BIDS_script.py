# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import pickle
import mne_bids
import sys

class CatFR1_Bids:
    ELEC_TYPES_DESCRIPTION = {'S': 'strip', 'G': 'grid', 'D': 'depth', 'uD': 'micro'}
    ELEC_TYPES_BIDS = {'S': 'ECOG', 'G': 'ECOG', 'D': 'SEEG', 'uD': 'SEEG'}
    wordpool_categorized_EN = np.loadtxt('wordpools/wordpool_categorized_EN.txt', dtype=str)
    wordpool_categorized_SP = np.loadtxt('wordpools/wordpool_categorized_SP.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, mni, tal, monopolar, bipolar, area, root='/scratch/hherrema/BIDS_storage/CatFR1/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.system_version = system_version
        self.monopolar = monopolar
        self.bipolar = bipolar
        self.mni = mni
        self.tal = tal
        self.area = area
        #self.area_data = area_data
        self.root = root

    # instantiate a reader, save as attribute
    def cml_reader(self):
        df = cml.get_data_index('r1', '/')
        df_sess = df.query("subject==@self.subject & experiment==@self.experiment & session==@self.session").iloc[0]
        self.reader = cml.CMLReader(subject=df_sess.subject, experiment=df_sess.experiment, session=df_sess.session, 
                                    localization=df_sess.localization, montage=df_sess.montage)
    
    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']
        if np.all([1 if x in self.wordpool_categorized_EN else 0 for x in word_evs[word_evs.list != -1].item_name]):
            self.wordpool_file = 'wordpools/wordpool_categorized_EN.txt'
        elif np.all([1 if x in self.wordpool_categorized_SP else 0 for x in word_evs[word_evs.list != -1].item_name]):
            self.wordpool_file = 'wordpools/wordpool_categorized_SP.txt'
        elif self.subject == 'R1039M' or self.subject == 'R1094T':          # edge cases, put in spanish
            self.wordpool_file = 'wordpools/wordpool_categorized_SP.txt'
        else:
            self.wordpool_file = 'n/a'

    # TO-DO: make CatFR1 specific changes
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = events.rename(columns={'eegoffset':'sample', 'type': 'trial_type'})               # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                         # onset from first event [ms]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0      # event duration [ms]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                # replace events with negative duration with 0.0 s
        events['response_time'] = 'n/a'                                                            # response time [ms]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') | 
                  (events.trial_type=='PROB'), 'response_time'] = events['duration']               # practice trial no record of recalls
        events['stim_file'] = np.where((events.trial_type=='WORD') & (events.list!=-1), self.wordpool_file, 'n/a')     # add wordpool to word events
        events.loc[events.answer==-999, 'answer'] = 'n/a'                                          # non-math events no answer
        events['item_name'] = events.item_name.replace('X', 'n/a')
        events['category'] = events.category.replace('X', 'n/a')
        events = events.drop(columns=['is_stim', 'stim_list', 'stim_params', 'mstime', 'protocol', 'item_num', 'iscorrect', 'eegfile', 'exp_version', 
                                      'montage', 'msoffset', 'category_num'])                      # drop unneeded fields
        events = events.drop(columns=['intrusion', 'recalled'])
        events = events.fillna('n/a')                                                              # change NaN to 'n/a'
        events = events.replace('', 'n/a')                                                         # no empty cells

        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'item_name', 'category', 
                        'serialpos', 'list', 'test', 'answer', 'experiment', 'session', 'subject']]     # re-order columns
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
            'category': {'Description': 'Semantic cateogry of word presented or recalled in a WORD or REC_WORD event.'},
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
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)

    
    # ---------- Electrodes ----------
    def load_contacts(self):
        self.contacts = self.reader.load('contacts')
        
    def generate_area_map(self):
        area_path = f'/data10/RAM/subjects/{self.subject}/docs/area.txt'
        area = np.loadtxt(area_path, dtype=str)
        self.area_map = dict(zip(area[:, 0], area[:, 1].astype(float)))

    def contacts_to_electrodes(self, atlas):
        electrodes = pd.DataFrame({'name': np.array(self.contacts.label)})         # name = label of contact
        electrodes['x'] = self.contacts[f'{atlas}.x']
        electrodes['y'] = self.contacts[f'{atlas}.y']
        electrodes['z'] = self.contacts[f'{atlas}.z']
        electrodes['group'] = [re.sub('\d+', '', x) for x in self.contacts.label]
        if self.area:                                                             # use area data if available
            electrodes['size'] = [self.area_map.get(x) if x in self.area_map.keys() else -999 for x in electrodes.group]
        else:
            electrodes['size'] = -999
        electrodes['hemisphere'] = ['L' if x < 0 else 'R' if x > 0 else 'n/a' for x in electrodes.x]     # use coordinates for hemisphere
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]
        electrodes = electrodes.fillna('n/a')                                 # remove NaN
        electrodes = electrodes.replace('', 'n/a')                            # resolve empty cell issue
        electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type']]          # re-order columns

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

    def pairs_to_channels(self):              # sessions with bipolar eeg
        channels = pd.DataFrame({'name': np.array(self.pairs.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x) for x in self.pairs.type_1]      # all pairs have same type
        channels['units'] = 'V'
        channels['low_cutoff'] = 'n/a'        # highpass filter
        channels['high_cutoff'] = 'n/a'       # lowpass filter (mne adds Nyquist frequency = 2 x sampling rate)
        channels['reference'] = 'bipolar'
        channels['group'] = [re.sub('\d+', '', x).split('-')[0] for x in self.pairs.label]
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.pairs.type_1]    # all pairs have same type
        channels['notch'] = 'n/a'
        channels = channels.fillna('n/a')              # remove NaN
        channels = channels.replace('', 'n/a')         # resolve empty cell issue

        return channels
    
    def contacts_to_channels(self):           # sessions with monopolar eeg
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
        sidecar['RecordingType'] = 'continuous'
        sidecar['ElectricalStimulation'] = False
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
        with open(bids_path.update(extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)
            
    # ---------- EEG (monopolar) ----------
    def eeg_mono_to_BIDS(self):
        eeg = self.reader.load_eeg()                                                              # convert to V before instantiating raw object
        if self.system_version == 2.0 or self.system_version == 4.0:
            eeg.data = eeg.data / 4000000.0                                                       # convert from 250 nV to V
        elif self.system_version >= 3.0 and self.system_version < 4.0:
            eeg.data = eeg.data / 10000000.0                                                      # convert from 0.1 uV to V
        eeg_mne = eeg.to_mne()
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_mono.type]))       # ecog or seeg
        eeg_mne.set_channel_types(mapping)                                                        # set channel types
        
        return eeg_mne
    
    # ---------- EEG (bipolar) ----------
    def eeg_bi_to_BIDS(self):
        eeg = self.reader.load_eeg(scheme=self.pairs)                                             # convert to V before instantiating raw object
        if self.system_version == 2.0 or self.system_version == 4.0:
            eeg.data = eeg.data / 4000000.0                                                       # convert from 250 nV to V
        elif self.system_version >= 3.0 and self.system_version < 4.0:
            eeg.data = eeg.data / 10000000.0                                                      # convert from 0.1 uV to V
        eeg_mne = eeg.to_mne()
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_bi.type]))         # ecog or seeg
        eeg_mne.set_channel_types(mapping)                                                        # set channel types
        
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
        if self.area:
            self.generate_area_map()                               # set self.area_map
        if self.mni:
            self.electrodes_mni = self.contacts_to_electrodes('mni')    # convert contacts to BIDS format
            self.write_BIDS_electrodes('mni')                           # write electrodes to BIDS format
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
            self.channels_bi = self.pairs_to_channels()            # convert pairs to BIDS format
        if self.monopolar:
            self.channels_mono = self.contacts_to_channels()       # convert contacts to BIDS format
            
        # ---------- EEG ----------
        if self.bipolar:
            self.eeg_sidecar_bi = self.eeg_sidecar('bipolar')
            self.eeg_bi = self.eeg_bi_to_BIDS()
            self.write_BIDS_ieeg('bipolar')                        # write bipolar iEEG to BIDS format
            self.write_BIDS_channels('bipolar')                    # write channels to BIDS format (overwrite automatic)
        if self.monopolar:
            self.eeg_bi = None                                     # save memory space
            self.eeg_sidecar_mono = self.eeg_sidecar('monopolar')
            self.eeg_mono = self.eeg_mono_to_BIDS()
            self.write_BIDS_ieeg('monopolar')                      # write monopolar iEEG to BIDS format
            self.write_BIDS_channels('monopolar')
            
            
# load in metadata
with open('/home1/hherrema/programming_data/BIDS_convert/catFR1/metadata/sub_sys_dict.pkl', 'rb') as f:
    sub_sys_dict = pickle.load(f)
    
with open('/home1/hherrema/programming_data/BIDS_convert/catFR1/metadata/atlas_ref_dict.json', 'r') as f:
    atlas_ref_dict = json.load(f)

with open('/home1/hherrema/programming_data/BIDS_convert/catFR1/metadata/area_dict.json', 'r') as f:
    area_dict = json.load(f)

with open('/home1/hherrema/programming_data/BIDS_convert/catFR1/metadata/pairs_too_long.pkl', 'rb') as f:
    pairs_too_long = pickle.load(f)
    
with open('/home1/hherrema/programming_data/BIDS_convert/catFR1/metadata/groups_num.pkl', 'rb') as f:
    groups_num = pickle.load(f)
    
with open('/home1/hherrema/programming_data/BIDS_convert/catFR1/metadata/good_area.pkl', 'rb') as f:
    good_area = pickle.load(f)
            
# run multiple sessions sequentially by iterating over dataframe
df = cml.get_data_index('r1')
df_select = df[df.experiment=='catFR1']
df_select = df_select[df_select.system_version != 4.0]
for _, row in df_select[int(sys.argv[1]):int(sys.argv[2])].iterrows():
    sub = row.subject
    exp = row.experiment
    sess = row.session
    sysv = sub_sys_dict.get(sub)
    submd = atlas_ref_dict.get(sub); sessmd = submd.get(f'session_{sess}')
    if (sysv == 2.0 or sysv == 3.0) and (sessmd.get('monopolar') and sessmd.get('bipolar')) and sub in good_area and sub not in pairs_too_long and sub not in groups_num:
        print(f'Running {sub, sess}')
        print('----------------------')
        converter = CatFR1_Bids(sub, exp, sess, sysv, sessmd.get('mni'), sessmd.get('tal'), sessmd.get('monopolar'), sessmd.get('bipolar'), True)
        converter.run()
        print('Bids Conversion Complete')
        print('----------------------')
            
print('All Iterated Sessions Complete')
            
#print(f'Running {sys.argv[1], sys.argv[3]}')            
#converter = CatFR1_Bids(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), bool(int(sys.argv[5])), bool(int(sys.argv[6])), bool(int(sys.argv[7])), #bool(int(sys.argv[8])), bool(int(sys.argv[9])))
#converter.run()
#print(f'Bids Conversion Complete')