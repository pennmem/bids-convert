# imports
import cmlreaders as cml
import numpy as np
import pandas as pd
import mne
import os
import h5py
import json
from glob import glob
import mne_bids
import shutil
import time

# class for converting FR1 intracranial data to BIDS format
class FR1BIDSConverter:
    # initialize
    def __init__(self, subject, experiment, session, wls_dict, root='/home1/hherrema/programming_data/BIDS_convert', 
                 overwrite_eeg=True, overwrite_beh=True):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.root = root
        self.wls_dict = wls_dict
        self.overwrite_beh = overwrite_beh
        self.overwrite_eeg = overwrite_eeg
        
    # ---------------- methods -------------------
    # return raw eeg file --> currently find 576 session eeg files, still missing 13 sessions
    def locate_raw_file(self):
        if self.subject in self.wls_dict:
            self.session = int(self.wls_dict.get(self.subject).get(self.session))
        raw_file = glob(f'/data/eeg/{self.subject}/raw/{self.experiment}_{self.session}/*.edf*') + \
            glob(f'/data/eeg/{self.subject}/raw/{self.experiment}_{self.session}/*.EDF*') + \
            glob(f'/data/eeg/{self.subject}/raw/{self.experiment}_{self.session}/*.EEG*') + \
            glob(f'/data/eeg/{self.subject}/raw/{self.experiment}_{self.session}/*.21E*') + \
            glob(f'/data/eeg/{self.subject}/behavioral/{self.experiment}/session_{self.session}/host_pc/20*/*.h5*') + \
            glob(f'/data/eeg/{self.subject}/raw/{self.experiment}_{self.session}/20*/*.ns2*') + \
            glob(f'/data/eeg/{self.subject}/raw/{self.experiment}_{self.session}/1*/*.ns2*') + \
            glob(f'/data/eeg/{self.subject}/raw/{self.experiment}_{self.session}/clinical_eeg/*.edf*')
        if len(raw_file) == 0:
            # check for other subject directory
            other_dir = glob(f'/data/eeg/{self.subject}/behavioral/NOTE_THIS_SUBJECT_ALSO_IN_{self.subject}_*')
            if len(other_dir) > 0:
                for i in range(len(other_dir)):
                    suffix = other_dir[i][-1]
                    raw_file += glob(f'/data/eeg/{self.subject}_{suffix}/raw/{self.experiment}_{self.session}/*.edf*') + \
                        glob(f'/data/eeg/{self.subject}_{suffix}/raw/{self.experiment}_{self.session}/*.EEG*') + \
                        glob(f'/data/eeg/{self.subject}_{suffix}/behavioral/{self.experiment}/session_{self.session}/host_pc/20*/*.h5*') + \
                        glob(f'/data/eeg/{self.subject}_{suffix}/raw/{self.experiment}_{self.session}/20*/*.ns2*')
                if len(raw_file) > 0:
                    print(f'sub:{self.subject}, exp:{self.experiment}, sess:{self.session}: {raw_file}')
                    return raw_file
                else:
                    print(f'sub:{self.subject}, exp:{self.experiment}, sess:{self.session}, no files')
                    raise FileNotFoundError

            # check if wrongly labeled session --> session 0 labled as FR1_1 (need to think about this, recursion unending)
            else:
                print(f'sub:{self.subject}, exp:{self.experiment}, sess:{self.session}, no files')
                raise FileNotFoundError
        elif len(raw_file) > 1:
            print(f'sub:{self.subject}, exp:{self.experiment}, sess:{self.session}, multiple files', raw_file)
            return raw_file
        else:
            print(f'sub:{self.subject}, exp:{self.experiment}, sess:{self.session}: {raw_file}')
            return raw_file[0]
    
    """
    # taken from eeg.py in bids_creation, don't no where to find config file
    def read_odin_config(self):
        with open(self.config['RAM.odin_config'], 'r') as f:
            odin_config = json.load(f)
        return odin_config
    """
    
    # load intracranial eeg
    # stores sfreq and recording_start attributes
    def load_ieeg(self):
        sources_file = open(f'/protocols/r1/subjects/{self.subject}/experiments/{self.experiment}/{self.session}/ephys/current_processed/sources.json')
        metadata = json.load(sources_file)
        self.n_samples = metadata[list(metadata.keys())[0]]['n_samples']            # taking index 0 won't work for multiple eeg files
        
        if self.file_type in ['.edf', '.EDF']:
            raw = mne.io.read_raw_edf(self.raw_filepath, preload=False)
            self.sfreq = raw.info['sfreq']
            self.recording_start = raw.info['meas_date']
        elif self.file_type in ['.EEG', '.21E']:              
            raw = mne.io.read_raw_nihon(self.raw_filepath, preload=False)            # should be working with pull request
            self.sfreq = raw.info['sfreq']
            self.recording_start = raw.info['meas_date']
        elif self.file_type == '.h5':
            # get sample_rate from sources.json, samplerate filed in hdf5 file is null --> removes need for odin config file
            self.sfreq = metadata[list(metadata.keys())[0]]['sample_rate']
            self.recording_start = metadata[list(metadata.keys())[0]]['start_time_str']        # gives day, month, year, hour, minute as string (not datetime.datetime object)
            with h5py.File(self.raw_filepath, 'r') as f:
                eeg_data = f['timeseries'][:]
                ch_names = f['names'].asstr()[:]
                #self.recording_start = f['start_ms']                                 # not the same type as raw.info['meas_date'], but same info
                
            eeg_info = mne.create_info(list(ch_names), self.sfreq, ch_types='eeg')       # doesn't accept array of channel names
            raw = mne.io.RawArray(eeg_data.T, eeg_info)                                  # transpose for accepted shape
        elif self.file_type == '.ns2':
            # https://github.com/NeuralEnsemble/python-neo/blob/master/neo/rawio/blackrockrawio.py#L790  --> might be a help for reading .ns2 files
            raise NotImplementedError
        else:
            raise ValueError('Unknown File Extension:', self.file_type)
            
        return raw
    
    # loads/formats information for electrodes.tsv --> names and coordinates
    def load_electrodes(self):
        contacts_path = self.df_sess.iloc[0]['contacts']
        contacts_file = open(contacts_path)
        contacts_jso = json.load(contacts_file)
        contacts = contacts_jso[self.subject]['contacts']
        mni_electrodes = pd.DataFrame(); tal_electrodes = pd.DataFrame()                  # dataframes for mni and tal coordinate systems
        for k in contacts.keys():
            mni = contacts[k]['atlases']['mni']; tal = contacts[k]['atlases']['tal']
            mni_electrodes = mni_electrodes.append({'name':k, 'x':mni['x'], 'y':mni['y'], 'z':mni['z'], 'size':1}, ignore_index=True)
            tal_electrodes = tal_electrodes.append({'name':k, 'x':tal['x'], 'y':tal['y'], 'z':tal['z'], 'size':1}, ignore_index=True)
            
        return mni_electrodes, tal_electrodes
    
    # loads/formats information for channels.tsv
    def load_channels(self):
        # load from cml readers
        reader = cml.CMLReader(subject=self.df_sess.iloc[0]['subject'], experiment=self.df_sess.iloc[0]['experiment'], session=self.df_sess.iloc[0]['session'],
                               localization=self.df_sess.iloc[0]['localization'], montage=self.df_sess.iloc[0]['montage'])
        pairs = reader.load('pairs')                      # all data bipolar referenced according to clinical team
        channels = pd.DataFrame()         # need to think about how to do this --> "channels should appear in the same order as they do in the iEEG datafile"
        
    # store the session row from dataframe for easy field access
    def set_df_sess(self):
        df = cml.get_data_index('r1', '/')
        df_select = df[(df['experiment']==self.experiment) & (df['subject']==self.subject)]
        self.df_sess = df_select.query('session == @self.session')
    
    # set wordpool file attribute --> eventually shouldn't reference absolute filepath
    def set_wordpool(self):
        self.wordpool_file = '/home1/hherrema/programming_data/BIDS_convert/bids_convert/FR1_wordpool.txt'
        
    # set system version --> could be useful to store system version --> must be 
    def set_system_version(self):
        self.system_version = self.df_sess.iloc[0]['system_version']
    
    # need to get all information --> where from?
    def set_sidecar(self):
        self.eeg_sidecar = {'TaskName': 'FR1'}
        #self.eeg_sidecar['InstitutionName'] = 
        #self.eeg_sidecar['InstitutionAddress'] =
        #self.eeg_sidecar['InstitutionalDepartmentName'] = 
        if self.system_version in [3.1, 3.2, 3.3, 3.4]:
            self.eeg_sidecar['Manufacturer'] = 'Medtronic'                         # presumably system dependent
        elif self.system_version == 2.0:
            self.eeg_sidecar['Manufacturer'] = 'Blackrock'
        elif self.system_version == 1.0:
            self.eeg_sidecar['Manufacturer'] = 'unknown-FILL_IN'
        else:
            self.eeg_sidecar['Manufacturer'] = 'unknown-FILL_IN'
        #self.eeg_sidecar['ManufacturersModelName'] = 
        #self.eeg_sidecar['SoftwareVersions'] = 
        self.eeg_sidecar['TaskDescription'] = 'delayed free recall of word lists'
        #self.eeg_sidecar['Instructions'] = 
        #self.eeg_sidecar['CogAtlasID'] = 
        #self.eeg_sidecar['CogPOID'] = 
        #self.eeg_sidecar['DeviceSerialNumber'] = 
        self.eeg_sidecar['iEEGReference'] = 'bipolar'               # always bipolar reference according to clincal team
        self.eeg_sidecar['SamplingFrequency'] = self.sfreq
        self.eeg_sidecar['PowerLineFrequency'] = 60.0
        self.eeg_sidecar['SoftwareFilters'] = 'n/a'
        self.eeg_sidecar['HardwareFilters'] = 'n/a'
        self.eeg_sidecar['ElectrodeManufacturer'] = 'enter'                # only if all electrodes same manufacturer
        #self.eeg_sidecar['ElectrodeManufacturersModelName']
        jacksheet = np.loadtxt(f'/data/eeg/{self.subject}/docs/jacksheet.txt', dtype=str)      # won't work for suffixed subjects (i.e. R1093J_1)
        self.eeg_sidecar['ECOGChannelCount'] = jacksheet.shape[0]                    # from jacksheet.txt
        self.eeg_sidecar['SEEGChannelCount'] = 0
        self.eeg_sidecar['EEGChannelCount'] = 0                     
        self.eeg_sidecar['EOGChannelCount'] = 0                     
        self.eeg_sidecar['ECGChannelCount'] = 0
        self.eeg_sidecar['EMGChannelCount'] = 0
        self.eeg_sidecar['MiscChannelCount'] = 0
        self.eeg_sidecar['TriggerChannelCount'] = 0
        self.eeg_sidecar['RecordingDuration'] = self.n_samples / self.sfreq
        self.eeg_sidecar['RecordingType'] = 'continuous'
        #self.eeg_sidecar['EpochLength'] = 
        #self.eeg_sidecar['iEEGGround'] = 
        #self.eeg_sidecar['iEEGPlacementScheme'] = 
        #self.eeg_sidecar['iEEGElectrodeGroups'] = 
        #self.eeg_sidecar['SubjectArtefactDescription'] = 
        self.eeg_sidecar['ElectricalStimulation'] = False
        #self.eeg_sidecar['ElectricalStimulationParameters'] = 
        
    
    # load behavioral events, put in correct structure for BIDS --> do I need beh_only argument if no eeg for subject?
    def load_events(self):
        reader = cml.CMLReader(self.subject, self.experiment, self.session)
        events = reader.load('events')
        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})
        events['onset'] = (events['mstime'] - events['mstime'].iloc[0]) / 1000
        events['duration'] = np.concatenate((np.diff(events['mstime']), np.array([0]))) / 1000
        events['response_time'] = 'n/a'
        events.loc[(events['trial_type']=='REC_WORD') | (events['trial_type']=='REC_WORD_VV') | 
                   (events['trial_type']=='PROB'), 'response_time'] = events['duration']
        events.loc[(events['trial_type']=='REC_WORD') | (evens['trial_type']=='REC_WORD_VV') | 
                   (events['trial_type']=='PROB'), 'duration'] = 'n/a'
        events['stim_file'] = np.where(events.trial_type == 'WORD', self.wordpool_file, 'n/a')
        events[['test_x', 'test_y', 'test_z']] = events['test'].apply(pd.Series)
        events['answer'] = events['test_x'] + events['test_y'] + events['test_z']
        events.loc[(events['answer']==0) | (events['answer']==-2997), 'answer'] = 'n/a'
        events.loc[(events['trial_type']!='WORD') & (events['trial_type']!='REC_WORD') & 
                   (events['recalled']==0), 'recalled'] = 'n/a'
        events['item_name'] = events['item_name'].replace('X', 'n/a')
        events = events.drop(columns=['is_stim', 'stim_list', 'stim_params', 'test', 'mstime', 'protocol', 
                                      'iscorrect', 'eegfile', 'exp_version', 'montage', 'msoffset'])
        events = events.replace(-999, 'n/a')
        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'subject', 
                           'experiment', 'session', 'list', 'item_name', 'item_num', 'serialpos', 'recalled', 'intrusion', 
                           'answer', 'test_x', 'test_y', 'test_z']]
        return events
    
    # write behavioral events to BIDS format
    # taken from Joey, work to undestand each step
    def write_bids_beh(self, overwrite=True):
        bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), datatye='beh', 
                                      suffix='beh', extension='.tsv', root=self.root)
        os.makedirs(bids_path.directory, exist_ok=True)
        self.events.to_csv(bids_path,fpath, sep='\t', index=False)
        with open(bids_path.update(suffix='beh', extension='.json').fpath, 'w') as f:
            json.dump(fp=f, obj=self.events_descriptor)
            
    # write eeg to BIDS format
    def write_bids_eeg(self, emp_path='temp.edf', overwrite=True):
        bids_path = mne_bids.BIDSPath(subject=self.subject, session=str(self.session), task=self.experiment, 
                                      datatype='eeg', root=self.root)
        # understand before filling in rest
            
    def make_event_descriptors(self):
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
            "PRACTICE_WORD": "Word presentation onset (in a practice list)",
            'PRACTICE_REC_START': "Recall phase begins (in a practice list).", 
            'PRACTICE_REC_END': "Recall phase ends (in a practice list).",
            'PRACTICE_DISTRACT_START': 'Beginning of math distractor phase (in a practice list).',
            'PRACTICE_DISTRACT_END': 'End of math distractor phase (in a practice list).',
            'TRIAL': 'Beginning of new word presentation list.',
            'ORIENT': 'Fixation preceding presentation of word list.'
        }
        HED = {
            "onset": {
                "Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file. ",
            },
            "subject": {
                "LongName": "Subject ID",
                "Description": "The string identifier of the subject, e.g. R1001P.",

            },
            'experiment': {
                'Description': 'The experimental paradigm completed.',
            },
            "session": {
                "Description": "The session number."
            },
            "list": {
                "LongName": "List Number",
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
            'serialpos': {
                'LongName': 'Serial Position',
                'Description': 'The order position of a word presented in an WORD event.  1-12 for experimental lists and 0-11 for practice lists.'
            },
            'recalled': {
                'Description': 'For WORD events, whether a presented word is recalled during the following recall period (1 = recalled, 0 = not recalled).  For REC_WORD events, \
                whether a recall is from the preceding list (1 = correct recall, 0 = intrusion).'
            },
            'intrusion': {
                'Description': 'Classification of recalled word.  0 = Correct recall, -1 = Extra-list intrusion, > 0 = Prior-list intusion \
                (value indicates number of lists ago word was presented).'
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
            }
        }
        self.events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}

            
    def run(self):
        self.set_df_sess()
        self.set_wordpool()
        self.events = self.load_events()
        self.make_event_descriptors()
        self.set_system_version()
        self.write_bids_beh(overwrite=self.overwrite_beh)
        self.write_filepath = self.locate_raw_file()
        self.file_type = os.path.splitext(self.raw_filepath)[1]
        self.raw_file = self.load_ieeg()
        self.set_sidecar()
        #self.set_montage()
        self.write_bids_eeg(temp_path=f'/home1/hherrema/.temp/{int(time.time()*100)}_temp.edf', overwrite=self.overwrite_eeg)
            
    # instantiate and run ---------------
    """
    if __name__ = '__main__':
        converter = FR1BIDSConverter(subject, experiment, session)
        converter.run()
    """
