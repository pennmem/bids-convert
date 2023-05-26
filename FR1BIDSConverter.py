# imports
import cmlreaders as cml
import numpy as np
import pandas as pd
import mne
import os
import json
from glob import glob
import mne_bids
import shutil
import time

# class for converting FR1 intracranial data to BIDS format
class FR1BIDSConverter:
    # initialize
    def __init__(self, subject, experiment, session, root='/home1/hherrema/programming_data/BIDS_convert', 
                overwrite_eeg=True, overwrite_beh=True):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.root = root
        self.overwrite_beh = overwrite_beh
        self.overwrite_eeg = overwrite_eeg
        
    # ---------------- methods -------------------
    # return raw eeg file --> currently find 416 session eeg files, still missing 173 session --> know where to find 137/173
    def locate_raw_file(sub, exp, sess):
        raw_file = glob(f'/data/eeg/{sub}/raw/{exp}_{sess}/*.edf*') + \
            glob(f'/data/eeg/{sub}/raw/{exp}_{sess}/*.EEG*') + \
            glob(f'/data/eeg/{sub}/raw/{exp}_{sess}/*.EDF*') + \
            glob(f'/data/eeg/{sub}/behavioral/{exp}/session_{sess}/host_pc/20*/*.h5*')
        if len(raw_file) == 0:
            # check for other subject directory
            other_dir = glob(f'/data/eeg/{sub}/behavioral/NOTE_THIS_SUBJECT_ALSO_IN_{sub}_*')
            if len(other_dir) > 0:
                for i in range(len(other_dir)):
                    suffix = other_dir[i][-1]
                    raw_file += glob(f'/data/eeg/{sub}_{suffix}/raw/{exp}_{sess}/*.edf*') + \
                        glob(f'/data/eeg/{sub}_{suffix}/raw/{exp}_{sess}/*.EEG*')
                return raw_file[0]                                   # what to do with multiple files???
            else:
                print(f'sub:{sub}, exp:{exp}, sess:{sess}, no files')
                raise FileNotFoundError
        elif len(raw_file) > 1:
            print('Multiple files', raw_file)
            raise ValueError
        else:
            print(f'sub:{sub}, exp:{exp}, sess:{sess}: {raw_file}')
            return raw_file[0]
        
    # load intracranial eeg
    # have lots of other filetypes I'm struggling to understand --> blackrock .nev, .ns2, .ccf, etc.
    def load_ieeg(self):
        if self.file_type in ['.edf', '.EDF']:
            raw = mne.io.read_raw_edf(self.raw_filepath, stim_channels='Status', preload=False)
        elif: self.file_type == '.EEG':              
            raw = mne.io.read_raw_nihon(self.raw_filepath, preload=False)            # docs say .EEG should be nihon but doesn't work
        else:
            raise ValueError('Unknown File Extension:', self.file_type)
        self.sfreq = raw.info['sfreq']
        self.recording_start = raw.info['meas_date']
    
    # set wordpool file attribute --> eventually shouldn't reference absolute filepath
    def set_wordpool(self):
        self.wordpool_file = '/home1/hherrema/programming_data/BIDS_convert/bids_convert/FR1_wordpool.txt'
        
    # set system version --> could be useful to store system version
    def set_system_version(self):
        self.system_version = 1
        
    # need to get all information --> where from?
    def set_montage(self):
        self.eeg_sidecar = {'PowerLineFrequency':60.0}
        
    
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
        self.set_wordpool()
        self.events = self.load_events()
        self.make_event_descriptors()
        self.write_bids_beh(overwrite=self.overwrite_beh)
        self.write_filepath = self.locate_raw_file()
        self.file_type = os.path.splitext(self.raw_filepath)[1]
        self.raw_file = self.load_ieeg()
        self.set_montage()
        self.write_bids_eeg(temp_path=f'/home1/hherrema/.temp/{int(time.time()*100)}_temp.edf', overwrite=self.overwrite_eeg)
            
    # instantiate and run ---------------
    """
    if __name__ = '__main__':
        converter = FR1BIDSConverter(subject, experiment, session)
        converter.run()
    """