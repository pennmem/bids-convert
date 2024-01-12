# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class FR1_BIDS_converter(intracranial_BIDS_converter):
    wordpool_EN = np.loadtxt('wordpools/wordpool_EN.txt', dtype=str)
    wordpool_SP = np.loadtxt('wordpools/wordpool_SP.txt', dtype=str)
    wordpool_short_EN = np.loadtxt('wordpools/wordpool_short_EN.txt', dtype=str)
    wordpool_long_EN = np.loadtxt('wordpools/wordpool_long_EN.txt', dtype=str)
    wordpool_long_SP = np.loadtxt('wordpools/wordpool_long_SP.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, root='/scratch/hherrema/BIDS_storage/FR1/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, root)

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']; word_evs[word_evs['list']!=-1]    # remover practice list
        if np.all([1 if x in self.wordpool_EN else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_EN.txt'
        elif np.all([1 if x in self.wordpool_short_EN else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_short_EN.txt'
        elif np.all([1 if x in self.wordpool_long_EN.txt else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_long_EN.txt'
        elif np.all([1 if x in self.wordpool_SP else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_SP.txt'
        elif np.all([1 if x in self.wordpool_long_SP else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_long_SP.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file
    
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
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    
    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar.insert(1, 'TaskDescription', 'delayed free recall of word lists')    # place in second column
        return sidecar