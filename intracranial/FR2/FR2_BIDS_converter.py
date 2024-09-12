# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class FR2_BIDS_converter(intracranial_BIDS_converter):

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/FR2'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']

        raise NotImplementedError
    
    def events_to_BIDS(self):
        events = self.reader.load('events')

        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})                    # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                               # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0            # event duration [s]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                      # replace events with negative duration with 0.0 s
        events = self.apply_event_durations(events)                                                      # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                                  # response time [s]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') | 
                (events.trial_type=='PROB'), 'response_time'] = events['rectime'] / 1000.0            # use rectime
        events['stim_file'] = np.where((events.trial_type=='WORD') & (events.list!=-1), self.wordpool_file, 'n/a')    # add wordpool to word events

        raise NotImplementedError
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            pass

        events['duration'] = durations               # preserves column order
        return events
    
    def make_events_descriptor(self):
        descriptions = {
            "SESS_START": "Beginning of session.",
            "SESS_END": "End of session.",
            "SESSION_SKIPPED": "",
            "START": "",
            "STOP": "",
            "TRIAL": "",
            "ORIENT": "",
            "COUNTDOWN_START": "Beginning of pre-list presentation countdown.",
            "COUNTDOWN_END": "End of pre-list presentation countdown.",
            "WORD": "Word presentation onset.",
            "PRACTICE_WORD": "Word presentation onset (on a practice list).",
            "DISTRACT_START": "Beginning of math distractor phase.",
            "DISTRACT_END": "End of math distractor phase.",
            "PRACTICE_DISTRACT_START": "Beginning of math distractor phase (on a practice list).",
            "PRACTICE_DISTRACT_END": "End of math distractor phase (on a practice list).",
            "PROB": "Math problem presentation onset.",
            "REC_START": "Start of recall phase.",
            "REC_END": "End of recall phase.",
            "PRACTICE_REC_START": "Start of recall phase (on a practice list).",
            "PRACTICE_REC_END": "End of recall phase (on a practice list).",
            "REC_WORD": "Recalled word, onset of speech (during free recall).",
            "REC_WORD_VV": "Vocalization (during free recall).",
            "STIM_ON": ""
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
                     "Description": "Word list (1-20) during which the event occurred. Trial = 0 indicates practice list."},
            "item_name": {"Description": "The word being presented or recalled in a WORD or REC_WORD event."},
            'serialpos': {'LongName': 'Serial Position', 
                          'Description': 'The order position of a word presented in an WORD event.'},
            'test': {"LongName": "Math problem", 
                     "Description": "Math problem with form X + Y + Z = ?  Stored in list [X, Y, Z]."},
            'answer': {"LongName": "Math problem response", 
                       "Description": "Participant answer to problem with form X + Y + Z = ?  Note this is not necessarily the correct answer."}
        }


    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', '')              # place in second column
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar
