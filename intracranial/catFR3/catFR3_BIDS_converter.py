# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class FR3_BIDS_converter(intracranial_BIDS_converter):
    wordpool = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/catFR3/wordpools/wordpool.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/catFR3/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    
    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type'] == 'WORD']    # practice lists all labeled PRACTICE_WORD
        if all([x in self.wordpool for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = events[events.mstime != -1]               # drop events with no mstime values
        events = self.unpack_stim_params(events)           # convert stimulation parameters to columns
        events['stim_duration'] = events['stim_duration'].replace([480, 490], 500)     # fix incorrect stim durations
        events = self.assign_stim_lists(events)            # assign stim_list values for math events

        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type', 'stim_on':'stimulation'})   # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                          # onset from first events [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0       # event duration [s]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                 # replace negative duration with 0.0 s
        events = self.apply_event_durations(events)                                                 # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                             # response time [s]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') |
                   (events.trial_type=='PROB'), 'response_time'] = events['rectime'] / 1000.0       # use rectime
        events['stim_file'] = np.where(events.trial_type=='WORD', self.wordpool_file, 'n/a')        # add wordpool to word events
        events.loc[events.answer==-999, 'answer'] = 'n/a'                                           # non-math events no answer
        events.loc[(events['trial_type'].isin(['START', 'PROB', 'STOP'])) & 
                   (events['recalled'] == -999), 'recalled'] = 0                                    # math events have recalled = -999
        events['item_name'] = events.item_name.replace('X', 'n/a')
        events['category'] = events.category.replace('X', 'n/a')
        events.loc[events.trial_type=='PRACTICE_WORD', 'serialpos'] = range(1, len(events.query("trial_type=='PRACTICE_WORD'")) + 1)   # practice words given serial position = -1
        events = events.fillna('n/a')                                                               # change NaN to 'n/a'
        events = events.replace('', 'n/a')

        # select and re-order columns
        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'item_name', 'category',
                         'serialpos', 'recalled', 'list', 'test', 'answer', 'stimulation', 'stim_list', 'stim_duration', 
                         'anode_label', 'cathode_label', 'amplitude', 'pulse_freq', 'n_pulses', 'pulse_width', 
                         'experiment', 'session', 'subject']]
        
        return events
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # fixation events = 1600 ms
            if row.trial_type == 'ORIENT' or row.trial_type == 'PRACTICE_ORIENT':
                durations.append(1.6)
            
            # or 500 ms
            elif row.trial_type == 'RETRIEVAL_ORIENT':
                durations.append(0.5)

            # countdown events = 10000 ms
            elif row.trial_type == 'COUNTDOWN_START':
                durations.append(10.0)

            # word presentation events = 1600 ms
            elif row.trial_type == 'WORD' or row.trial_type == 'PRACTICE_WORD':
                durations.append(1.6)

            # stimulation events = 500 ms
            elif row.trial_type == 'STIM_ON':
                durations.append(0.5)

            # keep current duration
            else:
                durations.append(row.duration)

        events['duration'] = durations
        return durations
    
    # unpack stimulation parameters from dictionary and add as columns to events dataframe
    def unpack_stim_params(self, events):
        stim_params_df = pd.DataFrame()
        for _, row in events.iterrows():
            stim_params_df = pd.concat([stim_params_df, pd.DataFrame.from_dict([row.stim_params])], ignore_index=True)

        return pd.concat([events, stim_params_df], axis=1)
    
    # assign stim_list values to math events with default -999
    def assign_stim_lists(self, events):
        stim_list = []
        for _, row in events.iterrows():
            if row['stim_list'] == -999:
                stim_list.append(max(events[events['list'] == row['list']].stim_list))
            else:
                stim_list.append(row.stim_list)
        
        events['stim_list'] = stim_list
        return events
    
    def make_events_descriptor(self):
        descriptions = {
            'SESS_START': "Start of session.",
            'SESS_END': "End of session.",
            'MIC_TEST': "Microphone test.", 
            'INSTRUCT_START': "Start of instructions.",
            'INSTRUCT_END': "End of instructions.",
            'PRACTICE_TRIAL': "Denotes start of practice trial (list).",
            'PRACTICE_ORIENT': "Start of fixation (on a practice list).",
            'PRACTICE_ORIENT_OFF': "End of fixation (on a practice list).",
            'PRACTICE_WORD': "Word presentation onset (on a practice list).", 
            'PRACTICE_WORD_OFF': "Word presentation offset (on a practice list).",
            'PRACTICE_REC_START': "Start of recall phase (on a practice list).",
            'PRACTICE_REC_END': "End of recall phase (on a practice list).",
            'TRIAL': "Denotes new trial (list).",
            'COUNTDOWN_START': "Start of countdown preceding encoding phase.",
            'COUNTDOWN_END': "End of countdown preceding encoding phase.",
            'ORIENT': "Fixation preceding encoding phase.", 
            'ORIENT_OFF': "Offset of fixation preceding encoding phase.",
            'WORD': "Word presentation onset.", 
            'WORD_OFF': "Word presentation offset.",
            'ENCODING_END': "End of encoding phase.",
            'DISTRACT_START': "Start of math distractor phase.",
            'DISTRACT_END': "End of math distractor phase.",
            'START': "Start of math distractor phase.",
            'STOP': "End of math distractor phase.",
            'PROB': "Math problem presentation onset.",
            'RETRIEVAL_ORIENT': "Fixation preceding recall phase.",
            'REC_START': "Start of recall phase.",
            'REC_END': "End of recall phase.",
            'REC_WORD': "Recalled word, onset of speech (during free recall).", 
            'REC_WORD_VV': "Vocalization (during free recall).",
            'STIM_ON': "Onset of electrical stimulation.",
            'STIM_OFF': "Offset of electrical stimulation.",
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category", 
                        "Description": "Indicator of type of task action that occurs at the marked time.", 
                        "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()}},
            "response_time": {"Description": "Time (in seconds) between onset of recall phase and recall (for recalls and vocalizations), or between onset of problem on screen and response (for math problems)."},
            "stim_file": {"LongName": "Stimulus File", 
                          "Description": "Location of wordpool file containing words presented in WORD events."},
            "item_name": {"Description": "The word being presented or recalled in a WORD or REC_WORD event."},
            "category": {"Description": "Semantic category of word presented or recalled in a WORD or REC_WORD event."},
            'serialpos': {'LongName': 'Serial Position', 
                          'Description': 'The order position (at encoding) of a word presented in an WORD event or a recall in a REC_WORD event.'},
            "recalled": {"Description": "For WORD events, denotes if presented word is recalled.  For REC_WORD events, denotes if recall is correct."},
            "list": {"LongName": "List Number",
                     "Description": "Word list (1-25) during which the event occurred. Trial = -1 indicates practice list."},
            'test': {"LongName": "Math problem", 
                     "Description": "Math problem with form X + Y + Z = ?  Stored in list [X, Y, Z]."},
            'answer': {"LongName": "Math problem response", 
                       "Description": "Participant answer to problem with form X + Y + Z = ?  Note this is not necessarily the correct answer."},
            "stimulation": {"Description": "Denotes if event occurs during electrical stimulation.  1 indicates stimulation on."},
            "stim_list": {"LongName": "Stimulation List",
                          "Description": "Denotes lists with electrical stimulation during encoding."},
            "stim_duration": {"LongName": "Stimulation Duration",
                              "Description": "Duration (in milliseconds) of electrical stimulation."},
            "anode_label": {"Description": "Anode electrode label."},
            "cathode_label": {"Description": "Cathode electrode label."},
            "amplitude": {"Description": "Amplitude (in milliamperes) of electrical stimulation."},
            "pulse_freq": {"LongName": "Pulse Frequency",
                           "Description": "Frequency (in hertz) of pulses in stimulation event."},
            "n_pulses": {"LongName": "Number of Pulses",
                         "Description": "Number of pulses in stimulation event."},
            "pulse_width": {"Description": "Phase width (in microseconds) of stimulation pulse.  Each pulse contains two phases."},
            'experiment': {'Description': 'The experimental paradigm completed.'},
            "session": {"Description": "The session number."},
            "subject": {"LongName": "Subject ID",
                        "Description": "The string identifier of the subject, e.g. R1001P."},
        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    

    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'categorized free recall with closed-loop stimulation at encoding')              # place in second column
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar