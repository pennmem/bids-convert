# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class PAL2_BIDS_converter(intracranial_BIDS_converter):
    wordpool_EN = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/PAL2/wordpools/wordpool_EN.txt', dtype=str)
    wordpool_SP = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/PAL2/wordpools/wordpool_SP.txt', dtype=str)
    

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/FR2'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type'] == 'STUDY_PAIR']
        if all([x in self.wordpool_EN for x in word_evs.study_1]) and all([x in self.wordpool_EN for x in word_evs.study_2]):
            wordpool_file = 'wordpools/wordpool_EN.txt'
        elif all([x in self.wordpool_SP for x in word_evs.study_1]) and all([x in self.wordpool_EN for x in word_evs.study_2]):
            wordpool_file = 'wordpools/wordpool_SP.txt'
        else:
            wordpool_file = 'n/a'
        
        return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = self.unpack_stim_params(events)                        # convert stimulation parameters into columns
        events['n_pulses'] = events['n_pulses'].replace(250, 230)       # event creation wrongly assings n_pulses = 250
        events = self.assign_stim_lists(events)                         # assign stim_list values for math events

        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type', 'stim_on':'stimulation'})   # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                      # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0   # event duration [s] --> lots of superfluous events may mess this up
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)             # replace events with negative duration with 0.0s
        events = self.apply_event_durations(events)                                             # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                         # response time [s]
        events.loc[events.trial_type=='PROB', 'response_time'] = events['rectime'] / 1000.0     # math events use rectime [s]
        events.loc[events.trial_type=='REC_EVENT', 'response_time'] = events['RT'] / 1000.0     # recall events use RT [s]
        events['stim_file'] = np.where((events.trial_type.isin(['STUDY_PAIR', 'TEST_PROBE']))
                                        & (events.list>0), self.wordpool_file, 'n/a')           # add wordpool to word events
        events.loc[events.answer==-999, 'answer'] = 'n/a'                                       # non-math events no answer
        events.loc[events.stimulation == 0, ['anode_label', 'cathode_label']] = ''              # set stim parameters to defaults if no stimulation
        events.loc[events.stimulation == 0, ['stim_duration', 'amplitude', 'pulse_freq', 'n_pulses', 'pulse_width']] = 0

        events = events.fillna('n/a')                  # change NaN to 'n/a'
        events = events.replace('', 'n/a')             # no empty cells

        # select and re-order columns
        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'study_1', 'study_2',
                         'serialpos', 'probepos', 'probe_word', 'resp_word', 'correct', 'list', 'test', 'answer', 'stimulation', 'stim_list',
                         'stim_duration', 'anode_label', 'cathode_label', 'amplitude', 'pulse_freq', 'n_pulses', 'pulse_width', 
                         'experiment', 'session', 'subject']]
        
        return events
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # fixation events = 250 ms
            if row.trial_type == 'STUDY_ORIENT' or row.trial_type == 'TEST_ORIENT':
                durations.append(0.250)

            # no countdown events = 10000 ms

            # word pair presentation events = 4000 ms
            elif row.trial_type == 'STUDY_PAIR':
                durations.append(4.0)

            # recall cue events = 4000 ms
            elif row.trial_type == 'TEST_PROBE':
                durations.append(4.0)

            # stimulation events
            elif row.trial_type == 'STIM_ON':
                durations.append(4.6)

            # keep current duration
            else:
                durations.append(row.duration)

        events['duration'] = durations
        return events
    
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
            "SESS_START": "Beginning of session.",
            "SESSION_SKIPPED": "Denotes a skipped session.",
            "ENCODING_START": "Beginning of word-pair presentation list.",
            "STUDY_ORIENT": "Fixation onset prior to word-pair presentation.",
            "STUDY_PAIR": "Word-pair presentation onset.",
            "START": "Beginning of math distractor phase.",
            "MATH_START": "Beginning of math distractor phase.",
            "STOP": "End of math distractor phase.",
            "MATH_END": "End of math distractor phase.",
            "PROB": "Math problem presentation onset.",                        # rectime gives answer time
            "TEST_START": "Recall phase begins.",
            "TEST_ORIENT": "Fixation prior to probed recall.",
            "TEST_PROBE": "Recall probe.",
            "REC_START": "(Single) probed recall begins.",
            "REC_END": "(Single) probed recall ends.",
            "REC_EVENT": "Recalled word, onset of speech (<> denotes vocalization).",
            "STIM_ON": "Onset of electrical stimulation.",
            "FEEDBACK_SHOW_ALL_PAIRS": "Showing all word pairs after recall phase."
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category", 
                           "Description": "Indicator of type of task action that occurs at the marked time", 
                           "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()}},
            "response_time": {"Description": "Time (in seconds) between onset of probed recall phase and recall (for recalls and vocalizations), or between onset of problem on screen and response (for math problems)."},
            "stim_file": {"LongName": "Stimulus File", 
                          "Description": "Location of wordpool file containing words presented in word-pair encoding events."},
            'study_1': {'Description': 'Word presented as part of paired associates.'},
            'study_2': {'Description': 'Word presented as part of paired associates.'},
            'serialpos': {'LongName': 'Serial Position', 
                          'Description': 'The order position of a word-pair presented during an encoding phase.'},
            'probepos': {'LongName': 'Probe Position',
                         'Description': 'The order position of a probe presented during a recall phase.'},
            'probe_word': {'Description': 'Word presented as probe for cued recall.'},
            'resp_word': {'LongName': 'Recall Response',
                          'Description': 'Word recalled in a recall event.'},     
            "correct": {"Description": "For STUDY_PAIR events, denotes if presented word-pair is recalled.  For REC_EVENT events, denotes if recall is correct."},        
            "list": {"LongName": "List Number",
                     "Description": "Experimental list (1-25) during which the event occurred. Trial <= 0 indicates practice list."},
            'test': {"LongName": "Math problem", 
                     "Description": "Math problem with form X + Y + Z = ?  Stored in list [X, Y, Z]."},
            'answer': {"LongName": "Math problem response", 
                       "Description": "Participant answer to problem with form X + Y + Z = ?  Note this is not necessarily the correct answer."},
            "stimulation": {"Description": "Denotes if event occurs during electrical stimulation.  1 indicates stimulation on."},
            "stim_list": {"LongName": "Stimulation List",
                          "Description": "Denotes lists with electrical stimulation during encoding or retrieval."},
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
