# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class PAL1_BIDS_converter(intracranial_BIDS_converter):
    wordpool_EN = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/PAL1/wordpools/wordpool_EN.txt', dtype=str)
    wordpool_SP = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/PAL1/wordpools/wordpool_SP.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/PAL1/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        if self.subject=='R1082N' or self.subject=='R1238N':
            wordpool_file = 'wordpools/wordpool_SP.txt'
        else:
            wordpool_file = 'wordpools/wordpool_EN.txt'
        
        return wordpool_file
    
    def events_to_BIDS(self):
        ### I THINK THIS NEEDS MORE WORK, OR AT LEAST TESTING
        events = self.reader.load('events')
        events = events.rename(columns={'eegoffset': 'sample', 'type': 'trial_type'})           # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                      # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0   # event duration [s] --> lots of superfluous events may mess this up
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)             # replace events with negative duration with 0.0s
        events = self.apply_event_durations(events)                                             # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                         # response time [s]
        events.loc[events.trial_type=='PROB', 'response_time'] = events['rectime'] / 1000.0     # math events use rectime [s]
        events.loc[events.trial_type=='REC_EVENT', 'response_time'] = events['RT'] / 1000.0     # recall events use RT [s]
        events['stim_file'] = np.where((events.trial_type.isin(['STUDY_PAIR', 'PROBE_START', 'TEST_PROBE']))
                                        & (events.list>0), self.wordpool_file, 'n/a')           # add wordpool to word events
        events.loc[events.answer==-999, 'answer'] = 'n/a'                                       # non-math events no answer
        
        events = events.fillna('n/a')                    # change NaN to 'n/a'
        events = events.replace('', 'n/a')               # no empty cells

        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file',
                         'serialpos', 'probepos', 'probe_word', 'resp_word', 'study_1', 'study_2',    # leave probepos and serialpos as is
                         'list', 'test', 'answer', 'experiment', 'session', 'subject']]         # re-order columns

        return events
    
    def apply_event_durations(self, events):
        durations = []
        
        # toggles
        study_orient_toggle = 'STUDY_ORIENT' in events['trial_type'].unique() and 'STUDY_ORIENT_OFF' not in events['trial_type'].unique()
        test_orient_toggle = 'TEST_ORIENT' in events['trial_type'].unique() and 'RETRIEVAL_ORIENT_OFF' not in events['trial_type'].unique()

        for _, row in events.iterrows():
            # fixation events
            # STUDY_ORIENT, TEST_ORIENT = 275 ms if missing offset events  --> DESIGN DOC SAYS 250 MS, UPDATE
            if row.trial_type == 'STUDY_ORIENT' and study_orient_toggle:
                durations.append(0.275)
            elif row.trial_type == 'TEST_ORIENT' and test_orient_toggle:
                durations.append(0.275)

            # countdown events = 10000 ms
            elif row.trial_type == 'COUNTDOWN_START':
                durations.append(10.0)

            # word pair presentation events = 4000 ms
            elif row.trial_type == 'STUDY_PAIR' or row.trial_type == 'PRACTICE_PAIR':
                durations.append(4.0)

            # recall cue events = 4000 ms
            elif row.trial_type == 'TEST_PROBE' or row.trial_type == 'PROBE_START' or row.trial_type == 'PRACTICE_PROBE':
                durations.append(4.0)

            # keep current duration
            else:
                durations.append(row.duration)

        events['duration'] = durations        # preserves column order
        return events

    
    def make_events_descriptor(self):
        descriptions = {
            'SESS_START': 'Beginning of session.',
            'SESS_END': 'End of session.',
            'REC_START': '(Single) probed recall begins.',
            'REC_END': '(Single) probed recall ends.',
            'COUNTDOWN_START': 'Beginning of pre-list presentation countdown.',
            'COUNTDOWN_END': 'End of pre-list presentation countdown.',
            'START': 'Beginning of math distractor phase.',
            'STOP': 'End of math distractor phase.',
            'PROB': 'Math problem presentation onset.',
            'TRIAL': 'Denotes new trial (list).',
            'MIC TEST_START': 'Beginning of microphone test.',
            'WAITING_START': 'Beginning of waiting period.',
            'INSTRUCT_START': 'Beginning of instructions.',
            'SESSION_SKIPPED': 'Denotes a skipped session.',
            'ORIENT_START': 'Beginning of fixation.',
            'MIC_TEST': 'Microphone test.',
            'ORIENT_END': 'End of fixation.',
            'INSTRUCT_END': 'End of instructions.',
            'TRIAL_END': 'End of trial (list).',
            'PRACTICE_ORIENT_OFF': 'End of fixation (in a practice list).',
            'MIC': 'Microphone test.',
            'PRACTICE_ORIENT': 'Fixation (in a practice list).',
            'TRIAL_START': 'Beginning of trial (list).',
            'MIC TEST_END': 'End of microphone test.',
            'PRACTICE_POST_INSTRUCT_START': 'Beginning of practice trial (list).',
            'PRACTICE_TRIAL': 'Denotes practice trial.',
            'ENCODING_END': 'End of word presentation list.',
            'PRACTICE_POST_INSTRUCT_END': 'End of practice trial (list).',
            'ENCODING_START': 'Beginning of word presentation list.',
            'WAITING_END': 'End of waiting period.',
            'MATH_START': 'Beginning of math distractor phase.',
            'MATH_END': 'End of math distractor phase.',
            'PROBE_START': 'Recall probe onset.',
            'PROBE_END': 'Recall probe offset.',
            'REC_EVENT': 'Recalled word, onset of speech (<> denotes vocalization).',
            'RETRIEVAL_START': 'Recall phase begins.',
            'RETRIEVAL_END': 'Recall phase ends.',
            'STUDY_PAIR': 'Word-pair presentation onset.',
            'STUDY_PAIR_OFF': 'End of word-pair presentation.',
            'FEEDBACK_SHOW_ALL_PAIRS': 'Display correct word pairs at end of recall phase.',
            'FORCED_BREAK': 'Break following recall phase.',
            'INSTRUCT_VIDEO_ON': 'Beginning of instructions.',
            'INSTRUCT_VIDEO_OFF': 'End of instructions.',
            'PAIR_OFF': 'End of word-pair presentation.',
            'PRACTICE_PAIR': 'Word-pair presentation onset (in a practice list).',
            'PRACTICE_PAIR_OFF': 'End of word-pair presentation (in a practice list).',
            'PRACTICE_PROBE': 'Recall probe (in a practice list).',
            'PRACTICE_RETRIEVAL_ORIENT': 'Fixation onset prior to probed recall (in a practice list).',
            'PRACTICE_RETRIEVAL_ORIENT_OFF': 'Fixation offset prior to probed recall (in a practice list).',
            'PROBE_OFF': 'Recall probe offset.',
            'RECALL_START': 'Recall phase begins.',
            'RECALL_END': 'Recall phase ends.',
            'RETRIEVAL_ORIENT_OFF': 'Fixation offset prior to probed recall.',
            'STUDY_ORIENT': 'Fixation onset prior to word-pair presentation.',
            'STUDY_ORIENT_OFF': 'Fixation offset prior to word-pair presentation.',
            'STUDY_START': 'Beginning of word-pair presentation list.',
            'TEST_ORIENT': 'Fixation prior to probed recall.',
            'TEST_PROBE': 'Recall probe.',
            'TEST_START': 'Recall phase begins.'
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
            "subject": {"LongName": "Subject ID",
                        "Description": "The string identifier of the subject, e.g. R1001P."},
            'experiment': {'Description': 'The experimental paradigm completed.'},
            "session": {"Description": "The session number."},
            "list": {"LongName": "List Number",
                     "Description": "Experimental list (1-25) during which the event occurred. Trial <= 0 indicates practice list."},
            'serialpos': {'LongName': 'Serial Position', 
                          'Description': 'The order position of a word-pair presented during an encoding phase.'},
            'probepos': {'LongName': 'Probe Position',
                         'Description': 'The order position of a probe presented during a recall phase.'},
            'test': {"LongName": "Math problem", 
                     "Description": "Math problem with form X + Y + Z = ?  Stored in list [X, Y, Z]."},
            'answer': {"LongName": "Math problem response", 
                       "Description": "Participant answer to problem with form X + Y + Z = ?  Note this is not necessarily the correct answer."},
            'probe_word': {'Description': 'Word presented as probe for cued recall.'},
            'resp_word': {'LongName': 'Recall Response',
                          'Description': 'Word recalled in a recall event.'},
            'study_1': {'Description': 'Word presented as part of paired associates.'},
            'study_2': {'Description': 'Word presented as part of paired associates.'}
        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    
    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'cued recall of paired associates')
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = False
        return sidecar
