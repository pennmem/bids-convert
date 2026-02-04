# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class PS2_BIDS_converter(intracranial_BIDS_converter):
    wordpool_EN = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/PAL2/wordpools/wordpool_EN.txt', dtype=str)
    wordpool_SP = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/PAL2/wordpools/wordpool_SP.txt', dtype=str)
    
    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/PAL2/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        return 'n/a'
        # evs = self.reader.load('events')
        # word_evs = evs[evs['type'] == 'STUDY_PAIR']
        # if all([x in self.wordpool_EN for x in word_evs.study_1]) and all([x in self.wordpool_EN for x in word_evs.study_2]):
        #     wordpool_file = 'wordpools/wordpool_EN.txt'
        # elif all([x in self.wordpool_SP for x in word_evs.study_1]) and all([x in self.wordpool_SP for x in word_evs.study_2]):
        #     wordpool_file = 'wordpools/wordpool_SP.txt'
        # else:
        #     wordpool_file = 'n/a'
        
        # return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = self.unpack_stim_params(events)                        # convert stimulation parameters into columns
        # events = self.assign_stim_lists(events)                         # assign stim_list values for math events

        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type', 'stim_on':'stimulation'})   # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                      # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0   # event duration [s] --> lots of superfluous events may mess this up
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)             # replace events with negative duration with 0.0s
        events = self.apply_event_durations(events)                                             # apply well-defined durations [s]
        # events['response_time'] = 'n/a'                                                         # response time [s]
        # events.loc[events.trial_type=='PROB', 'response_time'] = events['rectime'] / 1000.0     # math events use rectime [s]
        # events.loc[events.trial_type=='REC_EVENT', 'response_time'] = events['RT'] / 1000.0     # recall events use RT [s]
        # events['stim_file'] = np.where((events.trial_type.isin(['STUDY_PAIR', 'TEST_PROBE']))
                                        # & (events.list>0), self.wordpool_file, 'n/a')           # add wordpool to word events
        # events.loc[events.answer==-999, 'answer'] = 'n/a'                                       # non-math events no answer
        # events.loc[events.stimulation == 0, ['anode_label', 'cathode_label']] = ''              # set stim parameters to defaults if no stimulation
        # events.loc[events.stimulation == 0, ['stim_duration', 'amplitude', 'pulse_freq', 'n_pulses', 'pulse_width']] = 0

        events = events.fillna('n/a')                  # change NaN to 'n/a'
        events = events.replace('', 'n/a')             # no empty cells

        # select and re-order columns
        events = events[['eegoffset', 'ad_observed', 'eegfile', 'exp_version', 'experiment', 'is_stim', 'montage', 'msoffset', 'mstime', 'protocol', 'session', 'stim_params', 'subject', 'type']]
        
        return events
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # # fixation events = 250 ms
            # if row.trial_type == 'STUDY_ORIENT' or row.trial_type == 'TEST_ORIENT':
            #     durations.append(0.250)

            # # no countdown events = 10000 ms

            # # word pair presentation events = 4000 ms
            # elif row.trial_type == 'STUDY_PAIR':
            #     durations.append(4.0)

            # # recall cue events = 4000 ms
            # elif row.trial_type == 'TEST_PROBE':
            #     durations.append(4.0)

            # stimulation events
            if row.trial_type == 'STIM_ON':
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
    # def assign_stim_lists(self, events):
    #     stim_list = []
    #     for _, row in events.iterrows():
    #         if row['stim_list'] == -999:
    #             stim_list.append(max(events[events['list'] == row['list']].stim_list))
    #         else:
    #             stim_list.append(row.stim_list)
        
    #     events['stim_list'] = stim_list
    #     return events
    
    def make_events_descriptor(self):
        descriptions = {
            "STIM_ON": "Onset of electrical stimulation.",
            "STIM_OFF": "Offset of electrical stimulation.",
            "BEGIN_PS2": "Beginning of PS2 experiment.",
            "STIM_SINGLE_PULSE": "Single pulse electrical stimulation event.",
            "AD_CHECK": "Attention check event during encoding phase.",
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category", 
                           "Description": "Indicator of type of task action that occurs at the marked time", 
                           "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()}},
            'experiment': {'Description': 'The experimental paradigm completed.'},
            "session": {"Description": "The session number."},
            "subject": {"LongName": "Subject ID",
                        "Description": "The string identifier of the subject, e.g. R1001P."},
            "ad_observed": {"Description": "Indicator of whether the subject responded correctly to the attention check."},
            "eegfile": {"Description": "The name of the raw EEG data file associated with this event."},
            "exp_version": {"Description": "The version of the experimental paradigm software."},
            "is_stim": {"Description": "Indicator of whether electrical stimulation was applied during this event."},
            "montage": {"Description": "The name of the electrode montage used for recording."},
            "protocol": {"Description": "The name of the experimental protocol used."},
            "stim_params": {"Description": "A dictionary containing the stimulation parameters for this event."},
        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    
    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'Stimulation experiment: subjects are asked to sit quietly while stimulation parameters are varied. For PS2.1, 4 amplitudes are used (up to a maximum amplitude set by the experimenter) and SHAM, 10Hz, 50Hz, 100 Hz, and 200Hz stimulation. Duration can be set to a fixed duration (500 ms default).')
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar
