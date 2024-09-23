# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
import scipy.stats
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class YC1_BIDS_converter(intracranial_BIDS_converter):

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/YC2/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        

        raise NotImplementedError
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = self.expand_travel_paths(events)                     # expand events to have rows for travel paths

        events = events.rename(columns={'eegoffset': 'sample', 'type': 'trial_type', 'block_num': 'trial', 'resp_travel_time': 'resp_total_time'})     # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                      # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0   # event duration [s] --> lots of superfluous events may mess this up
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)             # replace events with negative duration with 0.0s
        events = self.apply_event_durations(events)                                             # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                         # response time [s]
        events.loc[(events.trial_type=='NAV_TEST') | (events.trial_type=='NAV_PRACTICE_TEST'),
                    'response_time'] = events['resp_total_time']
        events['stim_file'] = np.where(events.trial_type.isin(['NAV_PRACTICE_LEARN', 'NAV_LEARN']), 
                                       self.wordpool_file, 'n/a')                              # add wordpool to learning events
        events = events.fillna('n/a')                                                          # change NaN to 'n/a'
        events = events.replace('', 'n/a')                                                     # no empty cells

        raise NotImplementedError
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # learning events = 5000 ms (1s turn, 3s drive, 1s pause)
            if row.trial_type == 'NAV_LEARN' or row.trial_type == 'NAV_PRACTICE_LEARN':
                durations.append(5.0)

            # stimulation
            elif row.trial_type
