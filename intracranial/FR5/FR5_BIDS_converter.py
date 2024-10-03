# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class FR5_BIDS_converter(intracranial_BIDS_converter):
    wordpool = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/FR5/wordpools/wordpool.txt', dtype=str)

    # initialzie
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/FR5/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)


    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[(evs['type'] == 'WORD') & (evs['list'] > 0)]    # practice lists diferent wordpool
        if all([x in self.wordpool for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file
    

    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = events[events.mstime != -1]               # drop events with no mstime values
        events = self.unpack_stim_params(events)           # convert stimulation parameters to columns
        events = self.assign_stim_lists(events)            # assign stim_list values for math events

        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type', 'stim_on':'stimulation'})     # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                             # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0          # event duration [s]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                    # replace events with negative duration with 0.0 s
        events = self.apply_event_durations(events)                                                    # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                                # response time [s]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') | 
                (events.trial_type=='PROB'), 'response_time'] = events['rectime'] / 1000.0             # use rectime
        events['stim_file'] = np.where((events.trial_type=='WORD') & (events.list!=-1), self.wordpool_file, 'n/a')    # add wordpool to word events
        events.loc[events.answer==-999, 'answer'] = 'n/a'                                              # non-math events no answer
        events.loc[(events['trial_type'].isin(['START', 'PROB', 'STOP'])) & 
                   (events['recalled'] == -999), 'recalled'] = 0                                       # math events have recalled = -999
        events['item_name'] = events.item_name.replace('X', 'n/a')
        events = events.fillna('n/a')                              # change NaN to 'n/a'
        events = events.replace('', 'n/a')

        # select and re-order columns
        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'item_name',
                         'serialpos', 'recalled', 'list', 'test', 'answer', 'stimulation', 'stim_list', 'stim_duration',
                         'anode_label', 'cathode_label', 'amplitude', 'pulse_freq', 'n_pulses', 'pulse_width', 
                         'experiment', 'session', 'subject']]

        return events
    

    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'free recall with closed-loop stimulation at encoding and retrieval')     # place in second column
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar