# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class catFR2_BIDS_converter(intracranial_BIDS_converter):
    wordpool_categorized_EN = np.loadtxt('/home1/hherrema/BIDS/bids_convert/intracranial/catFR2/wordpools/wordpool_categorized_EN.txt', dtype=str)
    wordpool_categorized_SP = np.loadtxt('/home1/hherrema/BIDS/bids_convert/intracranial/catFR2/wordpools/wordpool_categorized_SP.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/catFR2/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type'] == 'WORD']
        if all([x in self.wordpool_categorized_EN for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_categorized_EN.txt'
        elif all([x in self.wordpool_categorized_SP for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_categorized_SP.txt'
        else:
            wordpool_file = 'n/a'
        
        return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = self.unpack_stim_params(events)             # convert stimulation parameters into columns
        events = self.assign_stim_lists(events)              # assign stim_list values for math events

        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type', 'stim_on':'stimulation'})   # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                          # onset from first events [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0       # event duration [s]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                 # replace negative duration with 0.0 s
        events = self.apply_event_durations(events)                                                 # apply well-defined durations [s]
        
        raise NotImplementedError
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            pass

        raise NotImplementedError
    
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

        }
        HED = {

        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columms}
        raise NotImplementedError
    
    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'categorized free recall with open-loop stimulation at encoding')
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar
