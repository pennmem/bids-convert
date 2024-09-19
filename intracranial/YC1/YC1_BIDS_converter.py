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
    wordpool = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/YC1/wordpools/wordpool.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/YC1/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):                 # all sessions same wordpool
        evs = self.reader.load('events')

        if all([x.upper() in self.wordpool for x in evs.stimulus]):
            wordpool_file = 'wordpools/wordpool.txt'
        else:
            wordpool_file = 'n/a'
        
        return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = self.expand_travel_paths(events)                     # expand events to have rows for travel paths

        events = events.rename(columns={'eegoffset': 'sample', 'type': 'trial_type'})     # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                      # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0   # event duration [s] --> lots of superfluous events may mess this up
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)             # replace events with negative duration with 0.0s
        events = self.apply_event_durations(events)                                             # apply well-defined durations [s]


        # select and re-order columns
        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'stimulus', 
                         'trial', 'block', 'paired_block', 'start_loc_x', 'start_loc_y', 'obj_loc_x', 'obj_loc_y', 'resp_loc_x', 'resp_loc_y', 
                         'resp_reaction_time', 'resp_travel_time', 'resp_path_length', 'x', 'y', 'direction', 'travel_time', 
                         'recalled', 'resp_performance_factor', 'session', 'experiment', 'subject']]

        raise NotImplementedError
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # learning events = 5000 ms (1s turn, 3s drive, 1s pause)
            if row.trial_type == 'NAV_LEARN' or row.trial_type == 'NAV_PRACTICE_LEARN':
                durations.append(5.0)

            else:
                durations.append(row.duration)

        events['duration'] = durations
        return events
    
    def apply_data_to_path(row, slope=0.5):
        path = pd.DataFrame(row.path)      # extract path data
        trial_type = row['type']
        
        path = path.rename(columns={'time': 'travel_time'})
        if trial_type == 'NAV_PRACTICE_LEARN':
            tt = 'PATH_PRACTICE_LEARN'
        elif trial_type == 'NAV_PRACTICE_TEST':
            tt = 'PATH_PRACTICE_TEST'
        elif trial_type == 'NAV_LEARN':
            tt = 'PATH_LEARN'
        elif trial_type == 'NAV_TEST':
            tt = 'PATH_TEST'
        else:
            raise ValueError(f'{trial_type} is not a valid trial type.')
        
        
        path['type'] = tt
        path['mstime'] = (row.mstime + path['travel_time']).astype(int)
        path['block'] = row.block
        path['block_num'] = row.block_num
        path['paired_block'] = row.paired_block
        path['stimulus'] = row.stimuluss
        path['start_loc_x'] = row.start_loc_x
        path['start_loc_y'] = row.start_loc_y
        path['obj_loc_x'] = row.obj_loc_x
        path['obj_loc_y'] = row.obj_loc_y
        path['resp_loc_x'] = row.resp_loc_x
        path['resp_loc_y'] = row.resp_loc_y
        path['session'] = row.session
        path['experiment'] = row.experiment
        path['subject'] = row.subject
        
        # add eegoffset values using slope of regression
        path['eegoffset'] = (row.eegoffset + slope * (path['mstime'] - row.mstime)).astype(int)
        
        # add data to row (from first path event)
        row['direction'] = path.iloc[0].direction
        row['travel_time'] = path.iloc[0].travel_time
        row['x'] = path.iloc[0].x
        row['y'] = path.iloc[0].y
        
        return pd.concat([row.to_frame().T, path], ignore_index=True)

    
    def expand_travel_paths(self, events):
        # expand out locations into x and y
        events[['start_loc_x', 'start_loc_y']] = pd.DataFrame(events.start_locs.to_list(), index=events.index)
        events[['obj_loc_x', 'obj_loc_y']] = pd.DataFrame(events.obj_locs.to_list(), index=events.index)
        events[['resp_loc_x', 'resp_loc_y']] = pd.DataFrame(events.resp_locs.to_list(), index=events.index)

        # get slope from regression of eegeffset and mstime
        slope, _, _, _, _ = scipy.stats.linregress(events.mstime, events.eegoffset)

        # expand path data for each event
        new_evs = []
        for _, row in events.iterrows():
            expanded_evs = self.apply_data_to_path(row, slope)
            new_evs.append(expanded_evs)

        return pd.concat(new_evs, ignore_index=True)
    
    def make_events_descriptor(self):
        descriptions = {

        }
        HED = {

        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    
    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'spatial memory')
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = False
        return sidecar
