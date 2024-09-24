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

class YC2_BIDS_converter(intracranial_BIDS_converter):
    wordpool = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/YC2/wordpools/wordpool.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/YC2/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        
        if all([x.upper() in self.wordpool for x in evs.stimulus]):
            wordpool_file = 'wordpools/wordpool.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = self.unpack_stim_params(events)              # convert stimulation parameters into columns
        events = self.expand_travel_paths(events)             # expand events to have rows for travel paths

        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type', 'block_num':'trial', 
                                        'resp_travel_time':'resp_total_time', 'stim_on':'stimulation'})     # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                      # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0   # event duration [s] --> lots of superfluous events may mess this up
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)             # replace events with negative duration with 0.0s
        events = self.apply_event_durations(events)                                             # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                         # response time [s]
        events.loc[(events.trial_type=='NAV_TEST') | (events.trial_type=='NAV_PRACTICE_TEST'),
                    'response_time'] = events['resp_total_time']
        events['stim_file'] = np.where(events.trial_type.isin(['NAV_PRACTICE_LEARN', 'NAV_LEARN']), 
                                       self.wordpool_file, 'n/a')                               # add wordpool to learning events
        events['stimulation'] = events['stimulation'].astype(int)                               # True = 1, False = 0
        events.loc[events.stimulation == 0, ['anode_label', 'cathode_label']] = ''              # set stim parameters to defaults if no stimulation
        events.loc[events.stimulation == 0, ['stim_duration', 'amplitude', 'pulse_freq', 'n_pulses', 'pulse_width']] = 0
        events = events.fillna('n/a')                                                           # change NaN to 'n/a'
        events = events.replace('', 'n/a')                                                      # no empty cells

        # select and re-order columns
        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'stimulus', 
                         'trial', 'block', 'paired_block', 'start_loc_x', 'start_loc_y', 'obj_loc_x', 'obj_loc_y', 'resp_loc_x', 'resp_loc_y', 
                         'resp_reaction_time', 'resp_total_time', 'resp_path_length', 'x', 'y', 'direction', 'travel_time', 
                         'stimulation', 'stim_duration', 'anode_label', 'cathode_label', 'amplitude', 'pulse_freq', 'n_pulses', 'pulse_width',
                         'session', 'experiment', 'subject']]

        raise NotImplementedError
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # learning events = 5000 ms (1s turn, 3s drive, 1s pause)
            if row.trial_type == 'NAV_LEARN' or row.trial_type == 'NAV_PRACTICE_LEARN':
                durations.append(5.0)

            # stimulation events not logged (5000 ms)

            # keep current duration
            else:
                durations.append(row.duration)
            
        events['duration'] = durations
        return events
    
    # unpack stimulation parameters from dictionary and add as columns to events dataframe
    def unpack_stim_params(self, events):
        stim_params_df = pd.DataFrame()
        for _, row in events.iterrows():
            stim_params = pd.DataFrame.from_dict(row.stim_params)   # dictionary already in list
            
            # no stimulation on test trials
            if row.type.isin(["NAV_TEST", "NAV_PRACTICE_TEST"]):
                stim_params['stim_on'] = False

            stim_params_df = pd.concat([stim_params_df, stim_params], ignore_index=True)

        return pd.concat([events, stim_params_df], axis=1)
    
    def apply_data_to_path(self, row, slope):
        path = pd.DataFrame(row.path)      # extract path data
        trial_type = row['type']
        
        path = path.rename(columns={'time': 'travel_time'})
        
        if trial_type == 'NAV_PRACTICE_LEARN':
            tt = 'TRAVEL_PRACTICE_LEARN'
        elif trial_type == 'NAV_PRACTICE_TEST':
            tt = 'TRAVEL_PRACTICE_TEST'
        elif trial_type == 'NAV_LEARN':
            tt = 'TRAVEL_LEARN'
        elif trial_type == 'NAV_TEST':
            tt = 'TRAVEL_TEST'
        else:
            raise ValueError(f'{trial_type} is not a valid trial type.')
        
        path['type'] = tt
        path['mstime'] = (row.mstime + (1000*row.resp_reaction_time) + path['travel_time']).astype(int)  # reaction time + travel time
        path['block'] = row.block
        path['block_num'] = row.block_num
        path['paired_block'] = row.paired_block
        path['stimulus'] = row.stimulus
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
        
        path['travel_time'] = path['travel_time'] / 1000.0    # convert from ms to s
        
        # add data to row (from first path event)
        row['direction'] = path.iloc[0].direction
        row['travel_time'] = 'n/a'
        row['x'] = path.iloc[0].x
        row['y'] = path.iloc[0].y

        # stimulation parameters
        path['amplitude'] = row.amplitude
        path['pulse_freq'] = row.pulse_freq
        path['n_pulses'] = row.n_pulses
        path['pulse_width'] = row.pulse_width
        path['stim_duration'] = row.stim_duration
        
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
            "NAV_PRACTICE_LEARN": "Learning trial with guided navigation and location encoding on a practice trial.",
            "TRAVEL_PRACTICE_LEARN": "Travel during learning practice trial.",
            "NAV_PRACTICE_TEST": "Test trial with self-navigation and location response on a practice trial.",
            "TRAVEL_PRACTICE_TEST": "Travel during test practice trial.",
            "NAV_LEARN": "Learning trial with guided navigation and location encoding.",
            "TRAVEL_LEARN": "Travel during learning trial.",
            "NAV_TEST": "Test trial with self-navigation and location response.",
            "TRAVEL_TEST": "Travel during test trial."
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category", 
                           "Description": "Indicator of type of task action that occurs at the marked time.", 
                           "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()}},
            "response_time": {"Description": "Time (in seconds) between onset of test phase and response."},
            "stim_file": {"LongName": "Stimulus File", 
                          "Description": "Location of wordpool file containing items presented in a NAV_PRACTICE_LEARN or NAV_LEARN event."},
            "stimulus": {"Description": "The object presented in a NAV_PRACTICE_LEARN or NAV_LEARN event."},
            "trial": {"Description": "Experimental trial during which event occurred.  Each overall trial consists of two learning trials and one test trial with the same object location.  Trial = -1 indicates practice trial."},
            "block": {"Description": "Experimental block during which event occurred.  Each block consists of two trials (i.e., 2 x [2 learn, 1 test]) with different object locations."},
            "paired_block": {"Description": "Counterbalanced experimental block of two trials with reflected object and starting locations occurring in opposite order."},
            "start_loc_x": {"LongName": "Start Location (x)",
                            "Description": "x-coordinate of starting location prior to navigation."},
            "start_loc_y": {"LongNmae": "Start Location (y)",
                            "Description": "y-coordinate of starting location prior to navigation."},
            "obj_loc_x": {"LongName": "Object Location (x)",
                          "Description": "x-coordinate of object location."},
            "obj_loc_y": {"LongName": "Object Location (y)",
                          "Description": "y-coordinate of object location."},
            "resp_loc_x": {"LongName": "Response Location (x)",
                           "Description": "x-coordinate of response location."},
            "resp_loc_y": {"LongName": "Response Location (y)",
                           "Description": "y-coordinate of response location."},
            "resp_reaction_time": {"LongName": "Response Reaction Time",
                                    "Description": "Time (in seconds) for navigation to begin, measured from start of trial."},
            "resp_total_time": {"LongName": "Response Total Time",
                                 "Description": "Time (in seconds) until response, measured from start of trial.  Includes reaction and travel time."},
            "resp_path_length": {"LongName": "Response Path Length",
                                 "Description": "Distance traveled prior to response."},
            "x": {"Description" : "x-coordinate of participant location."},
            "y": {"Description": "y-coordinate of participant location."},
            "direction": {"Description": "Direction (in degrees) participant is facing."},
            "travel_time": {"Description": "Time (in seconds) of navigation, measured from start of navigation."},
            "stimulation": {"Description": "Denotes if event occurs during electrical stimulation.  1 indicates stimulation on."},
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
            "session": {"Description": "The session number."},
            'experiment': {'Description': 'The experimental paradigm completed.'},
            "subject": {"LongName": "Subject ID",
                        "Description": "The string identifier of the subject, e.g. R1001P."},
        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    
    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'spatial navigation memory of object locations with open-loop stimulation at encoding')
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar