# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class PAL3_BIDS_converter(intracranial_BIDS_converter):
    wordpool = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/PAL3/wordpools/wordpool.txt', dtype=str)
     
    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/PAL3/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type'] == 'STUDY_PAIR']

        if all([x in self.wordpool for x in word_evs.study_1]) and all([x in self.wordpool for x in word_evs.study_2]):
            wordpool_file = 'wordpools/wordpool.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = events[events.mstime != -1].reset_index(drop=True)                    # drop events with no mstime values
        events = self.unpack_stim_params(events)                                       # convert stimulation parameters to columns
        events['stim_duration'] = events['stim_duration'].replace([480, 490], 500)     # fix incorrect stim durations

        return events
    
    # unpack stimulation parameters from dictionary and add as columns to events dataframe
    def unpack_stim_params(self, events):
        stim_params_df = pd.DataFrame()
        for _, row in events.iterrows():
            stim_params_df = pd.concat([stim_params_df, pd.DataFrame.from_dict([row.stim_params])], ignore_index=True)

        return pd.concat([events, stim_params_df], axis=1)
    
    def make_events_descriptor(self):
        return {'key': 'val'}
    
    """
    def events_to_BIDS(self):
        events = self.reader.load('events')


        raise NotImplementedError
    

    def make_events_descriptor(self):
        descriptions = {
            'SESS_START': "Start of session.",
            'SESS_END': "End of session.",
            'MIC_TEST': "Microphone test.",
            'INSTRUCT_VIDEO_ON': "Start of instructions.",
            'INSTRUCT_VIDEO_OFF': "End of instructions.",
            'TRIAL': "Denotes new trial (list).",
            'STUDY_ORIENT': "Fixation prior to word-pair presentation.",            #?
            'STUDY_ORIENT_OFF': "Fixation offset prior to word-pair presentation.",
            'ENCODING_START': "Start of encoding phase.",
            'ENCODING_END': "End of encoding phase.",
            'STUDY_PAIR': "Word-pair presentation onset.",
            'PAIR_OFF': "Word-pair presentation offset.",     #?



            'START': "Start of math distractor phase.",
            'STOP': "End of math distactor phase.",
            'MATH_START': "Start of math distractor phase.",    #?
            'MATH_END': "End of math distractor phase.",      #?
            'PROB': "Math problem presentation onset.",
            'REC_START': "(Single) cued recall begins.",       #?
            'RECALL_END': "",
            'TEST_ORIENT': "Fixation prior to cued recall."    #?
            'TEST_PROBE': "Recall probe.",
            'PROBE_OFF': "Recall probe offset",

            'STIM_ON': "Onset of electrical stimulation."
            'STIM_OFF': "Offset of electrical stimulation."
            

        }
        ,  
        , ', , ,
        , 'PRACTICE_ORIENT', 'PRACTICE_ORIENT_OFF',
        'PRACTICE_PAIR', 'PRACTICE_PAIR_OFF', 'PRACTICE_PROBE',
        'PRACTICE_RETRIEVAL_ORIENT', 'PRACTICE_RETRIEVAL_ORIENT_OFF',
        'PRACTICE_TRIAL',   '
        'RECALL_START', 'REC_END', 'REC_EVENT', 
        'RETRIEVAL_ORIENT_OFF', ' , ,
        , , , ', ,
        , ,  
    """

    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'cued recall of paired associates with closed-loop stimulation at encoding')              # place in second column
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar
