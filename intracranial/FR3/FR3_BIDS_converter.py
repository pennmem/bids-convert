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
    wordpool = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/FR3/wordpools/wordpool.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/FR3/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)


    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[(evs['type'] == 'WORD') & (evs['list'] > 0)]    # no practice lists
        if all([x in self.wordpool for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file
    

    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = self.unpack_stim_params(events)           # conver stimulation parameters to columns

        raise NotImplementedError
    


    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'free recall with closed-loop stimulation at encoding')              # place in second column
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar