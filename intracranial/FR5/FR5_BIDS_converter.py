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
    wordpool = None

    # initialzie
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/FR5/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)


    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')

        raise NotImplementedError
    

    def events_to_BIDS(self):
        events = self.reader.load('events')