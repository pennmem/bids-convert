# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids


class intracranial_BIDS_converter:
    # class 
    ELEC_TYPES_DESCRIPTION = {'S': 'strip', 'G': 'grid', 'D': 'depth', 'uD': 'micro'}
    ELEC_TYPES_BIDS = {'S': 'ECOG', 'G': 'ECOG', 'D': 'SEEG', 'uD': 'SEEG'}

    # initialize
    def __init__(self, subject, experiment, session, root='/scratch/hherrema/BIDS_storage/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.root = root
    
        
    # instantiate CMLReader object, save as attribute\
    def cml_reader(self):
        df = cml.get_data_index('r1', '/')
        sel = df.query("subject==@self.subject & experiment==@self.experiment & session==@self.session").iloc[0]
        reader = cml.CMLReader(subject=sel.subject, experiment=sel.experiment, session=sel.session, 
                               localization=sel.localization, montage=sel.montage)
        return reader
    
    # ---------- BIDS Utility ----------
    # return a base BIDS_path object to 
    def BIDS_path(self):
        bids_path = mne_bids.BIDS_path(subject=self.subject, task=self.experiment, session=str(self.session),
                                       root=self.root)
        return bids_path
    # ---------- Events ----------
    def set_wordpool(self):
        raise NotImplementedError
    
    def events_to_BIDS(self):
        raise NotImplementedError
    
    def make_events_descriptor(self):
        raise NotImplementedError
    
    @class
    def write_BIDS_beh(self):
        bids_path = cls.BIDS_path().update(suffix='beh', extension='.tsv', datatype='beh')
