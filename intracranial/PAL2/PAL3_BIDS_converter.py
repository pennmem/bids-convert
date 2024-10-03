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
     
    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/PAL3/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)
