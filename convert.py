#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import sys
from ScalpBIDSConverter import *

def convert_to_bids(subject, experiment, session):
#     if os.path.exists(f"/data8/PEERS_BIDS/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{experiment}_eeg.edf"):
#         return True
    converter = ScalpBIDSConverter(subject, experiment, session, root="/data8/PEERS_BIDS/", overwrite_eeg=True, overwrite_beh=True)
    return True

if __name__=="__main__":
    index = int(sys.argv[1])
    data = pd.read_csv("/home1/jrudoler/bids-convert/peers_sessions.csv")
    convert_to_bids(**data.iloc[index].to_dict())
