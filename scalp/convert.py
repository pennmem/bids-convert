#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import sys
from ScalpBIDSConverter import *
import argparse

def convert_to_bids(subject, experiment, session):
#     if os.path.exists(f"/data8/PEERS_BIDS/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{experiment}_eeg.edf"):
#         return True
    converter = ScalpBIDSConverter(subject, experiment, session, root="/data8/PEERS_BIDS/", overwrite_eeg=True, overwrite_beh=True)
    return True

# if __name__=="__main__":
#     index = int(sys.argv[1])
#     data = pd.read_csv("/home1/jrudoler/bids-convert/peers_sessions.csv")
#     convert_to_bids(**data.iloc[index].to_dict())

def convert_to_bids(subject, experiment, session,
                    root="/data8/PEERS_BIDS/",
                    overwrite_eeg=True,
                    overwrite_beh=True):
    """
    Wrapper around ScalpBIDSConverter for a single session.
    """
    converter = ScalpBIDSConverter(
        subject=subject,
        experiment=experiment,
        session=session,
        root=root,
        overwrite_eeg=overwrite_eeg,
        overwrite_beh=overwrite_beh,
    )
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert scalp EEG session to BIDS using ScalpBIDSConverter."
    )
    parser.add_argument(
        "--subject", "-s",
        required=True,
        help="Subject ID (e.g., LTP607)."
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        help="Experiment name (e.g., ltpFR, ltpFR2, VFFR, ValueCourier)."
    )
    parser.add_argument(
        "--session", "-n",
        type=int,
        required=True,
        help="Session number (integer)."
    )
    parser.add_argument(
        "--root", "-r",
        default="/data8/PEERS_BIDS/",
        help="Root BIDS directory (default: /data8/PEERS_BIDS/)."
    )
    parser.add_argument(
        "--no-overwrite-eeg",
        action="store_true",
        help="Do not overwrite existing EEG BIDS files."
    )
    parser.add_argument(
        "--no-overwrite-beh",
        action="store_true",
        help="Do not overwrite existing behavioral BIDS files."
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    convert_to_bids(
        subject=args.subject,
        experiment=args.experiment,
        session=args.session,
        root=args.root,
        overwrite_eeg=not args.no_overwrite_eeg,
        overwrite_beh=not args.no_overwrite_beh,
    )
