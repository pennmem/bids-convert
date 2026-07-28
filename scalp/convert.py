#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import sys
from ScalpBIDSConverter import *
import argparse

def convert_to_bids(subject, experiment, session,
                    root="/data8/PEERS_BIDS/",
                    overwrite=False,
                    force=False):
    """
    Wrapper around ScalpBIDSConverter for a single session.

    For anything beyond a one-off single session, use the repo entry point
    (``bids_convert.py``), which handles job building, parallelism, error
    logging and validation.
    """
    overrides = {stage: bool(overwrite) for stage in ScalpBIDSConverter.ALL_STAGES}
    converter = ScalpBIDSConverter(
        subject=subject,
        experiment=experiment,
        session=session,
        root=root,
        overrides=overrides,
        force=force,
    )
    converter.run()
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
        help="Experiment name (e.g., ltpFR, ltpFR2, VFFR, ValueCourier, VCBehOnly)."
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
        "--overwrite",
        action="store_true",
        help="Re-convert every stage even if its outputs already exist."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Downgrade per-stage conversion failures to warnings and "
             "continue. By default any stage failure is a hard error."
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    convert_to_bids(
        subject=args.subject,
        experiment=args.experiment,
        session=args.session,
        root=args.root,
        overwrite=args.overwrite,
        force=args.force,
    )
