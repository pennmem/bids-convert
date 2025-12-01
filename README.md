# bids-convert
Tools for converting CML data to the Brain Imaging Data Structure (BIDS). See format specification docs [here](https://bids-specification.readthedocs.io/en/stable/).

Requires [CMLReaders](https://github.com/pennmem/cmlreaders) as well as [MNE-BIDS](https://github.com/mne-tools/mne-bids)

BIDS formatting is required for upload to [OpenNeuro](https://openneuro.org).

NOTE: Currently this repo only supports conversion for scalp EEG, and specifically has been implemented for the PEERS studies (ltpFR, ltpFR2, VFFR) and ValueCourier. 
