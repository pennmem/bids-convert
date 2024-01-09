(Intracranial) BIDS conversion design document.

Abstract Class = BIDS_converter.
For each experiment, a subclass that extends the superclass.

Methods:
- cml_reader = instantiates a CMLReader object
- set_wordpool = sets worpool_file attribute to include in all word presentation events
- events_to_BIDS = loads event, converts to BIDS compliant format
- make_events_descriptor = make descriptions for json file
- write_BIDS_beh = write beh subfolder (holds events tsv and json)
- load_contacts = load monopolar contacts
- generate_area_map = makes dictionary mapping contact group to size
- contacts_to_electrodes = convert contacts to electrodes tsv
- write_BIDS_electrodes = write electrodes tsv
- coordinate_system = create coordinate system json
- write_BIDS_coords = write coordinate system json
- load_pairs = load bipolar pairs
- pairs_to_channels = convert bipolar pairs to channels tsv
- contacts_to_channels = convert monopolar contacts to channels tsv
- write_BIDS_channels = write channels tsv
- eeg_metadata = set sampling rate and recording duration attributes
- eeg_sidecar = generate json file to accompany EEG file
- write_BIDS_ieeg = write edf file and accompanying json file
- eeg_mono_to_BIDS = rescale EEG data, convert to mne, add channel types (monopolar)
- eeg_bi_to_BIDS = rescale EEG data, convert to mne, add channel types (bipolar)
- run
