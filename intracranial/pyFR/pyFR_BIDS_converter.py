# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
import scipy
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class pyFR_BIDS_converter(intracranial_BIDS_converter):
    wordpool_EN = np.loadtxt('pyFR/wordpools/wordpool_EN.txt', dtype=str)
    CH_TYPES = {'TJ027': 'ECOG', 'TJ029': 'SEEG', 'TJ030': 'SEEG', 'TJ032': 'ECOG', 'TJ061': 'ECOG', 'TJ083':'ECOG',
                'UP004': 'SEEG', 'UP008':'ECOG', 'UP011':'ECOG', 'UP037': 'ECOG'}

    # initialize
    # just hand empty dictionary for brain_regions
    def __init__(self, subject, experiment, session, montage, math_events, system_version, unit_scale, overrides=None, root='/scratch/hherrema/BIDS/pyFR/'):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.montage = montage
        self.math_events = math_events
        self.system_version = system_version
        self.unit_scale = unit_scale
        self.root = root
        self.overrides = overrides or {}

    # ---------- BIDS Utility ---------
    # return a base BIDS_path object to update
    def _BIDS_path(self):
        # update session if re-implant
        new_session = self.reassign_session()
        if new_session:
            bids_path = mne_bids.BIDSPath(subject=self.subject, task=self.experiment, session=str(new_session),
                                            root=self.root)
        else:
            bids_path = mne_bids.BIDSPath(subject=self.subject, task=self.experiment, session=str(self.session),
                                            root=self.root)
        return bids_path
    # instantiate CMLRead object, save as attribute
    def cml_reader(self):
        df = cml.get_data_index()
        sel = df.query("subject==@self.subject & experiment==@self.experiment & session==@self.session & montage==@self.montage").iloc[0]
        reader = cml.CMLReader(subject=sel.subject, experiment=sel.experiment, session=sel.session, 
                               localization=sel.localization, montage=sel.montage)
        return reader
    
    def reassign_session(self):
        re_implants = pd.read_csv('pyFR/metadata/re_implants.csv')
        ri = re_implants[(re_implants.subject == self.subject) & (re_implants.montage == self.montage) &
                         (re_implants.session == self.session)]
        if len(ri) == 1:
            return ri.iloc[0].new_session
        else:
            return None
        
    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']
        if np.all([x in self.wordpool_EN for x in word_evs.item]):
            wordpool_file = 'wordpools/wordpool_EN.txt'
        else:
            wordpool_file = 'n/a'
        
        return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        # load in math events
        if self.math_events:
            if self.montage != 0:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{self.subject}_{self.montage}_math.mat', squeeze_me=True)['events'])
            else:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{self.subject}_math.mat', squeeze_me=True)['events'])
            math_evs = math_evs[math_evs.session == self.session]                                        # select out session
            math_evs = math_evs[(math_evs.type != 'B') & (math_evs.type != 'E')]                         # remove the B and E events from math evs
            math_evs['list'] = math_evs['list'] - 1                                                      # math events given list + 1
            events = pd.concat([math_evs, events], ignore_index=True)
            events = events.sort_values(by='mstime', ascending=True, ignore_index=True)                  # sort in chronological order
        
        events['experiment'] = self.experiment                                                       # math events don't have experiment field
        # transformations
        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})                  # rename columns
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0        # event duration [s]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                  # replace events with negative duration with 0.0 s
        events = self.apply_event_durations(events)                                                  # apply well-defined durations [s]
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                           # onset from first event [s]
        events['response_time'] = 'n/a'                                                              # response time [s]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV') |
                   (events.trial_type=='PROB'), 'response_time'] = events['rectime'] / 1000.0
        events['stim_file'] = np.where(events.trial_type=='WORD', self.wordpool_file, 'n/a')              # add wordpool to word events
        events['item_name'] = events.item.replace('X', 'n/a')
        events = events.fillna('n/a')                                                                # chnage NaN to 'n/a'
        events = events.replace('', 'n/a')                                                           # no empty cells

        # update session if re-implant
        new_session = self.reassign_session()
        if new_session:
            events['session'] = new_session
            events['subject'] = self.subject   # remove _montage from subject
        
        if self.math_events:
            events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 
                            'stim_file', 'item_name', 'serialpos', 'list', 'test', 'answer', 
                            'experiment', 'session', 'subject']]                                        # re-order columns + drop unneeded fields
        else:
            events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time',
                             'stim_file', 'item_name', 'serialpos', 'list',
                             'experiment', 'session', 'subject']]
        return events
    
    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # fixation events = 1600 ms
            if row.trial_type == 'ORIENT':
                durations.append(1.6)

            # word events = 1600 ms
            elif row.trial_type == 'WORD':
                durations.append(1.6)

            # keep current duration
            else:
                durations.append(row.duration)

        events['duration'] = durations        # preserves column order
        return events
    
    def make_events_descriptor(self):
        descriptions = {
            'B': 'Beginning of session.',
            'SESS_START': 'Beginning of session.',
            'E': 'End of session.',
            'START': "Beginning of math distractor phase.",
            'PROB': "Math problem presentation onset.",
            'STOP': "End of math distractor phase.",
            'SESS_END': 'End of session.',
            'ORIENT': 'Fixation preceding presentation of word list.',
            'REC_START': 'Start of recall phase.',
            'TRIAL': 'Denotes new trial (list).',
            'WORD': 'Word presentation (onset).',
            'REC_WORD': 'Recalled word, onset of speech (during free recall).',
            'REC_WORD_VV': 'Vocalization, onset of speech (during free recall).'
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category", 
                           "Description": "Indicator of type of task action that occurs at the marked time", 
                           "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()}},
            "response_time": {"Description": "Time (in seconds) between onset of recall phase and recall (for recalls and vocalizations), or between onset of problem on screen and response (for math problems)."},
            "stim_file": {"LongName": "Stimulus File", 
                          "Description": "Location of wordpool file containing words presented in WORD events."},
            "subject": {"LongName": "Subject ID",
                        "Description": "The string identifier of the subject, e.g. R1001P."},
            'experiment': {'Description': 'The experimental paradigm completed.'},
            "session": {"Description": "The session number."},
            "list": {"LongName": "List Number",
                     "Description": "Word list (1-24) during which the event occurred. Trial <= 0 indicates practice list."},
            "item_name": {"Description": "The word being presented or recalled in a WORD or REC_WORD event."},
            'serialpos': {'LongName': 'Serial Position', 
                          'Description': 'The order position of a word presented in an WORD event.'},
            'test': {"LongName": "Math problem", 
                     "Description": "Math problem with form X + Y + Z = ?  Stored in list [X, Y, Z]."},
            'answer': {"LongName": "Math problem response", 
                       "Description": "Participant answer to problem with form X + Y + Z = ?  Note this is not necessarily the correct answer."}
        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    
    # ---------- Electrodes ----------
    def load_contacts(self):
        contacts = self.reader.load('contacts')
        contacts['type'] = [x if type(x)==str else 'n/a' for x in contacts.type]      # replace missing types with 'n/a'
        return contacts
    
    def _load_mni_coords(self):
        """Load the per-subject MNI coords text file. Returns Nx4 array
        ([contact, x, y, z]) or None if the file isn't available."""
        try:
            if self.montage != 0:
                path = f'/data/eeg/{self.subject}_{self.montage}/tal/RAW_coords.txt.mni'
            else:
                path = f'/data/eeg/{self.subject}/tal/RAW_coords.txt.mni'
            return np.loadtxt(path)
        except (OSError, ValueError):
            return None

    def _available_cml_spaces(self):
        available = []
        cols = set(self.contacts.columns)
        if {'x', 'y', 'z'}.issubset(cols):
            available.append('tal')
        if self._load_mni_coords() is not None:
            available.append('mni')
        return available

    def contacts_to_electrodes(self, cml_space):
        # Import here to avoid a cycle with the package __init__.
        from ..intracranial_BIDS_converter import CML_TO_BIDS_SPACE  # noqa: F401

        electrodes = pd.DataFrame({'name': np.array(self.contacts.label)})

        if cml_space == 'tal':
            electrodes['x'] = self.contacts['x'].astype(float)
            electrodes['y'] = self.contacts['y'].astype(float)
            electrodes['z'] = self.contacts['z'].astype(float)
        elif cml_space == 'mni':
            mni_coords = self._load_mni_coords()
            contacts_mask = [
                i for i, c in enumerate(mni_coords[:, 0])
                if int(c) in np.array(self.contacts.contact)
            ]
            mni_contacts = mni_coords[contacts_mask, :]
            electrodes['x'] = mni_contacts[:, 1]
            electrodes['y'] = mni_contacts[:, 2]
            electrodes['z'] = mni_contacts[:, 3]
        else:
            electrodes['x'] = np.nan
            electrodes['y'] = np.nan
            electrodes['z'] = np.nan

        electrodes['size'] = -999
        electrodes['group'] = np.array(self.contacts.grpName)
        electrodes['hemisphere'] = ['L' if 'Left' in x else 'R' if 'Right' in x else 'n/a' for x in self.contacts.Loc1]
        electrodes['type'] = [self.ELEC_TYPES_DESCRIPTION.get(x) for x in self.contacts.type]
        electrodes['lobe'] = np.array(self.contacts.Loc2)
        electrodes['region1'] = np.array(self.contacts.Loc3)
        electrodes['region2'] = np.array(self.contacts.Loc5)
        electrodes['gray_white'] = np.array(self.contacts.Loc4)
        electrodes = electrodes.fillna('n/a')
        electrodes = electrodes.replace('', 'n/a')
        electrodes = electrodes[['name', 'x', 'y', 'z', 'size', 'group', 'hemisphere', 'type',
                                 'lobe', 'region1', 'region2', 'gray_white']]
        return electrodes

    def make_electrodes_sidecar(self, cml_space):
        from ..intracranial_BIDS_converter import CML_TO_BIDS_SPACE
        bids_space = CML_TO_BIDS_SPACE[cml_space]
        sidecar = {
            'name': 'Label of electrode',
            'x': f'x-axis position in {bids_space} coordinates',
            'y': f'y-axis position in {bids_space} coordinates',
            'z': f'z-axis position in {bids_space} coordinates',
            'size': 'Surface area of electrode.',
            'group': 'Group of channels electrode belongs to (same shank).',
            'hemisphere': 'Hemisphere of electrode location.',
            'type': 'Type of electrode.',
            'lobe': 'Brain lobe of electrode location.',
            'region1': 'Brain region of electrode location.',
            'region2': 'Brain region of electrode location.',
            'gray_white': 'Denotes gray or white matter.',
        }
        return sidecar
    
    # ---------- Channels ----------
    def load_pairs(self):
        pairs = self.reader.load('pairs')
        pairs['type'] = [x if type(x)==str else 'n/a' for x in pairs.type]        # replace missing types with 'n/a'
        return pairs
    
    def pairs_to_channels(self):
        channels = pd.DataFrame({'name': np.array(self.pairs.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x.upper()) for x in self.pairs.type]
        channels['units'] = 'V'                                                    # convert EEG to V
        channels['low_cutoff'] = 'n/a'                                             # highpass filter (don't actually know this for clinical eeg)
        channels['high_cutoff'] = 'n/a'                                            # lowpass filter (mne adds Nyquist frequency = 2 x sampling rate)
        channels['reference'] = 'bipolar'
        channels['group'] = np.array(self.pairs.grpName)
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x.upper()) for x in self.pairs.type]
        channels['notch'] = 'n/a'
        channels = channels.fillna('n/a')                                          # remove NaN
        channels = channels.replace('', 'n/a')                                     # no empty cells
        return channels
    
    def contacts_to_channels(self):
        channels = pd.DataFrame({'name': np.array(self.contacts.label)})
        channels['type'] = [self.ELEC_TYPES_BIDS.get(x.upper()) if x.upper() in self.ELEC_TYPES_BIDS.keys() else 
                            self.CH_TYPES.get(self.subject) for x in self.contacts.type]
        channels['units'] = 'V'                                                    # convert EEG to V
        channels['low_cutoff'] = 'n/a'                                             # highpass filter (don't actually know this for clinical eeg)
        channels['high_cutoff'] = 'n/a'                                            # lowpass filter (mne adds Nyquist frequency = 2 x sampling rate)
        channels['group'] = np.array(self.contacts.grpName)
        channels['sampling_frequency'] = self.sfreq
        channels['description'] = [self.ELEC_TYPES_DESCRIPTION.get(x.upper()) for x in self.contacts.type]
        channels['notch'] = 'n/a'
        channels = channels.fillna('n/a')                                          # remove NaN
        channels = channels.replace('', 'n/a')                                     # no empty cells
        return channels
    
    # ---------- EEG ----------
    # set sfreq and recording_duration attributes
    def eeg_metadata(self):
        eeg = self.reader.load_eeg()
        sfreq = eeg.samplerate
        recording_duration = eeg.data.shape[-1] / sfreq
        return sfreq, recording_duration
    
    def eeg_sidecar(self, ref):                 # overwrite for different 'type' field
        sidecar = {'TaskName': self.experiment}
        sidecar['TaskDescription'] = 'delayed free recall of word lists'
        sidecar['SamplingFrequency'] = float(self.sfreq)
        sidecar['PowerLineFrequency'] = 60.0    # check for German subjects
        sidecar['SoftwareFilters'] = 'n/a'
        sidecar['HardwareFilters'] = 'n/a'
        sidecar['RecordingDuration'] = float(self.recording_duration)
        sidecar['RecordingType'] = 'continuous'
        sidecar['ElectricalStimulation'] = False
        if ref == 'bipolar':
            sidecar['iEEGReference'] = 'bipolar'
            sidecar['ECOGChannelCount'] = len(self.pairs[(self.pairs.type=='S') | (self.pairs.type=='G')].index)
            sidecar['SEEGChannelCount'] = len(self.pairs[(self.pairs.type=='D') | (self.pairs.type=='uD')].index)
            sidecar['MiscChannelCount'] = len(self.pairs[self.pairs.type=='n/a'].index)
            sidecar['EEGChannelCount'] = 0
        elif ref == 'monopolar':
            sidecar['iEEGReference'] = 'monopolar'
            sidecar['ECOGChannelCount'] = len(self.contacts[(self.contacts.type=='S') | (self.contacts.type=='G')].index)
            sidecar['SEEGChannelCount'] = len(self.contacts[(self.contacts.type=='D') | (self.contacts.type=='uD')].index)
            sidecar['MiscChannelCount'] = len(self.contacts[self.contacts.type=='n/a'].index)
            sidecar['EEGChannelCount'] = 0
        return sidecar
    
    # ---------- EEG (monopolar) ----------
    def eeg_mono_to_BIDS(self):
        eeg = self.reader.load_eeg(scheme=self.contacts)
        eeg.data = eeg.data / self.unit_scale              # convert to V before instantiating raw object
        eeg_mne = eeg.to_mne()
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_mono.type]))    # ecog or seeg
        eeg_mne.set_channel_types(mapping)                                                     # set channel types

        return eeg_mne
    
    # ---------- EEG (bipolar) ----------
    def eeg_bi_to_BIDS(self):
        eeg = self.reader.load_eeg(scheme=self.pairs)
        eeg.data = eeg.data / self.unit_scale               # convert to V before instantiating raw object
        eeg_mne = eeg.to_mne()
        mapping = dict(zip(eeg_mne.ch_names, [x.lower() for x in self.channels_bi.type]))
        eeg_mne.set_channel_types(mapping)

        return eeg_mne
    
    # ---------------------------------
    # run conversion
    def run(self):
        self.reader = self.cml_reader()

        # ---------- Behavioral ----------
        if self._should_run('behavioral'):
            self.wordpool_file = self.set_wordpool()
            self.events = self.events_to_BIDS()
            self.events_descriptor = self.make_events_descriptor()
            self.write_BIDS_beh()
        else:
            print(f"SKIP: behavioral outputs exist for {self.subject}/{self.experiment}/ses-{self.session}")

        # We always attempt both monopolar and bipolar; per-acquisition
        # blocks below catch and warn on missing data instead of crashing.
        run_mono_eeg = self._should_run('mono-eeg')
        run_bi_eeg = self._should_run('bi-eeg')
        run_mono_channels = self._should_run('mono-channels')
        run_bi_channels = self._should_run('bi-channels')
        run_electrodes = self._should_run('electrodes')

        needs_eeg_meta = run_mono_eeg or run_bi_eeg or run_mono_channels or run_bi_channels
        needs_contacts = run_electrodes or run_mono_eeg or run_mono_channels
        needs_pairs = run_bi_eeg or run_bi_channels

        if needs_eeg_meta:
            self.sfreq, self.recording_duration = self.eeg_metadata()

        if needs_contacts:
            self.contacts = self.load_contacts()

        # ---------- Electrodes ----------
        if run_electrodes:
            available = self._available_cml_spaces()
            if not available:
                print(f"WARNING: no known CML coordinate spaces found for {self.subject}")
            for cml_space in available:
                electrodes = self.contacts_to_electrodes(cml_space)
                sidecar = self.make_electrodes_sidecar(cml_space)
                self.write_BIDS_electrodes(cml_space, electrodes, sidecar)
                self.write_BIDS_coords(cml_space)
        else:
            print(f"SKIP: electrodes outputs exist for {self.subject}/{self.experiment}/ses-{self.session}")

        # ---------- Bipolar (channels + EEG) ----------
        if needs_pairs:
            try:
                self.pairs = self.load_pairs()
            except Exception as e:
                print(f"WARNING: bipolar pairs unavailable for {self.subject}/{self.experiment}/ses-{self.session} — skipping bipolar stages ({type(e).__name__}: {e})")
                run_bi_channels = run_bi_eeg = False

        if run_bi_eeg:
            try:
                self.eeg_sidecar_bi = self.eeg_sidecar('bipolar')
                self.eeg_bi = self.eeg_bi_to_BIDS()
                self.write_BIDS_ieeg('bipolar')
            except Exception as e:
                print(f"WARNING: bi-eeg failed for {self.subject}/{self.experiment}/ses-{self.session} — skipping ({type(e).__name__}: {e})")

        if run_bi_channels:
            try:
                self.channels_bi = self.pairs_to_channels()
                self.write_BIDS_channels('bipolar')
            except Exception as e:
                print(f"WARNING: bi-channels failed for {self.subject}/{self.experiment}/ses-{self.session} — skipping ({type(e).__name__}: {e})")

        if run_mono_eeg:
            try:
                self.eeg_sidecar_mono = self.eeg_sidecar('monopolar')
                self.eeg_mono = self.eeg_mono_to_BIDS()
                self.write_BIDS_ieeg('monopolar')
            except Exception as e:
                print(f"WARNING: mono-eeg failed for {self.subject}/{self.experiment}/ses-{self.session} — skipping ({type(e).__name__}: {e})")

        if run_mono_channels:
            try:
                self.channels_mono = self.contacts_to_channels()
                self.write_BIDS_channels('monopolar')
            except Exception as e:
                print(f"WARNING: mono-channels failed for {self.subject}/{self.experiment}/ses-{self.session} — skipping ({type(e).__name__}: {e})")

        return True
