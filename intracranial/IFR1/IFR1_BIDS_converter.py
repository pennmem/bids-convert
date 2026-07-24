# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from pathlib import Path
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

_HERE = Path(__file__).parent

class IFR1_BIDS_converter(intracranial_BIDS_converter):
    """Elemem / System-4 free recall.

    Behaviorally the same paradigm as FR1, so this mirrors
    ``FR1_BIDS_converter`` closely. Two differences drive the deviations
    below, both verified across all 56 sessions in the r1 data index:

    * **No math distractor.** ``DISTRACT_START``/``DISTRACT_END`` bracket an
      empty interval; there are no ``PROB`` events, no ``math_events`` file
      and no ``test``/``answer`` columns, so those columns are absent from
      the output entirely.
    * **Dead recognition columns.** ``recog_resp``, ``recog_rt``,
      ``recognized`` and ``rejected`` are -999 in every session (there is no
      recognition phase in the data), and ``phase`` only ever holds ''
      or 'NON-STIM'. All five are dropped rather than written as 'n/a'.
    """

    wordpool_EN = np.loadtxt(_HERE / 'wordpools' / 'wordpool_EN.txt', dtype=str)
    wordpool_SP = np.loadtxt(_HERE / 'wordpools' / 'wordpool_SP.txt', dtype=str)
    wordpool_short_EN = np.loadtxt(_HERE / 'wordpools' / 'wordpool_short_EN.txt', dtype=str)
    wordpool_long_EN = np.loadtxt(_HERE / 'wordpools' / 'wordpool_long_EN.txt', dtype=str)
    wordpool_long_SP = np.loadtxt(_HERE / 'wordpools' / 'wordpool_long_SP.txt', dtype=str)

    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, area, brain_regions, overrides=None, root='/scratch/hherrema/BIDS/IFR1/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, area, brain_regions, overrides, root)

    # ---------- Events ----------
    def set_wordpool(self):
        # Same cascade as FR1. Every IFR1 session on rhino resolves to
        # wordpool_long_EN.txt (checked against each session's own
        # experiment_files/wordpool.txt, which is byte-identical to it).
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']; word_evs = word_evs[word_evs['list']!=-1]    # remove practice list
        if np.all([1 if x in self.wordpool_EN else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_EN.txt'
        elif np.all([1 if x in self.wordpool_short_EN else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_short_EN.txt'
        elif np.all([1 if x in self.wordpool_long_EN else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_long_EN.txt'
        elif np.all([1 if x in self.wordpool_SP else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_SP.txt'
        elif np.all([1 if x in self.wordpool_long_SP else 0 for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_long_SP.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file

    def events_to_BIDS(self):
        events = self._load_events()
        events = cml.correct_retrieval_offsets(events, self.reader)            # apply offset corrections
        events = cml.correct_countdown_lists(events, self.reader)              # apply countdown list corrections
        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})                      # rename columns
        events['onset'] = self._onset_from_sample(events)                                                # onset from eegoffset/sfreq [s] (recording-relative)
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0            # event duration [s]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                      # replace events with negative duration with 0.0 s
        events = self.apply_event_durations(events)                                                      # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                                  # response time [s]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV'),
                   'response_time'] = events['rectime'] / 1000.0                                         # use rectime (no PROB events in IFR1)
        events['stim_file'] = np.where((events.trial_type=='WORD') & (events.list!=-1), self.wordpool_file, 'n/a')    # add wordpool to word events
        events['item_name'] = events.item_name.replace('X', 'n/a')
        events = events.fillna('n/a')                                                                    # change NaN to 'n/a'
        events = events.replace('', 'n/a')                                                               # resolve empty cell issue

        # Column whitelist. Everything not listed is dropped, including the
        # always--999 recognition fields (recog_resp, recog_rt, recognized,
        # rejected), the constant `phase`, and the usual FR1 exclusions
        # (is_stim, stim_list, stim_params, mstime, protocol, item_num,
        # eegfile, exp_version, montage, msoffset, intrusion, recalled).
        events = events[self._append_uncorrected_cols(events, ['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file',
                        'item_name', 'serialpos', 'list', 'experiment', 'session', 'subject'])]           # re-order columns

        return events

    def apply_event_durations(self, events):
        durations = []
        for _, row in events.iterrows():
            # countdown events = 10000 ms (measured median 10.4 s)
            if row.trial_type == 'COUNTDOWN' or row.trial_type == 'COUNTDOWN_START':
                durations.append(10.0)

            # word presentation events = 1600 ms (measured median 1602 ms)
            elif row.trial_type == 'WORD':
                durations.append(1.6)

            # keep current duration
            else:
                durations.append(row.duration)

        events['duration'] = durations        # preserves column order
        return events

    def make_events_descriptor(self):
        descriptions = {
            "SESS_START": "Beginning of session.",
            "SESS_END": "End of session.",
            "WORD": "Word presentation onset.",
            "REC_START": "Recall phase begins.",
            "REC_END": "Recall phase ends.",
            "REC_WORD": "Recalled word, onset of speech (during free recall).",
            "REC_WORD_VV": "Vocalization (during free recall).",
            'COUNTDOWN': 'Beginning of pre-list presentation countdown.',
            'COUNTDOWN_START': 'Beginning of pre-list presentation countdown.',
            'COUNTDOWN_END': 'End of pre-list presentation countdown.',
            'DISTRACT_START': 'Beginning of distractor phase.',
            'DISTRACT_END': 'End of distractor phase.',
            'TRIAL': 'Denotes new trial (list).',
            'INSTRUCT_START': 'Beginning of instructions.',
            'INSTRUCT_END': 'End of instructions.',
            'ENCODING_START': 'Beginning of word presentation list.',
            'ENCODING_END': 'End of word presentation list.',
            'WORD_OFF': 'End of word presentation.',
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category",
                           "Description": "Indicator of type of task action that occurs at the marked time",
                           "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()}},
            "response_time": {"Description": "Time (in seconds) between onset of recall phase and recall (for recalls and vocalizations)."},
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
        }
        HED.update(self.UNCORRECTED_HED)
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor

    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'delayed free recall of word lists')    # place in second column
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = False    # elemem experiment_config.json: stim_mode = none
        return sidecar
