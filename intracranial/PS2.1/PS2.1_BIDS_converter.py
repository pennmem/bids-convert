# imports
import pandas as pd
import numpy as np
from intracranial.intracranial_BIDS_converter import intracranial_BIDS_converter


class PS21_BIDS_converter(intracranial_BIDS_converter):
    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions,
                 root='/scratch/hherrema/BIDS/PS2.1/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        return 'n/a'

    def events_to_BIDS(self):
        events = self.reader.load('events')
        events = self.unpack_stim_params(events)                        # convert stimulation parameters into columns
        events = events.rename(columns={'eegoffset': 'sample', 'type': 'trial_type', 'stim_on': 'stimulation'})
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0     # onset from first event [s]
        events = self.apply_event_durations(events)                             # apply well-defined durations [s]

        events = events.fillna('n/a')                  # change NaN to 'n/a'
        events = events.replace('', 'n/a')             # no empty cells

        # select and re-order columns
        stim_cols = [c for c in ['amplitude', 'anode_label', 'anode_number', 'burst_freq',
                                  'cathode_label', 'cathode_number', 'n_bursts', 'n_pulses',
                                  'pulse_freq', 'pulse_width', 'stim_duration', 'stimulation']
                     if c in events.columns]
        base_cols = ['onset', 'duration', 'sample', 'trial_type', 'mstime', 'msoffset',
                     'subject', 'experiment', 'session', 'is_stim', 'eegfile', 'exp_version',
                     'montage', 'protocol', 'ad_observed', 'stim_params']
        events = events[[c for c in base_cols if c in events.columns] + stim_cols]

        return events

    def apply_event_durations(self, events):
        # STIM_ON and SHAM: use stim_duration from stim_params [ms -> s]
        # all other events: 0.0 s (instantaneous markers)
        has_stim_duration = 'stim_duration' in events.columns
        durations = []
        for _, row in events.iterrows():
            if row.trial_type in ('STIM_ON', 'SHAM') and has_stim_duration and pd.notna(row.stim_duration):
                durations.append(float(row.stim_duration) / 1000.0)
            else:
                durations.append(0.0)
        events['duration'] = durations
        return events

    # unpack stimulation parameters from list-of-dict into columns
    def unpack_stim_params(self, events):
        stim_params_rows = []
        for _, row in events.iterrows():
            sp = row.stim_params
            if isinstance(sp, list) and len(sp) > 0:
                stim_params_rows.append(sp[0])
            else:
                stim_params_rows.append({})
        stim_params_df = pd.DataFrame(stim_params_rows)
        return pd.concat([events.reset_index(drop=True), stim_params_df], axis=1)

    def make_events_descriptor(self):
        descriptions = {
            "STIM_ON": "Onset of electrical stimulation.",
            "STIM_OFF": "Offset of electrical stimulation.",
            "BEGIN_PS2": "Beginning of PS2 experiment.",
            "SESS_START": "Start of recording session.",
            "END_EXP": "End of experiment.",
            "AD_CHECK": "Attention check event.",
            "PAUSED": "Experiment paused.",
            "UNPAUSED": "Experiment unpaused.",
            "AMPLITUDE_CONFIRMED": "Stimulation amplitude confirmed by experimenter.",
            "SHAM": "Sham stimulation event (no actual stimulation delivered).",
        }
        HED = {
            "onset": {"Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file."},
            "duration": {"Description": "Duration (in seconds) of the event, measured from the onset of the event."},
            "sample": {"Description": "Onset of the event according to the sampling scheme (frequency)."},
            "trial_type": {"LongName": "Event category",
                           "Description": "Indicator of type of task action that occurs at the marked time",
                           "Levels": {k: descriptions[k] for k in self.events["trial_type"].unique() if k in descriptions}},
            'experiment': {'Description': 'The experimental paradigm completed.'},
            "session": {"Description": "The session number."},
            "subject": {"LongName": "Subject ID",
                        "Description": "The string identifier of the subject, e.g. R1001P."},
            "ad_observed": {"Description": "Indicator of whether the subject responded correctly to the attention check."},
            "eegfile": {"Description": "The name of the raw EEG data file associated with this event."},
            "exp_version": {"Description": "The version of the experimental paradigm software."},
            "is_stim": {"Description": "Indicator of whether electrical stimulation was applied during this event."},
            "montage": {"Description": "The name of the electrode montage used for recording."},
            "protocol": {"Description": "The name of the experimental protocol used."},
            "stim_params": {"Description": "A dictionary containing the stimulation parameters for this event."},
            "amplitude": {"Description": "Stimulation amplitude in μA."},
            "anode_label": {"Description": "Label of the anode stimulation contact."},
            "anode_number": {"Description": "Index of the anode stimulation contact."},
            "cathode_label": {"Description": "Label of the cathode stimulation contact."},
            "cathode_number": {"Description": "Index of the cathode stimulation contact."},
            "burst_freq": {"Description": "Frequency of stimulation bursts in Hz."},
            "n_bursts": {"Description": "Number of stimulation bursts delivered."},
            "n_pulses": {"Description": "Number of stimulation pulses delivered."},
            "pulse_freq": {"Description": "Frequency of stimulation pulses in Hz."},
            "pulse_width": {"Description": "Width of each stimulation pulse in μs."},
            "stim_duration": {"Description": "Duration of stimulation in ms."},
            "stimulation": {"Description": "Indicator of whether stimulation was delivered during this event."},
        }
        events_descriptor = {k: HED[k] for k in HED if k in self.events.columns}
        return events_descriptor

    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'Stimulation experiment: subjects are asked to sit quietly while stimulation parameters are varied. 4 amplitudes are used (up to a maximum amplitude set by the experimenter) and SHAM, 10Hz, 50Hz, 100Hz, and 200Hz stimulation. Duration can be set to a fixed duration (500 ms default).')
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar