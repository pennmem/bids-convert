# imports
import importlib.util
import os
import pandas as pd
import numpy as np

# PS2.1 folder contains a dot so it cannot be imported via standard importlib;
# load it directly from its file path.
_ps21_path = os.path.join(os.path.dirname(__file__), '..', 'PS2.1', 'PS2.1_BIDS_converter.py')
_ps21_spec = importlib.util.spec_from_file_location('PS21_BIDS_converter', _ps21_path)
_ps21_module = importlib.util.module_from_spec(_ps21_spec)
_ps21_spec.loader.exec_module(_ps21_module)
PS21_BIDS_converter = _ps21_module.PS21_BIDS_converter


class PS2_BIDS_converter(PS21_BIDS_converter):
    # initialize
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions,
                 root='/scratch/hherrema/BIDS/PS2/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def apply_event_durations(self, events):
        # STIM_ON and STIM_SINGLE_PULSE: use stim_duration from stim_params [ms -> s]
        # STIM_SINGLE_PULSE carries stim_duration=1 ms -> 0.001 s
        # SHAM events are not present in PS2 (introduced in PS2.1)
        # all other events: 0.0 s (instantaneous markers)
        has_stim_duration = 'stim_duration' in events.columns
        durations = []
        for _, row in events.iterrows():
            if row.trial_type in ('STIM_ON', 'STIM_SINGLE_PULSE') and has_stim_duration and pd.notna(row.stim_duration):
                durations.append(float(row.stim_duration) / 1000.0)
            else:
                durations.append(0.0)
        events['duration'] = durations
        return events

    def make_events_descriptor(self):
        descriptions = {
            "STIM_ON": "Onset of electrical stimulation (pulse train).",
            "STIM_OFF": "Offset of electrical stimulation.",
            "STIM_SINGLE_PULSE": "Single pulse electrical stimulation event (n_pulses=1, n_bursts=1, stim_duration=1 ms).",
            "BEGIN_PS2": "Beginning of PS2 experiment.",
            "SESS_START": "Start of recording session.",
            "END_EXP": "End of experiment.",
            "AD_CHECK": "Attention check event.",
            "PAUSED": "Experiment paused.",
            "UNPAUSED": "Experiment unpaused.",
            "AMPLITUDE_CONFIRMED": "Stimulation amplitude confirmed by experimenter.",
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
            "burst_freq": {"Description": "Frequency of stimulation bursts in Hz. -1 indicates not applicable for STIM_SINGLE_PULSE events."},
            "n_bursts": {"Description": "Number of stimulation bursts delivered. -1 indicates not applicable for STIM_SINGLE_PULSE events."},
            "n_pulses": {"Description": "Number of stimulation pulses delivered."},
            "pulse_freq": {"Description": "Frequency of stimulation pulses in Hz. -1 indicates not applicable for STIM_SINGLE_PULSE events."},
            "pulse_width": {"Description": "Width of each stimulation pulse in μs."},
            "stim_duration": {"Description": "Duration of stimulation in ms. For STIM_SINGLE_PULSE events this is 1 ms."},
            "stimulation": {"Description": "Indicator of whether stimulation was delivered during this event."},
        }
        events_descriptor = {k: HED[k] for k in HED if k in self.events.columns}
        return events_descriptor

    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super(PS21_BIDS_converter, self).eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'Stimulation experiment: subjects are asked to sit quietly while stimulation parameters are varied. Stimulation parameters (amplitude, frequency, duration) are swept across conditions. Single-pulse stimulation events (STIM_SINGLE_PULSE) are delivered in addition to standard pulse trains.')
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = True
        return sidecar