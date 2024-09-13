# imports
import cmlreaders as cml
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..intracranial_BIDS_converter import intracranial_BIDS_converter

class RepFR1_BIDS_converter(intracranial_BIDS_converter):
    wordpool_EN = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/RepFR1/wordpools/wordpool_EN.txt', dtype=str)
    wordpool_SP = np.loadtxt('/home1/hherrema/BIDS/bids-convert/intracranial/RepFR1/wordpools/wordpool_SP.txt', dtype=str)
    
    def __init__(self, subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root='/scratch/hherrema/BIDS/RepFR1/'):
        super().__init__(subject, experiment, session, system_version, unit_scale, monopolar, bipolar, mni, tal, area, brain_regions, root)

    # ---------- Events ----------
    def set_wordpool(self):
        evs = self.reader.load('events')
        word_evs = evs[evs['type']=='WORD']
        if np.all([x in self.wordpool_EN for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_EN.txt'
        elif np.all([x in self.wordpool_SP for x in word_evs.item_name]):
            wordpool_file = 'wordpools/wordpool_SP.txt'
        else:
            wordpool_file = 'n/a'

        return wordpool_file
    
    def events_to_BIDS(self):
        events = self.reader.load('events')
        events['type'] = events['type'].replace('session_start', 'SESS_START')                       # fix odd SESS_START event types
        events.loc[events['type'] == 'SESS_END', 'list'] = -999                                      # some SESS_END events get final list +1, just set all to default
        
        events = events.rename(columns={'eegoffset':'sample', 'type':'trial_type'})                  # rename columns
        events['onset'] = (events.mstime - events.mstime.iloc[0]) / 1000.0                           # onset from first event [s]
        events['duration'] = np.concatenate((np.diff(events.mstime), np.array([0]))) / 1000.0        # event duration [s]
        events['duration'] = events['duration'].mask(events['duration'] < 0.0, 0.0)                  # replace events with negative duration with 0.0 s
        events = self.apply_event_durations(events)                                                  # apply well-defined durations [s]
        events['response_time'] = 'n/a'                                                              # response time [s]
        events.loc[(events.trial_type=='REC_WORD') | (events.trial_type=='REC_WORD_VV'),
                   'response_time'] = events['rectime'] / 1000.0
        events['stim_file'] = np.where(events.trial_type=='WORD', self.wordpool_file, 'n/a')         # add wordpool to word events
        
        events = events[['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'item_name',
                         'serialpos', 'repeats', 'list', 'experiment', 'session', 'subject']]        # select and re-order columns
        events = events.fillna('n/a')
        events = events.replace('', 'n/a')

        return events
        

    def apply_event_durations(self, events):
        # word durations
        wd_path = '/home1/hherrema/BIDS/RepFR1/metadata/word_durations.csv'             # update to a shared location
        word_durations = pd.read_csv(wd_path)
        wd = word_durations[(word_durations.subject == self.subject) & (word_durations.session == self.session)].iloc[0].word_duration_rounded
        wd /= 1000                               # convert from ms to s

        durations = []
        for _, row in events.iterrows():
            # word presentation events
            if row.trial_type == 'WORD':
                durations.append(wd)

            # countdown events = 3000 ms
            elif row.trial_type == 'COUNTDOWN':
                durations.append(3.0)

            # keep current duration
            else:
                durations.append(row.duration)

        events['duration'] = durations           # preserves column order
        return events


    def make_events_descriptor(self):
        descriptions = {
            "SESS_START": "Beginning of session.",
            "SESS_END": "End of session.",
            "START": "Beginning of session.",
            "TRIAL_START": "Denotes start of new trial (list).",
            "COUNTDOWN": "Beginning of pre-list presentation countdown.",
            "WORD": "Word presentation onset.",
            "WORD_OFF": "Word presentation offset.",
            "REC_START": "Start of recall phase.",
            "REC_END": "End of recall phase.",
            "REC_WORD": "Recalled word, onset of speech (during free recall).", 
            "REC_WORD_VV": "Vocalization (during free recall)."
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
                     "Description": "Word list (1-20) during which the event occurred. Trial = 0 indicates practice list."},
            "item_name": {"Description": "The word being presented or recalled in a WORD or REC_WORD event."},
            'serialpos': {'LongName': 'Serial Position', 
                          'Description': 'The order position of a word presented in an WORD event.'},
            "repeats": {"Description": "Number of repetitions within the list of a word presented in a WORD event."}
        }
        events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        return events_descriptor
    
    
    # ---------- EEG ----------
    def eeg_sidecar(self, ref):
        sidecar = super().eeg_sidecar(ref)
        sidecar = pd.DataFrame(sidecar, index=[0])
        sidecar.insert(1, 'TaskDescription', 'free recall of word lists with repeated items')     # place in second column
        sidecar = sidecar.to_dict(orient='records')[0]
        sidecar['ElectricalStimulation'] = False
        return sidecar
