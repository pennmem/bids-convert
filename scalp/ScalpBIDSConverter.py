import cmlreaders as cml
import mne
import numpy as np
import pandas as pd
import os
import json
from glob import glob
import shutil
import mne_bids
import cmlreaders as cml
import mne
import time

class UnknownElectrodeCapError(Exception):
    pass
class MultiplePathsError(FileExistsError):
    pass

class ScalpBIDSConverter:
    event_column_dict = {
        "ltpFR": ['subject', 'experiment', 'session', 'trial', 'task', 'item_name', 'item_num', 'recog_resp', 'recog_conf', 
                  'resp', 'answer', 'test_x', 'test_y', 'test_z', 'color_r', 'color_g', 'color_b', 'case', 'font'],
        "ltpFR2": ['subject', 'experiment', 'session', 'trial', 'item_name', 'item_num',
                   'list', 'answer', 'test_x', 'test_y', 'test_z'],
        "VFFR": ['subject', 'experiment', 'session', 'trial', 'item_name', 'item_num', 'too_fast'],
    #     "ValueCourier": ['subject', 'experiment', 'session', 'trial', 'task', 'item', 'itemno', 'itemvalue', 'actualvalue',
    #                      'compensation', 'intruded', 'intrusion', 'multiplier', 'numingroupchosen','playerrotY', 'presX', 
    #                      'presZ', 'primacybuf', 'recalled', 'recencybuf','serialpos', 'store','storeX', 'storeZ', 'storepointtype', 
    #                      'valuerecall'],
        "ValueCourier": [
            'actualvalue',
            'compensation',
            'eegfile',
            'eegoffset',
            'eogArtifact',
            'experiment',
            'intruded',
            'intrusion',
            'item',
            'itemno',
            'itemvalue',
            'montage',
            'msoffset',
            # 'mstime',
            'multiplier',
            'numingroupchosen',
            'phase',
            'playerrotY',
            'presX',
            'presZ',
            'primacybuf',
            'protocol',
            'recalled',
            'rectime',
            'recencybuf',
            'serialpos',
            'session',
            'store',
            'storeX',
            'storeZ',
            'storepointtype',
            'subject',
            'task',
            'trial',
            'valuerecall',
        ]
    }
    # eegoffset,correctPointingDirection, eegfile, eogArtifact, finalrecalled, montage, msoffset, mstime, phase, protocol, submittedPointingDirection, type
    def __init__(self, subject, experiment, session, root="/scratch/PEERS_BIDS/",
                 overwrite_eeg=True, overwrite_beh=True):
        self.root = root
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.load_subject_info()
        self.set_wordpool()
        self.events = self.load_events(beh_only=True)
        self.make_event_descriptors()
        self.write_bids_beh(overwrite=overwrite_beh)
        self.raw_filepath = self.locate_raw_file()
        if self.raw_filepath.endswith(".bz2"):
            self.unzip_raw_files()
        self.file_type = os.path.splitext(self.raw_filepath)[1]
        self.raw_file = self.load_scalp_eeg()
        self.set_montage()
        self.events = self.load_events()
        self.write_bids_eeg(temp_path=f"/home1/jrudoler/.temp/{int(time.time()*100)}_temp.edf",
                            overwrite=overwrite_eeg)
    
    def locate_raw_file(self):
        # hacky way to find all matching files!
        raw_file = glob(f"/data/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.raw*") + \
                glob(f"/data/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.bdf*") + \
                glob(f"/data/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.mff*")
        if len(raw_file)==0:
            raise FileNotFoundError
        elif len(raw_file)>1:
            raw_file = [os.path.realpath(p) for p in raw_file]
            print(np.unique(raw_file))
            if len(np.unique(raw_file))>1:
                if np.unique(raw_file)[0].endswith(".raw"):
                    print(np.unique(raw_file)[0])
                    raw_file = np.unique(raw_file)[0]
                else:
                    raise MultiplePathsError(f"{raw_file}")
            else:
                raw_file = raw_file[0]
        else:
            raw_file = raw_file[0]
            print(f"Raw File Found:", raw_file)
        return raw_file
    
    def load_scalp_eeg(self):
        if self.file_type == ".bdf":
            raw = mne.io.read_raw_bdf(self.raw_filepath, stim_channel='Status', preload=False)
        elif self.file_type in (".raw", ".mff"):
            raw = mne.io.read_raw_egi(self.raw_filepath, preload=False)
        else:
            raise ValueError("Unknown File Extension:", self.file_type)
        self.sfreq = raw.info['sfreq']
        self.recording_start = raw.info['meas_date']
        return raw
    
    def unzip_raw_files(self):
        output_path = os.path.splitext(self.raw_filepath.replace(' ', '_'))[0]
        if not os.path.exists(output_path):
            success = os.system(f"bunzip2 -k '{self.raw_filepath}'") == 0
            if success:
                shutil.move(self.raw_filepath, output_path)
                self.raw_filepath = output_path
    
    def set_montage(self):
        self.eeg_sidecar = {"PowerLineFrequency":60.0}
        if self.file_type == ".bdf":
            self.raw_file.set_channel_types({'EXG1':'eog', 'EXG2':'eog', 'EXG3':'eog', 'EXG4':'eog',
                                             'EXG5':'misc', 'EXG6':'misc', 'EXG7':'misc', 'EXG8':'misc'})
            self.raw_file.set_montage('biosemi128', on_missing="warn")
            self.eeg_sidecar["Manufacturer"] = "BioSemi"
            self.eeg_sidecar["CapManufacturer"] = "BioSemi"
        elif self.file_type in (".raw", ".mff", ".edf"):
            self.eeg_sidecar["Manufacturer"] = "EGI"
            self.eeg_sidecar["CapManufacturer"] = "EGI"
            self.eeg_sidecar["EEGReference"] = "Cz"
            self.raw_file.rename_channels({'E129': 'Cz'})
            if "sync" in self.raw_file.ch_names:
                # GSN 200 v2.1 caps
                self.eeg_sidecar["CapManufacturersModelName"] = "Geodisic Sensor Net 200 v2.1"
                mon = mne.channels.read_custom_montage("montage_files/egi128_GSN_200.sfp")
                self.raw_file.set_montage(mon, on_missing="warn")
                self.raw_file.set_channel_types({'E8': 'eog', 'E26': 'eog', 'E126': 'eog', 
                                            'E127': 'eog'})
                ## peripheral electrodes tend to flip up during the session, and get poor signal
                ## peripheral [127 126 17 128 125 120 44 49 56 63 69 74 82 89 95 100 108 114]
            else:
#             elif "DI15" in self.raw_file.ch_names:
                # GSN HydroCel caps
                self.eeg_sidecar["CapManufacturersModelName"] = "HydroCel Geodisic Sensor Net"
                mon = mne.channels.read_custom_montage("montage_files/egi128_GSN_HydroCel.sfp")
                self.raw_file.set_montage(mon, on_missing="warn")
                self.raw_file.set_channel_types({'E8': 'eog', 'E25': 'eog', 'E126': 'eog',
                           'E127': 'eog'})
                ## peripheral [127 126]
#             else:
#                 raise UnknownElectrodeCapError
        else:
            raise UnknownElectrodeCapError
        self.montage = self.raw_file.get_montage()
            
    def set_wordpool(self):
        if self.experiment=='ltpFR':
            if self.subject <= 'LTP159':
                self.wordpool_file = "wordpools/wasnorm_wordpool.txt"
            else:
                self.wordpool_file = "wordpools/wasnorm_wordpool_less_exclusions.txt"
        elif np.isin(self.experiment, ["ltpFR2", "VFFR"]):
            self.wordpool_file = "wordpools/wasnorm_wordpool_576.txt"
        elif np.isin(self.experiment, ["ValueCourier"]):
            self.wordpool_file = "wordpools/valuecourier_wordpool.txt"
        else:
            raise Exception("Wordpool not known for this experiment.")
    
    def load_events(self, beh_only=False):
        reader = cml.CMLReader(self.subject, self.experiment, self.session)
        events = reader.load('events')
        events = events.rename(columns={"eegoffset":"sample", "type":"trial_type"})
        ## math distractor
        if "test" in events.columns:
            events[["test_x", "test_y", "test_z"]] = events['test'].apply(pd.Series)
            events = events.drop(columns=["test"])
        if "font" in events.columns:
            events["font"] = events['font'].apply(os.path.basename)
        if beh_only:
            standard_cols = ["mstime", "trial_type", 'stim_file']
            events["mstime"] = events["mstime"] - events["mstime"].iloc[0]
        else:
            events['onset'] = events['sample'] / self.sfreq
            events['duration'] = "n/a"
            standard_cols = ['onset', 'duration', "trial_type", "sample", 'stim_file']
        events['stim_file'] = np.where(events.trial_type.str.contains("WORD"), self.wordpool_file, "n/a")
        events = events.fillna("n/a")
        events = events.replace("", "n/a")
        events = events.replace("-999", "n/a")
        events = events.replace(-999, "n/a")
        cols_to_include = ScalpBIDSConverter.event_column_dict[self.experiment]
        cols_to_include = [col for col in cols_to_include if col in events.columns]
        events = events[standard_cols + cols_to_include]
        return events
    
    def make_event_descriptors(self):
        descriptions = {
            "SESS_START": "Beginning of session.",
            "SESS_END": "End of session.",
            "WORD": "Word presentation onset.",
            "WORD_ON": "Word presentation onset.",
            "WORD_OFF": "Word presentation offset.",
            "REC_START": "Recall phase begins.",
            "REC_END": "Recall phase ends.",
            "REC_STOP": "Recall phase ends.",
            "REST_REWET": "Mid-session break to rewet scalp cap.",
            "REC_WORD": "Recalled word, onset of speech (during free recall).",
            "REC_WORD_VV": "Vocalization (during free recall).",
            "FFR_REC_WORD": "Recalled word, onset of speech (during final free recall).",
            "FFR_REC_WORD_VV": "Vocalization (during final free recall).",
            "RECOG_CONF": "Confidence judgement for recognition.",
            "KEY_MSG": "Warning message telling the subject they pressed one of the keys corresponding to the wrong judgment.",
            "RECOG_LURE": "Recognition item that is a lure.",
            "RECOG_RESP": "Recognition response ('pess', 'po').",
            "RECOG_RESP_VV": "Vocalization (during recognition).",
            "RECOG_TARGET": "Recognition item that is a target.",
            "SLOW_MSG": "Warning message telling the subject that they took too long to make their judgment about a word.",
            "START": "Beginning of math distractor phase.",
            "STOP": "End of math distractor phase.",
            "PROB": "Math problem presentation onset.",
            "PRACTICE_WORD": "Word presentation onset (in a practice list)",
            "PRACTICE_WORD_OFF": "Word presentation offset (in a practice list)",
            "FFR_START": "Beginning of final free recall phase.",
            "FFR_END": "End of final free recall phase.",
            "FFR_STOP": "End of final free recall phase.",
            "DISTRACTOR": "Beginning of math distractoor phase.",
            'PRACTICE_REC_START': "Recall phase begins (in a practice list).", 
            'PRACTICE_REC_STOP': "Recall phase begins (in a practice list).",
            'PRACTICE_REC_WORD': "Recalled word, onset of speech (during practice list free recall).", 
            'PRACTICE_REC_WORD_VV': "Vocalization (during practice list free recall).", 
            "COUNTDOWN": "Initiate countdown before encoding.",
            "BREAK_START": "Start mid-session break to rewet scalp cap.",
            "BREAK_STOP": "Stop mid-session break to rewet scalp cap.",
                # ---------------- ValueCourier-specific trial types ----------------
            "store mappings": "Display or logging of store-to-location/value mappings in the task.",
            "VIDEO_START": "Start of instructional or task-related video.",
            "VIDEO_STOP": "End of instructional or task-related video.",
            "TL_START": "Start of a temporal-learning or timeline block.",
            "TL_END": "End of a temporal-learning or timeline block.",
            "PRACTICE_DELIVERY_START": "Start of a practice delivery trial in the task.",
            "PRACTICE_DELIVERY_END": "End of a practice delivery trial in the task.",
            "PRACTICE_VALUE_RECALL": "Value recall response during a practice trial.",
            "POINTER_ON": "Onset of the pointing/selection cursor during navigation or choice.",
            "TRIAL_START": "Start of a trial.",
            "TRIAL_END": "End of a trial.",
            "VALUE_RECALL": "Subject’s value recall response for the current list or item.",
            "FINAL_COMPENSATION": "Event marking display and computation of final monetary compensation.",
        }
        HED = {
            "onset": {
                "Description": "Onset (in seconds) of the event, measured from the beginning of the acquisition of the first data point stored in the corresponding task data file. ",
            },
            "subject": {
                "LongName": "Subject ID",
                "Description": "The string identifier of the subject, e.g. LTP123",

            },
            "session": {
                "Description": "The session number (1 - 24)."
            },
            "trial": {
                "LongName": "Trial Number",
                "Description": "Word list (1-24) during which the event occurred. Trial <= 0 indicates practice list.",
            },
            "trial_type": {
                "LongName": "Event category",
                "Description": "Indicator of type of task action that occurs at the marked time",
                "Levels": {k:descriptions[k] for k in self.events["trial_type"].unique()},
            },
            "item_name": {
                "Description": "The word being presented or recalled in a WORD or REC_WORD event."
            },
            "item_num": {
                "LongName": "Item number",
                "Description": "The ID number of the presented or recalled word in the word pool. -1 represents an intrusion or vocalization."
            },
            'task': {
                "LongName": "Task type",
                "Description": "Type of judgment made on a presented word",
                "Levels": {
                    -1: "Control, no task, just read the word",
                    0: "Size",
                    1: "Animacy"
                }
            },
            'recog_conf': {
                "LongName": "Confidence rating",
                "Description": "Confidence rating of recognition response, 1 (very low confidence) - 5 (complete confidence)"
            }, 
            'recog_resp': {
                "LongName": "Recognition response",
                "Description": "1 (old) or 0 (new) response in the recognition period.",
                "Levels": {0: "new", 1: "old"}
            },
            'resp': {
                "LongName": "Judgement response",
                "Description": "Judgement response during the study period for size/animacy. Responses >2 are wrong key presses.",
                "Levels": {-1: "control", 0: "small/nonliving", 1: "big/living"}
            },
            'answer': {
                "LongName": "Math problem response",
                "Description": "Answer to problem with form X + Y + Z = ?",
            },
            'test_x': {
                "LongName": "Math problem X",
                "Description": "X component of problem with form X + Y + Z = ?",
            },
            'test_y': {
                "LongName": "Math problem Y",
                "Description": "Y component of problem with form X + Y + Z = ?",
            },
            'test_z': {
                "LongName": "Math problem Z",
                "Description": "Z component of problem with form X + Y + Z = ?",
            }, 
            'color_b': {
                "LongName": "Blue",
                "Description": "Blue RGB value in [0, 255]",
            },
            'color_g': {
                "LongName": "Green",
                "Description": "Green RGB value in [0, 255]",
            },
            'color_r': {
                "LongName": "Red",
                "Description": "Red RGB value in [0, 255]",
            },
            'case': {
                "LongName": "Letter case",
                "Description": "Case of text presented on screen",
                "Levels": {'upper': 'upper case', 'lower': 'lower case'},
            },
            'font': {
                "LongName": "Word font",
                "Description": "File name for font of text presented on screen, found in stimuli/fonts.",
            },
            'too_fast':{
                "LongName": "'Too fast' message displayed",
                "Description": "Subject recalled word too quickly, warning displayed on screen."
            },
            
            # ---- ValueCourier-specific fields ----
            'item': {
                "LongName": "Item string",
                "Description": "The item identifier (often the string name of the object/value being judged)."
            },
            'itemno': {
                "LongName": "Item number",
                "Description": "Numeric ID for the item within the ValueCourier item set."
            },
            'itemvalue': {
                "LongName": "Displayed item value",
                "Description": "Point or monetary value displayed for the item on that trial."
            },
            'actualvalue': {
                "LongName": "True average tip value",
                "Description": "True average tip value of items in a delivery day"
            },
            'compensation': {
                "LongName": "Compensation",
                "Description": "Compensation to participant at the end of the session."
            },
            'intruded': {
                "LongName": "Intruded flag",
                "Description": "1 if the response was an intrusion relative to the current list/context, 0 otherwise."
            },
            'intrusion': {
                "LongName": "Intrusion item number",
                "Description": "Identifier of the intruded item when an intrusion occurs, -1 otherwise."
            },
            'multiplier': {
                "LongName": "Value multiplier",
                "Description": "Multiplicative factor applied to the max tip value (10$) which is used to calculate compensation."
            },
            'numingroupchosen': {
                "LongName": "Number in group chosen",
                "Description": "In temporal value association condition, the list of items is split in half and the each half has a high or low value. Number in group chosen describes the number of high items are in the high side of the list and vice versa for the low side."
            },
            'playerrotY': {
                "LongName": "Player rotation (Y axis)",
                "Description": "Rotation angle of the player avatar around the Y axis in the virtual environment."
            },
            'presX': {
                "LongName": "Presentation X coordinate",
                "Description": "X coordinate of the item at presentation in the virtual environment."
            },
            'presZ': {
                "LongName": "Presentation Z coordinate",
                "Description": "Z coordinate of the item at presentation in the virtual environment."
            },
            'primacybuf': {
                "LongName": "Primacy buffer length",
                "Description": "Length of the buffer in the beginning of the item list during the temporal association condition which is not split into halves."
            },
            'recencybuf': {
                "LongName": "Recency buffer length",
                "Description": "Length of the buffer at the end of the item list during the temporal association condition which is not split into halves."
            },
            'recalled': {
                "LongName": "Recalled flag",
                "Description": "1 if the item was successfully recalled, 0 otherwise."
            },
            'serialpos': {
                "LongName": "Serial position",
                "Description": "Serial position of the item within the list."
            },
            'store': {
                "LongName": "Store identifier",
                "Description": "Identifier of the store/location in the ValueCourier environment where the item is presented."
            },
            'storeX': {
                "LongName": "Store X coordinate",
                "Description": "X coordinate of the store in the virtual environment."
            },
            'storeZ': {
                "LongName": "Store Z coordinate",
                "Description": "Z coordinate of the store in the virtual environment."
            },
            'storepointtype': {
                "LongName": "Store point type",
                "Description": "Type of point or reward structure associated with the store."
            },
            'valuerecall': {
                "LongName": "Value recall response",
                "Description": "Subject’s recalled value or judgment about the average list value."
            },
            "msoffset": {
                "LongName": "Event offset (ms)",
                "Description": "Event time offset in milliseconds relative to a task-specific reference (typically the start of the trial or session).",
                "Units": "ms",
            },
            "eegoffset": {
                "LongName": "EEG sample offset",
                "Description": "Sample index of the event relative to the start of the EEG recording. For convenience, this is often duplicated in the 'sample' column.",
            },
            "rectime": {
                "LongName": "Recall Time",
                "Description": "Time when item was recalled.",
            },
            "eegfile": {
                "LongName": "EEG file",
                "Description": "Filename of the original EEG recording from which this BIDS dataset was derived.",
            },
            "eogArtifact": {
                "LongName": "EOG artifact flag",
                "Description": "Indicator of whether the event or trial was contaminated by EOG artifact (1 = artifact, 0 = clean, or 'n/a' if not assessed).",
            },
            "montage": {
                "LongName": "EEG montage name",
                "Description": "Name or identifier of the electrode montage or sensor net used for the recording (e.g., BioSemi 128, GSN-HydroCel-128).",
            },
            "phase": {
                "LongName": "Experiment phase",
                "Description": "Experiment phase during which the event occurred (e.g., ENCODING, RETRIEVAL, DISTRACTOR, PRACTICE, INSTRUCTION).",
            },
            "protocol": {
                "LongName": "CML protocol name",
                "Description": "High-level protocol identifier from the CML system (e.g., 'ltp', 'ltpFR', 'ltpFR2', 'ValueCourier').",
            },
        }
        self.events_descriptor = {k:HED[k] for k in HED if k in self.events.columns}
        
    
    def load_subject_info(self):
        #TODO
        pass
    
    def write_bids_beh(self, overwrite=True):
        task_name = self.experiment.lower() 
        # events = self.load_events(beh_only=True)
        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                          session=str(self.session),
                                          task=task_name,
                                          datatype="beh",
                                          suffix="beh",
                                          extension=".tsv",
                                          root=self.root)
        os.makedirs(bids_path.directory, exist_ok=True)
        self.events.to_csv(bids_path.fpath, sep="\t", index=False)
        with open(bids_path.update(suffix="beh", extension=".json").fpath, "w") as f:
            json.dump(fp=f, obj = self.events_desaacriptor)
    
    def write_bids_eeg(self, temp_path="temp.edf", overwrite=True):
        task_name = self.experiment.lower() 
        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                          session=str(self.session),
                                          task=task_name,
                                          datatype="eeg",
                                          root=self.root)
        try:
            if self.file_type != ".bdf":
                try:
                    mne.export.export_raw(temp_path, self.raw_file, add_ch_type=True)
                    print("temp file created")
                except FileExistsError as e:
                    print(e)
                edf_file = mne.io.read_raw_edf(temp_path, preload=False, infer_types=True)
                edf_file.set_montage(self.montage)
                mne_bids.write_raw_bids(
                    edf_file,
                    events_data=None,
                    bids_path=bids_path,
                    overwrite=overwrite
                )
                os.system(f"rm {temp_path}")
                print("temp file removed")
            else:
                mne_bids.write_raw_bids(
                    self.raw_file,
                    events_data=None,
                    bids_path=bids_path,
                    overwrite=overwrite
                )
        except FileExistsError as e:
            print(e)
        self.events.to_csv(os.path.join(bids_path.directory, bids_path.basename+"_events.tsv"),
                           sep="\t", index=False)
        with open(bids_path.update(suffix="events", extension=".json").fpath, "w") as f:
            json.dump(fp=f, obj = self.events_descriptor)
        mne_bids.update_sidecar_json(bids_path.update(suffix="eeg", extension=".json"),
                                     self.eeg_sidecar, verbose=True)
