import cmlreaders as cml
import mne
import numpy as np
import pandas as pd
import os
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
        "ltpFR": ['subject', 'experiment', 'session', 'trial', 'task', 'item_name', 'item_num', 'recog_conf', 'resp', 
                  'answer', 'test_x', 'test_y', 'test_z', 'color_b', 'color_g', 'color_r', 'case', 'font'],
        "ltpFR2": ['subject', 'experiment', 'session', 'trial', 'item_name', 'item_num',
                   'list', 'answer', 'test_x', 'test_y', 'test_z'],
        "VFFR": ['subject', 'experiment', 'session', 'trial', 'item_name', 'item_num', 'too_fast']
    }
    def __init__(self, subject, experiment, session, root="/scratch/PEERS_BIDS/", overwrite=True):
        self.root = root
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.load_subject_info()
        self.raw_filepath = self.locate_raw_file()
        if self.raw_filepath.endswith(".bz2"):
            self.unzip_raw_files()
        self.file_type = os.path.splitext(self.raw_filepath)[1]
        self.raw_file = self.load_scalp_eeg()
        self.set_wordpool()
        self.events = self.load_events()
        self.write_bids(temp_path=f"/scratch/jrudoler/{int(time.time()*100)}_temp.edf", overwrite=overwrite)
    
    def locate_raw_file(self):
        # hacky way to find all matching files!
#         raw_file = glob(f"/data[0-9]/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.raw*") + \
#                 glob(f"/data1[0-2]/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.raw*") + \
#                 glob(f"/data[0-9]/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.bdf*") + \
#                 glob(f"/data1[0-2]/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.bdf*") + \
#                 glob(f"/data[0-9]/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.mff*") + \
#                 glob(f"/data1[0-2]/eeg/scalp/ltp/{self.experiment}/{self.subject}/session_{self.session}/eeg/*.mff*")
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
        if self.file_type == ".bdf":
            self.raw_file = self.raw_file.set_montage('biosemi128')
            self.raw_file.set_channel_types({'EXG1':'eog', 'EXG2':'eog', 'EXG3':'eog', 'EXG4':'eog',
                                             'EXG5':'misc', 'EXG6':'misc', 'EXG7':'misc', 'EXG8':'misc'})
        
        elif self.file_type in (".raw", ".mff"):
            self.raw_file.rename_channels({'E129': 'Cz'})
            if "cal+" in self.raw_file.ch_names:
                # GSN HydroCel caps
                self.raw_file.set_montage("montage_files/egi128_GSN_HydroCel.sfp")
                self.raw_file.set_channel_types({'E8': 'eog', 'E25': 'eog', 'E126': 'eog',
                           'E127': 'eog', 'Cz': 'misc'})
                ## peripheral [127 126]
            elif "sync" in self.raw_file.ch_names:
                # GSN 200 v2.1 caps
                self.raw_file.set_montage("montage_files/egi128_GSN_200.sfp")
                self.raw_file.set_channel_types({'E8': 'eog', 'E26': 'eog', 'E126': 'eog', 
                                            'E127': 'eog', 'Cz': 'misc'})
                ## peripheral [127 126 17 128 125 120 44 49 56 63 69 74 82 89 95 100 108 114]
            else:
                raise UnknownElectrodeCapError
        else:
            raise UnknownElectrodeCapError
            
    def set_wordpool(self):
        if self.experiment=='ltpFR':
            if self.subject <= 'LTP159':
                self.wordpool_file = "wordpools/wasnorm_wordpool.txt"
            else:
                self.wordpool_file = "wordpools/wasnorm_wordpool_less_exclusions.txt"
        elif np.isin(self.experiment, ["ltpFR2", "VFFR"]):
            self.wordpool_file = "wordpools/wasnorm_wordpool_576.txt"
        else:
            raise Exception("Wordpool not known for this experiment.")
    
    def load_events(self):
        reader = cml.CMLReader(self.subject, self.experiment, self.session)
        events = reader.load('events')
        events = events.rename(columns={"eegoffset":"sample", "type":"trial_type"})
        ## math distractor
        if "test" in events.columns:
            events[["test_x", "test_y", "test_z"]] = events['test'].apply(pd.Series)
            events = events.drop(columns=["test"])
        events['onset'] = events['sample'] / self.sfreq
        events['duration'] = "n/a"
        events['stim_file'] = np.where(events.trial_type.str.contains("WORD"), self.wordpool_file, "n/a")
        events = events.fillna("n/a")
        events = events.replace("", "n/a")
        events = events.replace("-999", "n/a")
        events = events.replace(-999, "n/a")
        standard_cols = ['onset', 'duration', "trial_type", "sample", 'stim_file']
        cols_to_include = ScalpBIDSConverter.event_column_dict[self.experiment]
        cols_to_include = [col for col in cols_to_include if col in events.columns]
        events = events[standard_cols + cols_to_include]
        return events
    
    def load_subject_info(self):
        pass
    
    def write_bids(self, temp_path="temp.edf", overwrite=True):
        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                          session=str(self.session),
                                          task=self.experiment,
                                          datatype="eeg",
                                          root=self.root)
        if self.file_type != ".bdf":
            try:
                mne.export.export_raw(temp_path, self.raw_file)
                print("temp file created")
            except FileExistsError as e:
                print(e)
            edf_file = mne.io.read_raw_edf(temp_path, preload=False)
            mne_bids.write_raw_bids(
                edf_file,
                events_data=None,
                montage=self.raw_file.get_montage(),
                bids_path=bids_path,
                overwrite=overwrite
            )
            os.system(f"rm {temp_path}")
            print("temp file removed")
        else:
            mne_bids.write_raw_bids(
                self.raw_file,
                events_data=None,
                montage=self.raw_file.get_montage(),
                bids_path=bids_path,
                overwrite=overwrite
            )
        self.events.to_csv(os.path.join(bids_path.directory, bids_path.basename+"_events.tsv"),
                           sep="\t", index=False)
