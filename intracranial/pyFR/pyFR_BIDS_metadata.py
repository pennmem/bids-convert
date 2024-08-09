# imports
import cmlreaders as cml
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from ..intracranial_BIDS_metadata import intracranial_BIDS_metadata

# class to use when doing metadata checks before converting pyFR to BIDS format
class pyFR_BIDS_metadata(intracranial_BIDS_metadata):

    def __init__(self, experiment='pyFR', root='/home1/hherrema/BIDS/'):
        super().__init__(experiment, root)

    # try to load events, math events, contacts, pairs, monopolar and bipolar EEG
    # determine system_version, unit_scale, mni, tal, max_label_len, eegfiles
    def metadata(self):
        metadata_df = pd.DataFrame(columns=['subject', 'experiment', 'session', 'montage', 'events', 'math_events', 'contacts', 'pairs',
                                        'max_label_len', 'system_version', 'unit_scale', 'monopolar', 'bipolar', 'mni', 'tal', 'eegfiles'])
    
        for _, row in tqdm(self.df_select.iterrows()):
            reader = cml.CMLReader(subject=row.subject, experiment=row.experiment, session=row.session,
                                localization=row.localization, montage=row.montage)
            
            # load in behavioral events
            events_bool = self._load_events(reader)

            # load in math events
            math_bool = self._load_math(reader)

            # load in contacts, tal coordinates
            contacts_bool, tal = self._load_contacts(reader)
            
            # check for mni coordinates
            mni = self._mni_coords(reader)

            # load in pairs
            pairs_bool, max_label_len = self._load_pairs(reader)

            # number of EEG files
            eegfiles = self._n_eegfiles(reader, events_bool, math_bool)

            # load in monopolar EEG
            if contacts_bool:
                monopolar = self._load_eeg(reader, 'monopolar')
            else:
                monopolar = False

            # load in bipolar EEG
            if pairs_bool:
                bipolar = self._load_eeg(reader, 'bipolar')
            else:
                bipolar = False
                
            # determine system version and unit scale
            system_version, unit_scale = self._sysv_units(row)
            
            metadata_df = pd.concat([metadata_df,
                                    pd.DataFrame({'subject':row.subject, 'experiment':row.experiment, 'session':row.session, 'montage':row.montage,
                                                'events':events_bool, 'math_events':math_bool, 'contacts':contacts_bool, 'pairs':pairs_bool,
                                                'max_label_len':max_label_len, 'system_version':system_version, 'unit_scale':unit_scale,
                                                'monopolar':monopolar, 'bipolar':bipolar, 'mni':mni, 'tal':tal, 'eegfiles':eegfiles},
                                                index = [len(metadata_df.index)])])
            
        return metadata_df
    
    def _load_math(self, reader):
        try:
            if reader.montage != 0:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{reader.subject}_{reader.montage}_math.mat', squeeze_me=True)['events'])
            else:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{reader.subject}_math.mat', squeeze_me=True)['events'])

            math_evs = math_evs[math_evs.session == reader.session]                                        # select out session
            return len(math_evs) > 0
        except BaseException as e:
            return False

    def _load_contacts(self, reader):
        tal = False
        try:
            contacts = reader.load('contacts')
            if set(['x', 'y', 'z']).issubset(set(contacts.columns)):
                tal = True
                
            return True, tal
        except BaseException as e:
            return False, False
        
    def _mni_coords(self, reader):
        try:
            if reader.montage != 0:
                mni_coords = np.loadtxt(f'/data/eeg/{reader.subject}_{reader.montage}/tal/RAW_coords.txt.mni')
            else:
                mni_coords = np.loadtxt(f'/data/eeg/{reader.subject}/tal/RAW_coords.txt.mni')
                
            if mni_coords.shape[0] > 0:
                return True
            else:
                return False
        except BaseException as e:
            return False
        
    def _n_eegfiles(self, reader, events_bool, math_bool):
        if events_bool and math_bool:
            events = reader.load('events')
            if reader.montage != 0:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{reader.subject}_{reader.montage}_math.mat', squeeze_me=True)['events'])
            else:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{reader.subject}_math.mat', squeeze_me=True)['events'])
                
            math_evs = math_evs[math_evs.session == reader.session]
            
            all_evs = pd.concat([events, math_evs], ignore_index=True)
            try:
                eegfiles = [x for x in all_evs['eegfile'].unique() if x != '']
                return len(eegfiles)
            except BaseException as e:
                return -1
        elif events_bool:
            events = reader.load('events')
            try:
                eegfiles = [x for x in events['eegfile'].unique() if x != '']
                return len(eegfiles)
            except BaseException as e:
                return -1
        elif math_bool:
            if reader.montage != 0:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{reader.subject}_{reader.montage}_math.mat', squeeze_me=True)['events'])
            else:
                math_evs = pd.DataFrame(scipy.io.loadmat(f'/data/events/pyFR/{reader.subject}_math.mat', squeeze_me=True)['events'])
                
            math_evs = math_evs[math_evs.session == reader.session]
            try:
                eegfiles = [x for x in math_evs['eegfile'].unique() if x != '']
                return len(eegfiles)
            except BaseException as e:
                return -1
        else:
            return -1
        
    def _sysv_units(self, row):
        system_version = 1.0  # pyFR
        unit_scale = self._determine_unit_scale(row)
        return system_version, unit_scale
        
    def _determine_unit_scale(self, row):
        sys1_units = pd.read_csv('system_1_unit_conversions.csv')
        return sys1_units[(sys1_units.subject == row.subject) &
                        (sys1_units.experiment == row.experiment) & 
                        (sys1_units.session == row.session)].iloc[0]['conversion_to_V']
