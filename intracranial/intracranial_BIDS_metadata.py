# imports
from types import NotImplementedType
import cmlreaders as cml
import pandas as pd
import numpy as np
import re
import json
import os
import mne_bids
from glob import glob
import pickle
from tqdm import tqdm
import math
import string

# class to use when doing metadata checks before converter experiment to BIDS format
class intracranial_BIDS_metadata:
    BRAIN_REGIONS = ['wb.region', 'ind.region', 'das.region', 'stein.region']

    def __init__(self, experiment, root='/home1/hherrema/BIDS/'):
        self.experiment = experiment
        self.root = root
        self.df_select = self.load_exp_cmlreaders()

    # get data index with all session from experiment on cmlreaders
    def load_exp_cmlreaders(self):
        df = cml.get_data_index()
        return df.query("experiment==@self.experiment")
    
    # try to load events, contacts, pairs, monopolar and bipolar EEG
    # determine system_version, unit_scale, mni, tal, area, eegfiles
    def metadata(self):
        metadata_df = pd.DataFrame(columns=['subject', 'experiment', 'session', 'system_version', 'unit_scale', 
                                            'monopolar', 'bipolar', 'mni', 'tal', 'area', 'wb.region', 'ind.region', 
                                            'das.region', 'stein.region', 'eegfiles'])
        for _, row in tqdm(self.df_select.iterrows()):
            reader = cml.CMLReader(subject=row.subject, experiment=row.experiment, session=row.session, 
                                   localization=row.localization, montage=row.montage)
            
            # values and toggles to set in metadata json file for each subject-session
            area = False

            # load in behavioral events
            events_bool = self._load_events(reader)

            # load in contacts, which checks for mni and tal coordinates
            # provide mni/tal if all contacts have coordinate values --> perhaps too strict a stipulation
            contacts_bool, mni, tal = self._load_contacts(reader)

            # load in pairs
            pairs_bool = self._load_pairs(reader)

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
            
            # determine system_version and unit_scale
            system_version, unit_scale = self._sysv_units(row)

            # determine area --> don't actually make the mapping here
            area = self._area_data(row)

            # create dictionary mapping brain regions to number of contacts with valid entries
            brain_regions = self._brain_regions(row)

            # determine number of EEG files
            eegfiles = self._n_eegfiles(reader)

            # append to dataframe of metadata
            metadata_df = pd.concat([metadata_df, 
                                     pd.DataFrame({'subject':row.subject, 'experiment':row.experiment, 'session':row.session, 
                                                   'events':events_bool, 'contacts':contacts_bool, 'pairs':pairs_bool, 
                                                   'system_version':system_version, 'unit_scale':unit_scale, 
                                                   'monopolar':monopolar, 'bipolar':bipolar, 'mni':mni, 'tal':tal, 'area':area, 
                                                   'wb.region':brain_regions['wb.region'], 'ind.region':brain_regions['ind.region'], 
                                                   'das.region':brain_regions['das.region'], 'stein.region':brain_regions['stein.region'], 
                                                   'eegfiles':eegfiles}, index=[len(metadata_df.index)])])
            
        return metadata_df
    
    def write_metadata(self, metadata_df):
        fpath = self.root + f'{self.experiment}/metadata/metadata_df.csv'
        metadata_df.to_csv(fpath, index=False)
    
    def _load_events(self, reader):
        try:
            events = reader.load('events')
            if len(events.index) > 0:
                return True
            else:
                return False
        except BaseException as e:
            return False
        
    def _load_contacts(self, reader):
        mni = False; tal = False
        try:
            contacts = reader.load('contacts')
            # look for mni and tal coordinates
            if set(['mni.x', 'mni.y', 'mni.z']).issubset(set(contacts.columns)):
                if (None not in contacts['mni.x'].unique()) and (None not in contacts['mni.y'].unique()) and (None not in contacts['mni.z'].unique()):
                    mni = True
            if set(['tal.x', 'tal.y', 'tal.z']).issubset(set(contacts.columns)):
                if (None not in contacts['tal.x'].unique()) and (None not in contacts['tal.y'].unique()) and (None not in contacts['tal.z'].unique()):
                    tal = True
            return True, mni, tal
        except BaseException as e:
            return False, mni, tal
        
    def _load_pairs(self, reader):
        try:
            pairs = reader.load('pairs')
            return True
        except BaseException as e:
            return False
        
    def _load_eeg(self, reader, ref):
        if ref == 'bipolar':
            pairs = reader.load('pairs')
            try:
                eeg_bi = reader.load_eeg(scheme=pairs)
                return True
            except BaseException as e:
                return False
        elif ref == 'monopolar':
            contacts = reader.load('contacts')
            try:
                eeg_mono = reader.load_eeg(scheme=contacts)
                return True
            except BaseException as e:
                return False
        else:
            raise ValueError("Reference must be either 'monopolar' or 'bipolar'.")
        
    def _sysv_units(self, row):
        if math.isnan(row.system_version):
            # do some detective work
            if row.experiment == 'pyFR':    # pyFR = system 1
                system_version = 1.0
            else:
                system_version = self._determine_system_version(row)
            unit_scale = self._determine_unit_scale(row, system_version)
        elif row.system_version == 1.0:
            system_version = row.system_version
            # have to determine units
            unit_scale = self._determine_unit_scale(row, system_version)
        elif row.system_version == 2.0 or row.system_version == 4.0:     # convert from 250 nV to V
            system_version = row.system_version
            unit_scale = 4000000.0
        elif row.system_version >= 3.0 and row.system_version < 4.0:     # convert from 0.1 uV to V
            system_version = 3.0
            unit_scale = 10000000.0
        else:
            raise ValueError("Not a valid system version in data index.")
        
        return system_version, unit_scale
    
    # query results from system_version_finder.py for NaN system versions
    def _determine_system_version(self, row):
        sys_vers = pd.read_csv('system_versions.csv')
        sv = sys_vers[(sys_vers.subject == row.subject) & (sys_vers.experiment==row.experiment) & 
                      (sys_vers.session == row.session)]
        return sv.iloc[0].system_version
    
    # sleuth system version by inferring from files
    # elemem folder = system 4
    # .ns2 = system 2
    # .h5 = system 3
    def _determine_system_version_search(self, row):
        sub_root = f'/data10/RAM/subjects/{row.subject}/'
        if self._system_4(row, sub_root):
            return 4.0
        elif self._system_3(row, sub_root):
            return 3.0
        elif self._system_2(row, sub_root):
            return 2.0
        else:
            return 1.0
    
    def _system_4(self, row, sub_root):
        sess_dir = sub_root + f'behavioral/{row.experiment}/session_{row.session}/'
        return os.path.exists(sess_dir) and 'elemem' in os.listdir(sess_dir)
    
    def _system_3(self, row, sub_root):
        sess_dir = sub_root + f'behavioral/{row.experiment}/session_{row.session}/'

        timestamped_directories = glob(sess_dir + 'host_pc/*')
        # remove all invalid names (valid names = only contains numbers and _)
        timestamped_directories = [
            d for d in timestamped_directories 
            if os.path.isdir(d) and all([c in string.digits for c in os.path.basename(d).replace('_', '')])
        ]
        # check each timestamped directory for a .h5 file, stop if find one
        for d in timestamped_directories:
            if 'eeg_timeseries.h5' in os.listdir(d):
                return True
        
        return False
    
    def _system_2(self, row, sub_root):
        raw_dir = sub_root + f'raw/{row.experiment}_{row.session}/'
        
        timestamped_directories = glob(raw_dir + '1*') + glob(raw_dir + '2*')    # some cases with 20 in year, others without
        # remove all invalid names (valid names = only contains numbers and -)
        timestamped_directories = [
            d for d in timestamped_directories 
            if os.path.isdir(d) and all([c in string.digits for c in os.path.basename(d).replace('-', '')])
        ]
        # check each timestamped directory for a .ns2, stop if find one
        for d in timestamped_directories:
            if len(glob(d + '/*.ns2')) > 0:
                return True
        
        return False

    def _determine_unit_scale(self, row, system_version):
        if system_version == 2.0 or system_version == 4.0:         # convert from 250 nV to v
            return 4000000.0
        elif system_version >= 3.0 and system_version < 4.0:       # convert from 0.1 uV to V
            return 10000000.0
        else:
            # read in from csv
            sys1_units = pd.read_csv('system_1_unit_conversions.csv')
            return sys1_units[sys1_units.subject == row.subject].iloc[0]['conversion_to_V']
    
    def _area_data(self, row):
        area_path = f'/data10/RAM/subjects/{row.subject}/docs/area.txt'
        return os.path.exists(area_path)
    
    def _brain_regions(self, row):
        # read in from csv
        region_data = pd.read_csv('bids_brain_regions.csv')
        regions = region_data[(region_data.subject==row.subject) & (region_data.experiment==row.experiment) & (region_data.session==row.session)]
        if len(regions) == 0:
            return dict(zip(self.BRAIN_REGIONS, np.zeros(len(self.BRAIN_REGIONS), dtype=int)))
        else:
            return {'wb.region':regions['wb.region'].iloc[0], 'ind.region':regions['ind.region'].iloc[0],
                    'das.region':regions['das.region'].iloc[0], 'stein.region':regions['stein.region'].iloc[0]}
        
    def _n_eegfiles(self, reader):
        # load events
        try:
            events = reader.load('events')
            # number of eegfile values
            eegfiles = [x for x in events['eegfile'].unique() if x != '']
            return len(eegfiles)
        except BaseException as e:
            return 0

