# imports
import mne_bids
import pandas as pd

# ---------- BIDS Utility ----------

# return a base BIDS_path object to update
def BIDS_path(self):
    bids_path = mne_bids.BIDS_path(subject=self.subject, task=self.experiment, session=str(self.session),
                                    root=self.root)
    return bids_path

# write pandas dataframe to tsv file
def to_tsv(self, dframe, fpath):
    dframe.to_csv(fpath, sep='\t', index=False)