### script to run BIDS metadata

# imports
from intracranial_BIDS_metadata import intracranial_BIDS_metadata, pyFR_BIDS_metadata
import sys

exp = sys.argv[1]

# create metadata object
if exp == 'pyFR':
    md = pyFR_BIDS_metadata()
else:
    md = intracranial_BIDS_metadata(exp)

# run metadata
metadata_df = md.metadata()

# save out metadata
md.write_metadata(metadata_df)
