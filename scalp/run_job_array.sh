#!/bin/bash
#
#SBATCH --job-name=bids_convert
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=15GB
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jrudoler@sas.upenn.edu
#SBATCH --output=slurm_%A_%a.out

SRCDIR=$HOME/bids-convert

/home1/jrudoler/anaconda3/envs/py38/bin/python -u $SRCDIR/convert.py $SLURM_ARRAY_TASK_ID
