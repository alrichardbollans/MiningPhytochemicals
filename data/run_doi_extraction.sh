#!/bin/bash

#SBATCH --partition=medium
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="run_doi_extraction_log"
#SBATCH -o run_doi_extraction_log.out
#SBATCH --mail-user=a.richard-bollans@kew.org
#SBATCH --mail-type=END,FAIL

set -e # stop if fails
cd $SCRATCH/MiningPhytochemicals/data
source activate scratch_interpreter
python parse_refs.py
