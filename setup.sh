#!/bin/bash
set -eu
source activate base 
conda env create -f environment.yml
conda activate llr_SP4
python setup.py install
