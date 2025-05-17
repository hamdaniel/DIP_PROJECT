#!/bin/bash
ENV_NAME="dip-env-rnn"
source activate $ENV_NAME
python session_init/gpu_checker.py
squeue -u $USER | grep ${USER:0:8} | awk '{print "To end process, run: scancel " $1}'