#!/bin/bash
ENV_NAME="dip-env-rnn"
source activate $ENV_NAME
python session_init/gpu_checker.py
squeue -u $USER | grep ${USER:0:8} | awk '{print "scancel " $1}' > ./end_session.src
echo "rm end_session.src" >> ./end_session.src
echo "to end the session, run: source end_session.src"