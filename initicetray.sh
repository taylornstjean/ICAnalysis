#!/bin/bash

export HDF5_DISABLE_VERSION_CHECK=1

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)
"$SROOT"/metaprojects/icetray/v1.8.2/env-shell.sh /data/i3home/tstjean/icecube/venv/bin/python3.11 "$@"


