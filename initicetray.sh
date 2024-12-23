#!/bin/bash

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`
$SROOT/metaprojects/icetray/v1.8.2/env-shell.sh python3 "$@"

