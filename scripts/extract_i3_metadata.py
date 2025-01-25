#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.8.2
#!/bin/env python3

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    type=str,
    dest="input_file",
    required=True
)
parser.add_argument(
    "-o",
    type=str,
    dest="output_file",
    required=True
)

args = parser.parse_args()

import sys
sys.path.append("/data/i3home/tstjean/icecube")

try:
    from analysis.core import I3File
except ImportError as e:
    print(f"Error importing 'analysis': {e}")
    sys.exit(1)

import json

file = I3File(args.input_file)
metadata = file.metadata()

with open(args.output_file, "w+") as metadata_file:
    json.dump(metadata, metadata_file, indent=4)