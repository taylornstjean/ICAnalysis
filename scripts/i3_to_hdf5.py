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

from icecube import hdfwriter, icetray


class I3File:

    """Class for opening and managing .hdf5 files."""

    def __init__(self, path: str):
        self._path = path

    def to_hdf5(self, path: str) -> None:
        tray = icetray.I3Tray()
        tray.Add("I3Reader", 'reader', FileNameList=[self._path], DropBuffers=True)
        tray.Add(
            hdfwriter.I3HDFWriter,
            SubEventStreams=["InIceSplit"],
            keys=["PolyplopiaPrimary", "I3MCWeightDict"],
            output=path
        )
        tray.Execute()


file = I3File(args.input_file)
file.to_hdf5(args.output_file)
