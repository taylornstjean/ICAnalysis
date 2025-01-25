#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.8.2
#!/usr/bin/env python3

import warnings


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

from analysis.core import I3FileGroup, H5File, H5FileGroup
import analysis.config as config
import os
import json


def main():

    h5filegroup = H5FileGroup("data/hdf5/21220")
    h5filegroup.combine()

    h5file = H5File("data/hdf5/21220/combined.21220.hdf5")
    h5file.weights(9954, "data/weights/21220/weights.21220.json")

    i3filegroup = I3FileGroup(config.I3FILEDIR_NUMU, 21220)
    i3filegroup.get_p_frame_count()
    i3filegroup.generate_weight_config_file()
    i3filegroup.extract_metadata()
    i3filegroup.get_alert_rate("HESE")

if __name__ == "__main__":
    main()
