#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.8.2
#!/usr/bin/env python3

import warnings


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*HDF5 library version mismatched error.*"
)

from analysis.core import I3FileGroup, H5File, H5FileGroup, I3File
import analysis.config as config

import numpy as np


def main():

    run_group_id = config.RUN_GROUP_ID

    #i3filegroup = I3FileGroup(config.I3FILEDIR_NUMU[run_group_id], run_group_id)
    #i3filegroup.to_hdf5()

    #h5filegroup = H5FileGroup(f"data/hdf5/{run_group_id}", run_group_id)
    #h5filegroup.combine()

    h5file = H5File(f"data/hdf5/{run_group_id}/combined.{run_group_id}.hdf5")
    weights, _, hese = h5file.weights(config.NFILES[run_group_id], run_group_id)

    hese_bool = list(map(bool, hese))
    hese_count = weights[hese_bool]

    print(np.sum(hese_count) * 3.154e+7 * 9.6)


if __name__ == "__main__":
    main()
