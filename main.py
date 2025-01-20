#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.8.2
#!/usr/bin/env python3

import warnings

from analysis.core.models import I3Plotter

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

from analysis.core import I3FileGroup, I3Plotter, I3File, H5File
import analysis.config as config


def main():

    file = H5File("output/main.hdf5")
    weights = file.weights(9954)

    # group = I3FileGroup(config.I3FILEDIRMAIN)
    #  group.to_hdf5("./output/main.hdf5")

    i3plotter = I3Plotter("/data/i3home/tstjean/icecube/jobs/output/json")
    i3plotter.plot([1, 2], weights)


if __name__ == "__main__":
    main()
