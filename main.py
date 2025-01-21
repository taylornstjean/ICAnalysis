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
import os
import json


def main():

    # file = H5File("output/main.hdf5")
    # weights = file.weights(9954)

    # group = I3FileGroup(config.I3FILEDIRMAIN)
    # HESE_count = group.get_total_HESE_count([-500, 500], [-500, 500], [-500, 500])
    # print(HESE_count)

    # i3plotter = I3Plotter("/data/i3home/tstjean/icecube/jobs/output/json")
    # i3plotter.plot([2], weights)

    count = 0

    for file in os.listdir("jobs/output/HC"):
        with open(os.path.join("jobs/output/HC", file), "r") as f:
            entry = json.load(f)
            count += entry["HESE_count"]

    with open("output/HESE_total_count", "w") as f:
        json.dump({"HESE_total_count": count}, f)

if __name__ == "__main__":
    main()
