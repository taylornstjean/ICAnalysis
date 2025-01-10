#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.8.2
#!/bin/env python3

import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

from analysis.core import I3FileGroup, I3File
import analysis.config as config


def main():

    i3filegroup = I3FileGroup(config.I3FILEDIRMAIN)
    i3filegroup.plot_vertices(interact_type=[1, 2], projection=True, histogram=True, d=2)


if __name__ == "__main__":
    main()
