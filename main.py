#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.8.2
#!/bin/env python3

from analysis.core import H5File, H5FileGroup, I3File, I3FileGroup
import analysis.config as config
import os

def main():


    h5file = H5File(config.TESTHDF5)
    h5file.plot_simweight("test.png")


if __name__ == "__main__":
    main()
