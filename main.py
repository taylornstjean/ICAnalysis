#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.8.2
#!/bin/env python3

from analysis.core import H5File, H5FileGroup, I3File
import analysis.config as config
import os

def main():

    filegroup = H5FileGroup(config.DATADIR)
    i3file = I3File(config.I3FILEPATH)

    print(i3file)
    print(filegroup)
    print(i3file.extract_metadata())
    

if __name__ == "__main__":
    main()
