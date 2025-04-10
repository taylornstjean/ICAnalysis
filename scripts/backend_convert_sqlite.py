#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.8.2
#!/bin/env python3

def main():
    from wrappers import parse_args
    args = parse_args()

    import sys
    sys.path.append("/data/i3home/tstjean/icecube")

    try:
        from analysis.core import I3FileGroup
        from analysis import config
    except ImportError as e:
        print(f"Error importing 'analysis': {e}")
        sys.exit(1)

    run_group_id = config.RUN_GROUP_ID

    i3file = I3FileGroup(args.input_file, run_group_id)
    i3file.to_backend("sqlite", workers=4)

if __name__ == "__main__":
    main()
