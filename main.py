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

from analysis.core import I3FileGroup, H5File, H5FileGroup, I3File, DBFileGroup, DBFile
from analysis.utils import reset_temp_dir
import analysis.config as config
import tempfile

import cProfile
import numpy as np
import os
from pathlib import Path
import pstats


def main():

    run_group_id = config.RUN_GROUP_ID
    db_path = os.path.join(config.DATA_DIR, f"backend/test/")

    #i3filegroup = I3FileGroup(config.I3FILEDIR_NUMU_TEST, run_group_id)
    #i3filegroup.to_backend("sqlite", 60, merge=False, outdir=db_path)

    #dbfilegroup = DBFileGroup(db_path)
    #db_merged = dbfilegroup.merge()

    db_merged = os.path.join(db_path, "merged/merged.db")
    mergefile = DBFile(db_merged)

    mergefile.generate_training_config()
    mergefile.train_model(None, 100, 10, 16, 16, "run0")


if __name__ == "__main__":
    main()
