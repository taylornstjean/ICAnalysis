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

from analysis.core import I3FileGroup, H5File, H5FileGroup, I3File, I3BatchManager, DBFileGroup, DBFile
from analysis.utils import reset_temp_dir
import analysis.config as config

import cProfile
import numpy as np
import pstats


def main():

    run_group_id = config.RUN_GROUP_ID

    i3filegroup = I3FileGroup(config.I3FILEDIR_NUMU[run_group_id], run_group_id)
    db_path = i3filegroup.to_backend("sqlite", 40, merge=True)

    mergefile = DBFile(db_path)

    mergefile.generate_config()
    mergefile.plot_feature_distribution()


if __name__ == "__main__":
    main()
