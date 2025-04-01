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

from analysis.core import I3FileGroup, H5File, H5FileGroup, I3File, I3BatchManager
from analysis.utils import reset_temp_dir
import analysis.config as config

import numpy as np


def main():

    run_group_id = config.RUN_GROUP_ID

    i3batch = I3FileGroup(config.I3FILEDIR_NUMU[run_group_id], run_group_id)
    i3batch.backend_convert("sqlite"
)


if __name__ == "__main__":
    main()
