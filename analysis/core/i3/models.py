# Standard library imports
import os
from glob import glob
import tempfile

# IceCube-specific modules
from icecube import icetray, dataio, hdfwriter

# Internal utility and job management modules
from analysis.utils import listdir_absolute, reset_temp_dir
from analysis.jobs import HTCondorJob, HTCondorBatch
from analysis.core.modules import I3Alerts
from analysis import config

# GraphNeT extractors and converters
from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
)
from graphnet.data import I3ToSQLiteConverter, I3ToParquetConverter

# Supported backend formats and corresponding converters
CONVERTER_CLASS = {
    "sqlite": I3ToSQLiteConverter,
    "parquet": I3ToParquetConverter,
}


class I3File:
    """
    Class for managing individual `.i3` files.

    Supports basic operations like converting to HDF5 or running GraphNeT converters.
    """

    __slots__ = ["_path", "_file", "__weights_cache"]

    def __init__(self, path: str) -> None:
        """
        Initializes an I3File object.

        Args:
            path (str): Absolute path to an `.i3` file.
        """
        self._path = path
        self._file = dataio.I3File(self._path)
        self.__weights_cache = {}  # For future caching of weights if needed

    def __repr__(self) -> str:
        return os.path.basename(self._path)

    def to_hdf5(self, path: str) -> None:
        """
        Converts the `.i3` file to an HDF5 format.

        Args:
            path (str): Destination path for the output HDF5 file.
        """
        tray = icetray.I3Tray()
        tray.Add("I3Reader", 'reader', FileNameList=[self._path], DropBuffers=True)
        tray.Add(I3Alerts)
        tray.Add(
            hdfwriter.I3HDFWriter,
            SubEventStreams=["InIceSplit"],
            keys=["PolyplopiaPrimary", "I3MCWeightDict", "HESEBool"],
            output=path
        )
        tray.Execute()

    def to_backend(self, backend: str, workers: int, group_id: int) -> str:
        """
        Convert a single `.i3.zst` file using a specified GraphNeT backend.

        Args:
            backend (str): Output format ('sqlite' or 'parquet').
            workers (int): Number of workers for parallel extraction.
            group_id (int): Group ID to organize output paths.

        Returns:
            str: Path to the converted output file.
        """
        assert backend in CONVERTER_CLASS

        # Clean previous temporary data
        reset_temp_dir()

        # Create new temp directory and symlink the file
        temp_dir = tempfile.mkdtemp(dir=os.path.join(config.BASE_DIR, "data/temp"))
        os.symlink(self._path, os.path.join(temp_dir, os.path.basename(self._path)))

        # Define inputs and output directory
        inputs = [temp_dir]
        outdir = os.path.join(config.DATA_DIR, f"backend/{group_id}/")
        gcd_rescue = config.SAMPLE_GCD_FILE
        os.makedirs(outdir, exist_ok=True)

        # Initialize the appropriate converter and execute
        converter = CONVERTER_CLASS[backend](
            extractors=[
                I3FeatureExtractorIceCube86("InIceDSTPulses"),
                I3TruthExtractor(mctree="I3MCTree_preMuonProp"),
            ],
            outdir=outdir,
            gcd_rescue=gcd_rescue,
            workers=workers,
        )
        converter(inputs)

        return os.path.join(outdir, os.path.basename(self._path).replace(".i3.zst", ".db"))


class I3FileGroup:
    """
    Class for managing a collection of `.i3.zst` files and processing them together.
    """

    def __init__(self, directory: str, group_id: int) -> None:
        """
        Initializes the I3FileGroup with directory and group ID.

        Args:
            directory (str): Path to directory with `.i3.zst` files.
            group_id (int): Group ID used to organize outputs and metadata.
        """
        self._directory = directory
        self._paths = listdir_absolute(self._directory)
        self.group_id = group_id
        self._metadata_directory = os.path.join(config.BASE_DIR, f"data/metadata/{group_id}")

        # Only keep files with correct extension
        self._paths = [path for path in self._paths if path.endswith(".i3.zst")]
        self._filenames = [os.path.basename(path) for path in self._paths]

    def __repr__(self) -> str:
        return self._directory

    def to_backend(self, backend: str, workers: int, outdir: str = "", merge: bool=False) -> str:
        """
        Convert all `.i3.zst` files in the group to specified backend format.

        Args:
            backend (str): Output format ('sqlite' or 'parquet').
            workers (int): Number of parallel workers.
            outdir (str): Output directory, defaults to default configured output directory.
            merge (bool): Whether to merge outputs into a single file.

        Returns:
            str: Path to merged output ".db" file if merge=True, or path to output directory if merge=False.
        """
        assert backend in CONVERTER_CLASS

        inputs = [self._directory]
        if not outdir:
            outdir = os.path.join(config.DATA_DIR, f"backend/{self.group_id}/")
        gcd_rescue = config.SAMPLE_GCD_FILE
        os.makedirs(outdir, exist_ok=True)

        converter = CONVERTER_CLASS[backend](
            extractors=[
                I3FeatureExtractorIceCube86("InIceDSTPulses"),
                I3TruthExtractor(mctree="I3MCTree_preMuonProp"),
            ],
            outdir=outdir,
            gcd_rescue=gcd_rescue,
            num_workers=workers
        )
        converter(inputs)

        if merge:
            converter.merge_files()
            return os.path.join(outdir, "merged/merged.db")
        else:
            return outdir

    def to_backend_batched(self) -> str:
        """
        Splits `.i3.zst` files into batches and submits HTCondor jobs to convert them.

        Returns:
            str: Path to output directory.
        """

        reset_temp_dir()  # Clean old temp dirs
        batch_size = 100  # Number of files per batch

        def batch_files(directory: str, b_size: int):
            """
            Utility to split directories of `.i3.zst` files into evenly-sized batches.
            """
            files = sorted(glob(f"{directory}/*.i3.zst"))
            return [files[i:i + b_size] for i in range(0, len(files), b_size)]

        # Create top-level temp directory
        temp_dir = tempfile.mkdtemp(dir=os.path.join(config.BASE_DIR, "data/temp"))

        # For each batch, create a subdirectory and symlink its files into it
        sub_dirs = []
        for batch in batch_files(self._directory, batch_size):
            sub_dir = tempfile.mkdtemp(dir=temp_dir)
            sub_dirs.append(sub_dir)
            for file in batch:
                os.symlink(file, os.path.join(sub_dir, os.path.basename(file)))

        # Output directory for all converted files
        path = os.path.join(config.BASE_DIR, f"data/backend/{self.group_id}")
        os.makedirs(path, exist_ok=True)

        # Configure and submit HTCondor batch job
        job = HTCondorBatch(
            temp_dir, path,
            "scripts/backend_convert_sqlite.py", ".i3.zst", "",
            python_executable_path="/data/i3home/tstjean/icecube/venv/bin/python3.11",
            DAGMAN_MAX_JOBS_SUBMITTED=5000,
            DAGMAN_MAX_JOBS_IDLE=5000,
            DAGMAN_MAX_SUBMITS_PER_INTERVAL=1000,
            DAGMAN_USER_LOG_SCAN_INTERVAL=1,
            venv_path="/data/i3home/tstjean/icecube/venv/bin/activate",
            has_avx2=True,
            request_cpus=4,
            request_memory=8192,
            exports=[
                "PYTHONPATH=/data/i3home/tstjean/icecube/venv/lib/python3.11/site-packages:$PYTHONPATH"
            ]
        )
        job.configure(clean=True)
        job.submit(monitor=True)

        return path

    def objects(self) -> I3File:
        """
        Generator for I3File objects from paths in this group.

        Yields:
            I3File: Wrapper object for each file.
        """
        for path in self._paths:
            yield I3File(path)

    def to_hdf5(self) -> None:
        """
        Submits a job to HTCondor to convert `.i3.zst` files to HDF5 format.
        """
        print("\nConverting .i3 files to .hdf5 file format...")

        path = os.path.join(config.BASE_DIR, f"data/hdf5/{self.group_id}")
        os.makedirs(path, exist_ok=True)

        job = HTCondorJob(
            self._directory, path,
            "scripts/i3_to_hdf5.py", ".i3.zst", ".hdf5",
            python_executable_path="/data/i3home/tstjean/icecube/venv/bin/python3.11",
            DAGMAN_MAX_JOBS_SUBMITTED=5000,
            DAGMAN_MAX_JOBS_IDLE=5000,
            DAGMAN_MAX_SUBMITS_PER_INTERVAL=1000,
            DAGMAN_USER_LOG_SCAN_INTERVAL=1,
            venv_path="/data/i3home/tstjean/icecube/venv/bin/activate"
        )
        job.configure(clean=True)
        job.submit(monitor=True)
