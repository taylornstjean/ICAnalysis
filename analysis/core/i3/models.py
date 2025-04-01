########################################################################################################################
# imports

import os
import json
import tqdm
from glob import glob
import subprocess
import platform
import tempfile

from icecube import icetray, dataio, dataclasses, hdfwriter
from icecube.dataclasses import I3Particle

from analysis.utils import listdir_absolute, reset_temp_dir
from analysis.render import PointCloud3D
from analysis.jobs import HTCondorJob, HTCondorBatch
from analysis.core.modules import I3Alerts
from analysis import config

from scipy.spatial import Delaunay

from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCube86,
    I3RetroExtractor,
    I3TruthExtractor,
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

########################################################################################################################

class I3File:

    """
    A class for opening and managing `.i3` files.

    Attributes:
        _path (str): The path to the `.i3` file.
        _file (dataio.I3File): An instance of the `I3File` class from the `dataio` module used to read `.i3` files.
    """

    __slots__ = ["_path", "_file", "__weights_cache"]

    def __init__(self, path: str) -> None:
        """
       Initializes an I3File object.

       Args:
           path (str): The file path to the `.i3` file.
       """
        self._path = path
        self._file = dataio.I3File(self._path)

        # initialize cache for weights
        self.__weights_cache = {}

    def __repr__(self) -> str:
        _repr = os.path.basename(self._path)
        return _repr

    def to_hdf5(self, path: str) -> None:
        """
        Converts the `.i3` file to an .hdf5 file.

        Args:
            path (str): The output file path for the HDF5 file.
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

    def __preload_weights(self, group_id) -> None:
        """
        Preloads weights into memory.

        Args:
            group_id (int): The dataset group id.
        """
        # open and load the weights configuration file
        with open(
            os.path.join(config.BASE_DIR, f"data/config/{group_id}/config.{group_id}.json"), "r"
        ) as config_file:
            weight_config = json.load(config_file)

        # open and load the weights file
        with open(
            os.path.join(config.BASE_DIR, f"data/weights/{group_id}/weights.{group_id}.json"), "r"
        ) as weights_file:
            weight_data = json.load(weights_file)

        # pull the starting index and size for the current file from the weights configuration file
        key = repr(self).replace(".i3.zst", "")

        init_index = weight_config[key]["index_start"]
        frame_count = weight_config[key]["size"]

        # save sliced values to conserve memory
        self.__weights_cache = weight_data[init_index:init_index + frame_count + 1]

    def p_frames(self):
        """
        Counts the number of Physics frames in the `.i3` file.

        Returns:
            int: The number of Physics frames in the `.i3` file.
        """
        counter = 0

        # iterate over frames in the file
        while self._file.more():
            frame = self._file.pop_frame()

            # Check if the frame is a Physics frame
            if frame.Stop == icetray.I3Frame.Physics:
                counter += 1

        return counter

    def metadata(self, group_id) -> list:
        """
        Extracts relevant metadata from the `.i3` file, including vertex information, interaction type,
        any alerts, and associated weights.

        Args:
            group_id (int): The dataset group id.

        Returns:
            list: A list of dictionaries containing metadata for each Physics frame.
        """
        # load weights into memory
        self.__preload_weights(group_id)

        # initialize metadata list
        _metadata = []
        index = 0

        # iterate over each frame in the file
        while self._file.more():
            frame = self._file.pop_frame()

            # Check if the frame is a Physics frame
            if not frame.Stop == icetray.I3Frame.Physics:
                continue

            # pull data from the frame
            mc_tree = frame["I3MCTree_preMuonProp"]
            i3mc_wd = frame["I3MCWeightDict"]
            alerts = list(frame["AlertNamesPassed"])

            if not mc_tree or not i3mc_wd:
                continue

            # grab the primary neutrino
            primary_neutrino = dataclasses.get_most_energetic_primary(mc_tree)

            if not primary_neutrino or not self.is_neutrino(primary_neutrino):
                # if the primary neutrino is non-existent, or isn't even a neutrino, we don't care about it
                continue

            # get the interacting neutrino
            interacting_neutrino = self.get_interacting_nu(primary_neutrino, mc_tree)

            if not interacting_neutrino:
                continue

            # get the vertex of the interaction
            vertex = self.get_vertex(interacting_neutrino)

            # get the interaction type
            interaction_type = i3mc_wd["InteractionType"]

            # append it to the metadata list
            _metadata.append({
                "vertex": vertex,
                "interact_type": interaction_type,
                "alerts": alerts,
                "weight": self.__weights_cache[index]
            })

            index += 1

        return _metadata

    def get_interacting_nu(self, primary_neutrino: I3Particle, mc_tree) -> I3Particle | None:
        """
        Walks through the MC tree to find the first InIce neutrino.

        Args:
            primary_neutrino (I3Particle): The neutrino particle to check.
            mc_tree: The MC tree containing particle information.

        Returns:
            I3Particle | None: The first InIce neutrino found, or None if no InIce neutrino is found.
        """
        # iterate through the tree, stopping when we find an InIce neutrino
        stack = [primary_neutrino]
        while stack:
            # use stack approach to prevent recursion
            neutrino = stack.pop()
            if neutrino.location_type == dataclasses.I3Particle.LocationType.InIce:
                return neutrino
            stack.extend(child for child in mc_tree.get_daughters(neutrino) if self.is_neutrino(child))
        return None

    @staticmethod
    def get_vertex(neutrino: I3Particle) -> [float]:
        """
        Retrieves the vertex position of a neutrino.

        Args:
            neutrino (I3Particle): The neutrino particle to get the vertex position for.

        Returns:
            [float]: A list containing the x, y, and z coordinates of the vertex position.
        """
        stop_pos = neutrino.pos + neutrino.dir * neutrino.length

        # convert to list of floats for easier processing
        vertex = [float(stop_pos.x), float(stop_pos.y), float(stop_pos.z)]
        return vertex

    @staticmethod
    def is_neutrino(neutrino: I3Particle) -> bool:
        """
        Determines if a particle is a neutrino based on its metadata.

        Args:
            neutrino (I3Particle): The particle to check.

        Returns:
            bool: True if the particle is a neutrino, False otherwise.
        """
        _is_neutrino = False
        i3pt = I3Particle.ParticleType
        neutrino_type = neutrino.type

        # determine if the particle is any neutrino type
        if neutrino_type in [i3pt.NuE, i3pt.NuEBar, i3pt.NuMu, i3pt.NuMuBar, i3pt.NuTau, i3pt.NuTauBar]:
            _is_neutrino = True

        return _is_neutrino


class I3FileGroup:
    """
    A class for managing a group of `.i3.zst` files and processing them.

    Attributes:
        _directory (str): The directory containing the `.i3.zst` files.
        _paths (list): List of absolute paths to `.i3.zst` files in the directory.
        group_id (int): The dataset group id.
        _metadata_directory (str): The directory where metadata for the files is stored.
    """

    CONVERTER_CLASS = {
        "sqlite": SQLiteDataConverter,
        "parquet": ParquetDataConverter,
    }

    def __init__(self, directory: str, group_id: int) -> None:
        """
        Initializes the I3FileGroup with a specified directory and group ID.

        Args:
            directory (str): The directory containing the `.i3.zst` files.
            group_id (int): The dataset group id.
        """
        self._directory = directory
        self._paths = listdir_absolute(self._directory)
        self.group_id = group_id
        self._metadata_directory = os.path.join(config.BASE_DIR, f"data/metadata/{group_id}")

        # remove any files without a .i3.zst extension
        for path in self._paths:
            if not path.endswith(".i3.zst"):
                self._paths.remove(path)

        self._filenames = [os.path.basename(path) for path in self._paths]

    def __repr__(self) -> str:
        _repr = self._directory
        return _repr

    def backend_convert(self, backend: str):
        assert backend in self.CONVERTER_CLASS

        # define file paths
        inputs = [self._directory]
        outdir = os.path.join(config.BASE_DIR, f"data/backend/{self.group_id}/")
        gcd_rescue = config.SAMPLE_GCD_FILE

        # initialize the converter
        converter = self.CONVERTER_CLASS[backend](
            extractors=[
                I3FeatureExtractorIceCube86("InIceDSTPulses"),
                I3TruthExtractor(),
            ],
            outdir=outdir,
            gcd_rescue=gcd_rescue,
            workers=40,
        )
        converter(inputs)
        if backend == "sqlite":
            converter.merge_files()

    def objects(self) -> I3File:
        """
       Returns an iterator over `I3File` objects corresponding to the `.i3.zst` files in the group.

       Yields:
           I3File: An `I3File` object for each file in the group.
       """
        for path in self._paths:
            yield I3File(path)

    def to_hdf5(self) -> None:
        """
        Converts `.i3.zst` files to .hdf5 format.

        This method submits a job to HTCondor to run the conversion script.
        """
        print("\nConverting .i3 files to .hdf5 file format...")

        # verify the output directory exists
        path = os.path.join(config.BASE_DIR, f"data/hdf5/{self.group_id}")
        os.makedirs(path, exist_ok=True)

        # create and submit htcondor job
        job = HTCondorJob(
            self._directory, path,
            "scripts/i3_to_hdf5.py", ".i3.zst", ".hdf5",
            python_executable_path="/data/i3home/tstjean/icecube/venv/bin/python3.11", DAGMAN_MAX_JOBS_SUBMITTED=5000,
            DAGMAN_MAX_JOBS_IDLE=5000, DAGMAN_MAX_SUBMITS_PER_INTERVAL=1000, DAGMAN_USER_LOG_SCAN_INTERVAL=1,
            venv_path="/data/i3home/tstjean/icecube/venv/bin/activate"
        )
        job.configure(clean=True)
        job.submit(monitor=True)

    def get_alert_rate(self, alert: str):
        """
        Calculates the alert rate for a given alert type.

        Args:
            alert (str): The alert type to calculate the rate for.

        Returns:
            int: The computed alert rate.
        """
        print("\nComputing an alert rate...")

        count = 0

        # iterate over each file in the metadata directory
        for file in tqdm.tqdm(os.listdir(self._metadata_directory)):
            # load the metadata for the sim file group
            with open(os.path.join(self._metadata_directory, file), "r") as metadata_file:
                metadata = json.load(metadata_file)

            # iterate over each entry in the metadata
            for entry in metadata:

                # skip if the entry does not include the specified alert
                if not alert in entry["alerts"]:
                    continue

                # add the weighted event count to the tally
                count += entry["weight"]

        # verify the directory exists
        alert_dir = os.path.join(config.BASE_DIR, f"data/events/{self.group_id}/{alert.lower()}")
        os.makedirs(alert_dir, exist_ok=True)

        rate_file_path = os.path.join(alert_dir, f"rate.{alert.lower()}.{self.group_id}.json")

        # save the rate to a file
        with open(rate_file_path, "w+") as rate_file:
            json.dump({"rate": count}, rate_file, indent=4)

        return count

    def plot_vertices(self, interact_type: list, projection: bool=False, histogram: bool=False, d: int=1) -> None:
        """
        Visualizes 3D vertex data with projections and histograms.

        Args:
            interact_type (list): A list of interaction types to filter the vertex data.
            projection (bool, optional): Whether to plot a projection plot. Defaults to False.
            histogram (bool, optional): Whether to plot a histogram. Defaults to False.
            d (int, optional): The number of dimensions for the histogram (if plotting a histogram). Defaults to 1.
        """
        # load the pre-extracted metadata
        metadata = self.load_metadata()
        data = []

        # iterate over all metadata and retrieve vertices and associated weights
        print("\n\nParsing...")
        for j, entry in tqdm.tqdm(metadata.items()):
            for packet in entry:
                # only pull data for specified interaction types
                if any([int(packet["interact_type"]) == n for n in interact_type]):
                    data.append([packet["vertex"], packet["weight"]])

        # initialize a PointCloud() object
        fig = PointCloud3D(data)

        # verify the plot directory exists
        os.makedirs(
            os.path.join(config.BASE_DIR, "data/plots"), exist_ok = True
        )

        path = os.path.join(config.BASE_DIR, "data/plots/vertices.html")

        # choose plot type based in input arguments
        if not projection:
            fig.plot_3d(path)
            return

        if not histogram:
            fig.plot_2d_projections(path)
            return

        if d == 1:
            fig.plot_1d_histograms(path)
        elif d == 2:
            fig.plot_2d_histograms(path)


class I3BatchManager:
    """Class to generate batches for processing, generally for HTCondor submissions."""

    def __init__(self, directory: str, group_id: int):
        self._directory = directory
        self._group_id = group_id

    def batch_backend_convert(self):

        # clean the temp directory
        reset_temp_dir()

        batch_size = 100

        def split_files(directory: str, b_size: int):
            """Split the files into batches"""
            files = sorted(glob(f"{directory}/*.i3.zst"))  # List and sort files
            batches = [files[i:i + b_size] for i in range(0, len(files), b_size)]
            return batches

        # define the base temp directory
        temp_dir = tempfile.mkdtemp(dir=os.path.join(config.BASE_DIR, "data/temp"))

        # create a temp subdir for each batch, and simlink the batch contents
        sub_dirs = []
        for batch in split_files(self._directory, batch_size):
            sub_dirs.append(tempfile.mkdtemp(dir=temp_dir))

            for file in batch:
                os.symlink(file, os.path.join(sub_dirs[-1], os.path.basename(file)))

        # init the jobs
        path = os.path.join(config.BASE_DIR, f"data/backend/{self._group_id}")
        os.makedirs(path, exist_ok=True)

        job = HTCondorBatch(
            temp_dir, path,
            "scripts/backend_convert_sqlite.py", ".i3.zst", "",
            python_executable_path="/data/i3home/tstjean/icecube/venv/bin/python3.11", DAGMAN_MAX_JOBS_SUBMITTED=5000,
            DAGMAN_MAX_JOBS_IDLE=5000, DAGMAN_MAX_SUBMITS_PER_INTERVAL=1000, DAGMAN_USER_LOG_SCAN_INTERVAL=1,
            venv_path="/data/i3home/tstjean/icecube/venv/bin/activate", has_avx2=True, request_cpus=4,
            exports=[
                "PYTHONPATH=/data/i3home/tstjean/icecube/venv/lib/python3.11/site-packages:$PYTHONPATH"
            ]
        )
        job.configure(clean=True)
        job.submit(monitor=True)


########################################################################################################################
