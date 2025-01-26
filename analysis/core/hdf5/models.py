########################################################################################################################
# imports

import h5py
import os
import json
import pandas as pd
import numpy as np
import simweights
from numpy.typing import ArrayLike
import subprocess

from analysis.render import SimweightHist, EventDetailHist
from analysis import config

########################################################################################################################

class H5File:
    """
    A class for managing .hdf5 files, extracting and processing data, and generating relevant plots.

    Attributes:
        _path (str): The file path to the .hdf5 file.
        _file (h5py.File): The opened h5py file object for reading the .hdf5 data.
    """

    __slots__ = ["_path", "_file"]

    def __init__(self, path):
        """
        Initializes the H5File object and opens the .hdf5 file at the given path.

        Args:
            path (str): File path to .hdf5 file.
        """
        self._path = path
        self._file = h5py.File(path, "r")

    def __repr__(self) -> str:
        _repr = os.path.basename(self._path)
        return _repr

    @property
    def charge(self) -> np.ndarray:
        """
        Retrieves charge data from the .hdf5 file.

        Returns:
            np.ndarray: Charge data.
        """
        try:
            charge = self._file['Homogenized_QTot']['value'][:]
        except KeyError:
            charge = np.array([])
        return charge

    @property
    def zenith(self) -> np.ndarray:
        """
        Retrieves zenith data from the .hdf5 file.

        Returns:
            np.ndarray: Zenith data.
        """
        try:
            zenith = self._file['PolyplopiaPrimary']['zenith'][:]
        except KeyError:
            zenith = np.array([])
        return zenith

    def weights(self, nfiles, group_id: int, spectrum_exp: float=-2.19) -> tuple:
        """
        Calculates weights for the data set.

        Args:
            nfiles (int): Number of constituent files in the .hdf5 file.
            group_id (int): The dataset group id.
            spectrum_exp (float): Assumed spectrum (E^spectrum_exp). Defaults to -2.19.

        Returns:
            tuple: Tuple of the form (Weights, Primary Energies)
        """
        print("Calculating weights, this may take a while for files with large numbers of events.")

        # open the .hdf5 file as a hdfstore object, and calculate weights
        with pd.HDFStore(self._path, "r") as hdfstore:
            def _northern_track(energy: ArrayLike) -> ArrayLike:
                """
                This function to represent the IceCube northern track limit.
                Note that the units are GeV^-1 * cm^-2 * sr^-1 * s^-1 per particle type.
                """
                return 1.44e-18 / 2 * (energy / 1e5) ** spectrum_exp

            # define the weighter and calculate weights, pull primary energy
            weighter = simweights.NuGenWeighter(hdfstore, nfiles=nfiles)

            weight = weighter.get_weights(_northern_track)
            primary_energy = weighter.get_column("PolyplopiaPrimary", "energy")

        # verify the directory exists and save the weights to a file
        path = os.path.join(config.BASE_DIR, f"data/weights/{group_id}")

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"weights.{group_id}.json"), "w+") as weight_file:
            json.dump(list(weight), weight_file)

        return weight, primary_energy

    def plot_distribution(self, path: str, nfiles: int, group_id: int) -> None:
        """
        Generates and saves a plot of the primary energy distribution with weights.

        Args:
            path (str): The directory path where the plot should be saved.
            nfiles (int): Number of constituent files in the .hdf5 file.
            group_id (int): The group ID for the dataset.
        """
        # retrieve weights for the dataset
        weight, primary_energy = self.weights(nfiles, group_id)

        # create a SimweightHist object and populate it
        hist = SimweightHist()
        hist.populate(primary_energy, weight)
        hist.save(path)

    def plot_event_details(self, path: str):
        """
        Generates and saves a plot of event details (charge and zenith).

        Args:
            path (str): The directory path where the plot should be saved.
        """
        # create an EventDetailHist object and populate it
        hist = EventDetailHist()
        hist.populate(self.charge, self.zenith)
        hist.save(path)


class H5FileGroup:

    """
    A class for managing multiple .hdf5 files.

    Attributes:
        _directory (str): Directory containing the .hdf5 files.
        group_id (int): The group ID for the dataset.
        _paths (list): List of .hdf5 file paths.
    """

    def __init__(self, directory, group_id):
        """
        Initializes the H5FileGroup object.

        Args:
            directory (str): Directory containing the .hdf5 files.
            group_id (int): The group ID for the dataset.
        """
        self._directory = directory
        self.group_id = group_id
        self._paths = [os.path.join(directory, path) for path in os.listdir(directory) if not path.startswith("combined")]

    def combine(self) -> None:
        """
        Merges all the .hdf5 files in the directory into a single file, saves the merge order.
        """

        # verify the directory exists
        os.makedirs(os.path.join(
            config.BASE_DIR, f"data/hdf5/{self.group_id}"
        ), exist_ok=True)

        # configure the merge command
        merge_command = [
            "hdfwriter-merge", "-o", os.path.join(
                config.BASE_DIR, f"data/hdf5/{self.group_id}/combined.{self.group_id}.hdf5"
            )
        ] + self._paths

        # run merge
        subprocess.run(merge_command, check=True)

        # verify the directory exists
        os.makedirs(os.path.join(
            config.BASE_DIR, f"data/config/{self.group_id}"
        ), exist_ok=True)

        # save a file with the order in which the files were merged. this is important so we can match up weights
        # to proper events later
        with open(os.path.join(config.BASE_DIR, f"data/config/{self.group_id}/merge_order.{self.group_id}.json"), "w+") as order_file:
            json.dump(self._paths, order_file, indent=4)

########################################################################################################################
