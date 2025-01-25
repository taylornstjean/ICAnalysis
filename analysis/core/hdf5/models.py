import h5py
import os
import json
import pandas as pd
import numpy as np
import simweights
from numpy.typing import ArrayLike
import subprocess

from analysis.render import SimweightHist, EventDetailHist
from analysis.jobs import HTCondorJob

class H5File:

    """Class for opening and managing .hdf5 files."""

    __slots__ = ["_path", "_file"]

    def __init__(self, path):
        self._path = path
        self._file = h5py.File(path, "r")

    def __repr__(self) -> str:
        _repr = os.path.basename(self._path)
        return _repr

    @property
    def charge(self) -> np.ndarray:
        try:
            charge = self._file['Homogenized_QTot']['value'][:]
        except KeyError:
            charge = np.array([])
        return charge

    @property
    def zenith(self) -> np.ndarray:
        try:
            zenith = self._file['PolyplopiaPrimary']['zenith'][:]
        except KeyError:
            zenith = np.array([])
        return zenith

    def weights(self, nfiles, path: str=None, spectrum_exp: float=-2.19):
        with pd.HDFStore(self._path, "r") as hdfstore:
            def _northern_track(energy: ArrayLike) -> ArrayLike:
                """
                This function to represent the IceCube northern track limit.
                Note that the units are GeV^-1 * cm^-2 * sr^-1 * s^-1 per particle type.
                """
                return 1.44e-18 / 2 * (energy / 1e5) ** spectrum_exp

            weighter = simweights.NuGenWeighter(hdfstore, nfiles=nfiles)

            weight = weighter.get_weights(_northern_track)
            primary_energy = weighter.get_column("PolyplopiaPrimary", "energy")

        if path:
            with open(path, "w+") as weight_file:
                json.dump(list(weight), weight_file)

        return weight, primary_energy

    def plot_distribution(self, path: str, nfiles):
        weight, primary_energy = self.weights(nfiles)

        hist = SimweightHist()
        hist.populate(primary_energy, weight)
        hist.save(path)

    def plot_event_details(self, path: str):
        hist = EventDetailHist()
        hist.populate(self.charge, self.zenith)
        hist.save(path)


class H5FileGroup:

    """Class for managing multiple .hdf5 files."""

    def __init__(self, directory):
        self._directory = directory
        self._paths = [os.path.join(directory, path) for path in os.listdir(directory) if not path.startswith("combined")]

    def combine(self):
        merge_command = ["hdfwriter-merge", "-o", "/data/i3home/tstjean/icecube/data/hdf5/21220/combined.21220.hdf5"] + self._paths
        subprocess.run(merge_command, check=True)

        with open("/data/i3home/tstjean/icecube/data/config/21220/merge_order.21220.json", "w+") as order_file:
            json.dump(self._paths, order_file, indent=4)
