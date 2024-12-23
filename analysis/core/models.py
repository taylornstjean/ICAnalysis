import h5py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simweights
from numpy.typing import ArrayLike

from icecube import icetray, dataio, dataclasses, hdfwriter, simclasses
from icecube.dataclasses import I3Particle
from icecube.icetray import I3Tray
from tables import NoSuchNodeError

from analysis.utils import listdir_absolute
from analysis.render import EventDetailHist, SimweightHist


class H5File:

    """Class for opening and managing .hdf5 files."""

    __slots__ = ["_path", "_file"]

    def __init__(self, path):
        self._path = path
        self._file = h5py.File(path, "r")

    def __repr__(self) -> str:
        _repr = "\n<-- H5File Object -->\n"
        _repr = f"\nFile:\t\t\t\t\t\t{os.path.basename(self._path)}\n"
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


class H5FileGroup:

    """Object that holds a collection of .hdf5 files. Each file is stored as an H5File object."""

    __slots__ = ["_directory", "_paths", "_filenames"]

    def __init__(self, directory: str):
        self._directory = directory
        self._paths = listdir_absolute(self._directory)
        self._filenames = [os.path.basename(path) for path in self._paths]

    def __repr__(self) -> str:
        _repr = "\n<-- H5FileGroup Object -->\n"
        _repr += f"\nDirectory:\t\t\t\t\t\t{self._directory}"
        _repr += f"\nNumber of files in Group:\t\t{len(self._paths)}"

        return _repr

    def plot(self, _type: str, path: str) -> None:
        """Generate quick plots."""
        if _type == "hist":
            plot = EventDetailHist()
            plot.populate(self.charge, self.zenith)
            plot.save(path)

    @property
    def charge(self) -> np.ndarray:
        return self._get_parameter("charge")

    @property
    def zenith(self) -> np.ndarray:
        return self._get_parameter("zenith")

    def _get_parameter(self, parameter: str) -> np.ndarray:
        print(f"\n--> Concatenating {parameter}...")

        array = np.array([])
        for i, file in enumerate(self._h5file_objects):
            print(f"\rCurrent file: {self._filenames[i]}", end="")
            array = np.concatenate([array, getattr(file, parameter)])

        print("\rCurrent file: Done")
        return array

    @property
    def _h5file_objects(self) -> iter:
        for path in self._paths:
            yield H5File(path)

    @staticmethod
    def _get_weight(hdfstore: pd.HDFStore, weight: list, primary_energy: list) -> bool:

        def _northern_track(energy: ArrayLike) -> ArrayLike:
            """
            This function to represent the IceCube northern track limit.
            Note that the units are GeV^-1 * cm^-2 * sr^-1 * s^-1 per particle type.
            """
            return 1.44e-18 / 2 * (energy / 1e5) ** -2.2

        try:
            weighter = simweights.NuGenWeighter(hdfstore, nfiles=1)
        except NoSuchNodeError:
            return False

        weight.extend(weighter.get_weights(_northern_track))
        primary_energy.extend(weighter.get_column("PolyplopiaPrimary", "energy"))

        return True

    def plot_simweight(self, path: str):
        weight = []
        primary_energy = []

        for file in self._paths:
            print(f"\rCurrent file: {file}", end="")
            with pd.HDFStore(file, "r") as hdfstore:
                self._get_weight(hdfstore, weight, primary_energy)

        # convert to numpy arrays
        weight = np.array(weight)
        primary_energy = np.array(primary_energy)

        hist = SimweightHist()
        hist.populate(primary_energy, weight)
        hist.save(path)

class I3File:

    """Class for opening and managing .i3 files."""

    __slots__ = ["_path", "_file"]

    def __init__(self, path: str):
        self._path = path
        self._file = dataio.I3File(self._path)

    def __repr__(self) -> str:
        _repr = "\n<-- I3File Object -->\n"
        _repr = f"\nFile:\t\t\t\t\t\t{os.path.basename(self._path)}\n"
        return _repr

    def extract_metadata(self) -> list:
        _metadata = []

        while self._file.more():
            frame = self._file.pop_frame()

            # Check if the frame is a Physics frame
            if frame.Stop == icetray.I3Frame.Physics:
                mc_tree = frame["I3MCTree_preMuonProp"]
                primary_neutrino = dataclasses.get_most_energetic_primary(mc_tree)

                if primary_neutrino is None or not self.is_neutrino(primary_neutrino):
                    # if the primary neutrino is non-existent, or isn't even a neutrino, we don't care about it
                    continue

                interacting_neutrino = self.get_interacting_nu(primary_neutrino, mc_tree)
                vertex = self.get_vertex(interacting_neutrino)

                _metadata += [{
                    "frame": frame,
                    "primary_nu": primary_neutrino,
                    "interacting_nu": interacting_neutrino,
                    "vertex": vertex
                }]

        return _metadata

    def get_interacting_nu(self, neutrino: I3Particle, mc_tree) -> I3Particle | None:
        """Walk through the tree to find the first InIce neutrino."""
        while neutrino.location_type != dataclasses.I3Particle.LocationType.InIce:
            children = mc_tree.get_daughters(neutrino)
            _found_next = False

            # iterate through each child particle
            for child in children:
                if self.is_neutrino(child):
                    neutrino = child
                    _found_next = True
                    break

            # if the tree ends, break
            if not _found_next:
                return None

        return neutrino

    @staticmethod
    def get_vertex(neutrino: I3Particle) -> float:
        """Get the vertex position of a neutrino."""
        stop_pos = neutrino.pos + neutrino.dir * neutrino.length
        return stop_pos

    @staticmethod
    def is_neutrino(neutrino: I3Particle) -> bool:
        """Determine if a particle is a neutrino by checking its metadata."""
        _is_neutrino = False
        i3pt = I3Particle.ParticleType
        neutrino_type = neutrino.type

        if neutrino_type in [i3pt.NuE, i3pt.NuEBar, i3pt.NuMu, i3pt.NuMuBar, i3pt.NuTau, i3pt.NuTauBar]:
            _is_neutrino = True

        return _is_neutrino


class I3FileGroup:

    def __init__(self, directory: str):
        self._directory = directory
        self._paths = listdir_absolute(self._directory)
        self._filenames = [os.path.basename(path) for path in self._paths]

    def __repr__(self) -> str:
        _repr = "\n<-- I3FileGroup Object -->\n"
        _repr += f"\nDirectory:\t\t\t\t\t\t{self._directory}"
        _repr += f"\nNumber of files in Group:\t\t{len(self._paths)}"

        return _repr

    def to_hdf5(self, path: str) -> None:
        tray = I3Tray()
        tray.Add("I3Reader", FileNameList=self._paths)
        tray.Add(
            hdfwriter.I3HDFWriter,
            SubEventStreams=["InIceSplit"],
            keys=["PolyplopiaPrimary", "I3MCWeightDict"],
            output=path
        )
        tray.Execute()

