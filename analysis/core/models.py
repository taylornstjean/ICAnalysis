import h5py
import os
import json
import pandas as pd
import numpy as np
import simweights
from numpy.typing import ArrayLike
import tqdm

from icecube import icetray, dataio, dataclasses, hdfwriter
from icecube.dataclasses import I3Particle
from icecube.icetray import I3Tray

from analysis.utils import listdir_absolute
from analysis.render import EventDetailHist, SimweightHist, PointCloud3D


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

    @staticmethod
    def _get_weight(hdfstore: pd.HDFStore) -> tuple:

        def _northern_track(energy: ArrayLike) -> ArrayLike:
            """
            This function to represent the IceCube northern track limit.
            Note that the units are GeV^-1 * cm^-2 * sr^-1 * s^-1 per particle type.
            """
            return 1.44e-18 / 2 * (energy / 1e5) ** -2.2

        weighter = simweights.NuGenWeighter(hdfstore, nfiles=100)

        weight = weighter.get_weights(_northern_track)
        primary_energy = weighter.get_column("PolyplopiaPrimary", "energy")

        return weight, primary_energy

    def plot_simweight(self, path: str):

        with pd.HDFStore(self._path, "r") as hdfstore:
            weight, primary_energy = self._get_weight(hdfstore)

        hist = SimweightHist()
        hist.populate(primary_energy, weight)
        hist.save(path)


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
        self._objects = [I3File(path) for path in self._paths]

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

    def plot_vertices(self, projection: bool=False, histogram: bool=False, d: int=1) -> None:

        metadata = {}
        print("Gathering Metadata...")
        for i, _obj in tqdm.tqdm(enumerate(self._objects), total=len(self._paths)):
            metadata[i] = _obj.extract_metadata()

        vertices = []
        print("\n\nParsing...")
        for j, entry in tqdm.tqdm(metadata.items()):
            for packet in entry:
                vertex = packet["vertex"]
                vertices.append([float(vertex.x), float(vertex.y), float(vertex.z)])

        fig = PointCloud3D(vertices)

        if not projection:
            fig.plot_3d("vertices.html")
        else:
            if not histogram:
                fig.plot_2d_projections("vertices_projected.html")
            else:
                if d == 1:
                    fig.plot_1d_histograms("vertices_projected_hist.html")
                elif d == 2:
                    fig.plot_2d_histograms("vertices_projected_hist_2d.html")

