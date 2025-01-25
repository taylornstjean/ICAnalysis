import os
import json
import tqdm

from icecube import icetray, dataio, dataclasses, hdfwriter
from icecube.dataclasses import I3Particle
from icecube.icetray import I3Tray

from analysis.utils import listdir_absolute
from analysis.render import PointCloud3D
from analysis.jobs import HTCondorJob
from analysis import config

from scipy.spatial import Delaunay


class I3File:

    """Class for opening and managing .i3 files."""

    __slots__ = ["_path", "_file"]

    def __init__(self, path: str):
        self._path = path
        self._file = dataio.I3File(self._path)

    def __repr__(self) -> str:
        _repr = os.path.basename(self._path)
        return _repr

    def to_hdf5(self, path: str) -> None:
        tray = icetray.I3Tray()
        tray.Add("I3Reader", 'reader', FileNameList=[self._path], DropBuffers=True)
        tray.Add(
            hdfwriter.I3HDFWriter,
            SubEventStreams=["InIceSplit"],
            keys=["PolyplopiaPrimary", "I3MCWeightDict"],
            output=path
        )
        tray.Execute()

    def weight(self, index, group_id) -> iter:
        with open(os.path.join(config.BASE_DIR, f"data/config/{group_id}/config.{group_id}.json"), "r") as config_file:
            weight_config = json.load(config_file)

        with open(os.path.join(config.BASE_DIR, f"data/config/{group_id}/weights.{group_id}.json"), "r") as weights_file:
            weight_data = json.load(weights_file)

        init_index = weight_config[repr(self).replace(".i3.zst", "")]["index_start"]

        return weight_data[index + init_index]

    def p_frames(self):
        counter = 0

        while self._file.more():
            frame = self._file.pop_frame()

            # Check if the frame is a Physics frame
            if frame.Stop == icetray.I3Frame.Physics:
                counter += 1

        return counter

    def metadata(self, group_id) -> list:
        _metadata = []
        index = 0

        while self._file.more():
            frame = self._file.pop_frame()

            # Check if the frame is a Physics frame
            if frame.Stop == icetray.I3Frame.Physics:
                mc_tree = frame["I3MCTree_preMuonProp"]
                i3mc_wd = frame["I3MCWeightDict"]
                alerts = list(frame["AlertNamesPassed"])

                primary_neutrino = dataclasses.get_most_energetic_primary(mc_tree)

                if primary_neutrino is None or not self.is_neutrino(primary_neutrino):
                    # if the primary neutrino is non-existent, or isn't even a neutrino, we don't care about it
                    continue

                interacting_neutrino = self.get_interacting_nu(primary_neutrino, mc_tree)
                vertex = self.get_vertex(interacting_neutrino)

                interaction_type = i3mc_wd["InteractionType"]

                _metadata += [{
                    "vertex": vertex,
                    "interact_type": interaction_type,
                    "alerts": alerts,
                    "weight": self.weight(index, group_id)
                }]

                index += 1

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
    def get_vertex(neutrino: I3Particle) -> [float]:
        """Get the vertex position of a neutrino."""
        stop_pos = neutrino.pos + neutrino.dir * neutrino.length
        vertex = [float(stop_pos.x), float(stop_pos.y), float(stop_pos.z)]
        return vertex

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

    def __init__(self, directory: str, group_id: int):
        self._directory = directory
        self._paths = listdir_absolute(self._directory)
        self.group_id = group_id
        self._metadata_directory = os.path.join(config.BASE_DIR, f"data/metadata/{group_id}")

        for path in self._paths:
            if not path.endswith(".i3.zst"):
                self._paths.remove(path)

        self._filenames = [os.path.basename(path) for path in self._paths]

    def __repr__(self) -> str:
        _repr = self._directory
        return _repr

    def __iter__(self):
        for obj in self.objects():
            yield obj

    def objects(self):
        for path in self._paths:
            yield I3File(path)

    def extract_metadata(self):
        job = HTCondorJob(
            self._directory, self._metadata_directory, "scripts/extract_i3_metadata.py", ".i3.zst", ".json"
        )
        job.configure()
        job.submit(monitor=True)


    def get_p_frame_count(self):
        job = HTCondorJob(
            self._directory, os.path.join(config.BASE_DIR, f"data/frame_counts/{self.group_id}/physics/"),
            "scripts/physics_frame_count.py", ".i3.zst", ".json"
        )
        job.configure()
        job.submit(monitor=True)

    def generate_weight_config_file(self):
        base_dir = os.path.join(config.BASE_DIR, f"data/frame_counts/{self.group_id}/physics")

        with open(os.path.join(config.BASE_DIR, f"data/config/{self.group_id}/merge_order.{self.group_id}.json"), "r") as merge_order_file:
            files = json.load(merge_order_file)

        index = 0
        data = {}
        for file in files:
            with open(os.path.join(base_dir, str(os.path.basename(file).replace("hdf5", "json"))), "r") as size_file:
                content = json.load(size_file)
                data[os.path.basename(file).replace(".hdf5", "")] = {"size": content["size"], "index_start": index}
                index += content["size"]

        with open(os.path.join(config.BASE_DIR, f"data/config/21220/config.{self.group_id}.json"), "w+") as config_file:
            json.dump(data, config_file, indent=4)

    def load_metadata(self) -> dict:
        _metadata = {}
        for i, file in tqdm.tqdm(enumerate(self._metadata_directory)):
            with open(file, 'r') as f:
                _metadata[i] = json.load(f)
        return _metadata

    def to_hdf5(self) -> None:
        job = HTCondorJob(
            self._directory, os.path.join(config.BASE_DIR, f"data/hdf5/{self.group_id}"),
            "scripts/i3_to_hdf5.py", ".i3.zst", ".hdf5"
        )
        job.configure()
        job.submit(monitor=True)

    def get_alert_rate(self, alert: str):
        def point_in_bounds(point):
            vertices = [
                (-500., -500., -500.),  # Bottom face
                (500., -500., -500.),
                (500., 500., -500.),
                (-500., 500., -500.),
                (-500., -500., 500.),  # Top face
                (500., -500., 500.),
                (500., 500., 500.),
                (-500., 500., 500.)
            ]

            delaunay = Delaunay(vertices)
            return delaunay.find_simplex(point) >= 0


        count = 0
        for file in tqdm.tqdm(os.listdir(self._metadata_directory)):
            with open(os.path.join(self._metadata_directory, file), "r") as metadata_file:
                metadata = json.load(metadata_file)

            for entry in metadata:
                if not alert in entry["alerts"]:
                    continue

                if not point_in_bounds(entry["vertex"]):
                    continue

                count += entry["weight"]

        rate_file_path = os.path.join(config.BASE_DIR, f"data/events/{self.group_id}/{alert.lower()}/rate.{alert.lower()}.{self.group_id}.json")

        with open(rate_file_path, "w+") as rate_file:
            json.dump({"rate": count}, rate_file, indent=4)

        return count

    def plot_vertices(self, interact_type: list, projection: bool=False, histogram: bool=False, d: int=1) -> None:
        metadata = self.load_metadata()
        data = []
        counter = 0

        print("\n\nParsing...")
        for j, entry in tqdm.tqdm(metadata.items()):
            for packet in entry:
                if any([int(packet["interact_type"]) == n for n in interact_type]):
                    data.append([packet["vertex"]]) # weights[counter] ]
                counter += 1

        fig = PointCloud3D(data)

        path = os.path.join(config.BASE_DIR, "data/plots/vertices.html")

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
