########################################################################################################################
# imports
from scipy.constants import point

from scipy.spatial import KDTree
from scipy.stats import trim_mean

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product

########################################################################################################################

class FeatureDistribution:

    def __init__(self, data: np.array, features: list, bins=int) -> None:
        self.fig = make_subplots(rows=2, cols=2, start_cell="top-left")
        self.data = data
        self.bins = bins
        self.features = features

    def populate(self) -> None:
        for i, [j, k] in enumerate(product([1, 2], repeat=2)):
            self.fig.add_trace(
                go.Histogram(
                    x=self.data[:, i], nbinsx=self.bins
                ), row=j, col=k
            )

            self.fig.update_yaxes(type='log', title_text="Pulses", row=j, col=k)
            self.fig.update_xaxes(title_text=f"{self.features[i].upper().replace('_', ' ')}", row=j, col=k)

    def save(self, path: str) -> None:
        self.fig.write_html(path, include_plotlyjs="cdn")


class FeatureDistribution3D:

    def __init__(self, data: np.array) -> None:
        self.fig = go.Figure()

        data = np.array(data, dtype=float)

        # Filter out rows with NaN
        mask = ~np.isnan(data).any(axis=1)

        clusters = self.get_clusters(data[mask], 0.01)

        clustered_data = {}
        for cluster in clusters:
            ave_t = trim_mean(cluster[:, 3], 0.1)
            x, y, z = cluster[0, 0], cluster[0, 1], cluster[0, 2]
            clustered_data[(x, y, z, ave_t)] = len(cluster)

        max_count = max(clustered_data.values())
        for k in clustered_data:
            clustered_data[k] /= max_count

        self.data = clustered_data

    @staticmethod
    def get_clusters(data, radius):
        xyz = data[:, :3]
        remaining_mask = np.ones(len(data), dtype=bool)

        tree = KDTree(xyz)
        clusters = []

        while remaining_mask.any():
            # Get index of first remaining point
            seed_idx = np.argmax(remaining_mask)
            seed_point = xyz[seed_idx]

            # Find all neighbors within radius
            neighbor_indices = tree.query_ball_point(seed_point, radius)
            neighbor_indices = [i for i in neighbor_indices if remaining_mask[i]]

            # Extract and save cluster
            cluster = data[neighbor_indices]
            clusters.append(cluster)

            # Mark these points as processed
            remaining_mask[neighbor_indices] = False

        return clusters

    def populate(self) -> None:
        point_data = np.array(list(self.data.keys()))
        count_data = np.array(list(self.data.values()))

        self.fig.add_trace(
            go.Scatter3d(
                x=point_data[:, 0],
                y=point_data[:, 1],
                z=point_data[:, 2],
                mode="markers",
                marker=dict(
                    size=count_data * 10 + 2,
                    color=point_data[:, 3],
                    colorscale="blackbody"
                )
            )
        )

    def save(self, path: str) -> None:
        self.fig.write_html(path, include_plotlyjs="cdn")


class EventDetailHist:

    def __init__(self):
        self._fig = plt.figure(figsize=(10, 5))
        self._ax = [self._fig.add_subplot(121), self._fig.add_subplot(122)]
        self._bins = [np.linspace(1, 5, 50), np.linspace(-1, 1, 20)]  # [log(charge), cos(zenith)]

        # label axes
        self._ax[0].set_ylabel('Number of Events')
        self._ax[0].set_xlabel(r'log$_{10}$(charge)')
        self._ax[0].set_yscale("log")

        self._ax[1].set_ylabel('Number of Events')
        self._ax[1].set_xlabel(r'cos($\theta$)')

    def populate(self, charge: np.ndarray, zenith: np.ndarray, weights):
        self._ax[0].hist(np.log10(charge), weights=weights, bins=self._bins[0], alpha=0.5)
        self._ax[1].hist(np.cos(zenith), bins=self._bins[1], alpha=0.5)

    def save(self, path):
        plt.tight_layout()
        self._fig.savefig(path, dpi=300)


class SimweightHist:

    def __init__(self):
        self._fig: plt.Figure = plt.figure(figsize=(10, 5))
        self._ax: plt.Axes = self._fig.add_subplot(111)

        self._bins = np.geomspace(1e2, 1e8, 50)

        self._ax.loglog()

        self._ax.set_xlabel("Primary Energy [GeV]")
        self._ax.set_ylabel("Event Rate [Hz]")

        self._ax.set_xlim(self._bins[0], self._bins[-1])
        self._ax.set_ylim(1e-8, 2e-5)

    def populate(self, primary_energy: np.ndarray, weight: np.ndarray):
        self._ax.hist(primary_energy, weights=weight, bins=self._bins, histtype="step", color="blue")

    def save(self, path):
        plt.tight_layout()
        self._fig.savefig(path, dpi=300, format="png")


class PointCloud3D:

    def __init__(self, data: list[list]):
        self._points = [d[0] for d in data]
        self._weights = [d[1] for d in data]

        self._x = [point[0] for point in self._points]
        self._y = [point[1] for point in self._points]
        self._z = [point[2] for point in self._points]
        self._r = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(self._x, self._y)]

        self._xy_color = [x * y for x, y in zip(self._x, self._y)]
        self._rz_color = [r * z for r, z in zip(self._r, self._z)]

    def plot_3d(self, path: str):
        # initialize figure
        _fig = go.Figure(data=[
            go.Scatter3d(x=self._x, y=self._y, z=self._z, mode="markers", marker={"size": 3})
        ])

        # label axes
        _fig.update_layout(scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ))

        # save the file (reference cdn instead of including plotly.js)
        _fig.write_html(path, include_plotlyjs="cdn")

    def plot_2d_projections(self, path: str):
        _fig = make_subplots(rows=1, cols=2, subplot_titles=("X vs Y", "R vs Z"))

        _fig.add_trace(
            go.Scatter(x=self._x, y=self._y, mode="markers", marker={"size": 3}),
            row=1, col=1
        )
        _fig.add_trace(
            go.Scatter(x=self._r, y=self._z, mode="markers", marker={"size": 3}),
            row=1, col=2
        )

        _fig.update_xaxes(title_text="X", row=1, col=1)
        _fig.update_yaxes(title_text="Y", row=1, col=1)

        _fig.update_xaxes(title_text="R", row=1, col=2)
        _fig.update_yaxes(title_text="Z", row=1, col=2)

        _fig.update_layout(height=600, width=1200)

        _fig.write_html(path, include_plotlyjs="cdn")

    def plot_1d_histograms(self, path: str):
        # initialize figure
        _fig = make_subplots(rows=2, cols=2, subplot_titles=("X", "Y", "Z", "R"))

        # label axes
        _fig.update_xaxes(title_text="X", row=1, col=1)
        _fig.update_xaxes(title_text="Y", row=1, col=2)
        _fig.update_xaxes(title_text="Z", row=2, col=1)
        _fig.update_xaxes(title_text="R", row=2, col=2)

        _fig.update_yaxes(title_text="Count", row=1, col=1)
        _fig.update_yaxes(title_text="Count", row=1, col=2)
        _fig.update_yaxes(title_text="Count", row=2, col=1)
        _fig.update_yaxes(title_text="Count", row=2, col=2)

        # populate with data
        _fig.add_trace(
            go.Histogram(x=self._x, nbinsx=200),
            row=1, col=1
        )
        _fig.add_trace(
            go.Histogram(x=self._y, nbinsx=200),
            row=1, col=2
        )
        _fig.add_trace(
            go.Histogram(x=self._z, nbinsx=200),
            row=2, col=1
        )
        _fig.add_trace(
            go.Histogram(x=self._r, nbinsx=200),
            row=2, col=2
        )

        # save the file (reference cdn instead of including plotly.js)
        _fig.write_html(path, include_plotlyjs="cdn")

    def plot_2d_histograms(self, path: str):
        # initialize figure
        _fig = make_subplots(rows=1, cols=2, subplot_titles=("X vs Y", "R vs Z"), horizontal_spacing=0.2)

        # label axes
        _fig.update_xaxes(title_text="X", row=1, col=1)
        _fig.update_xaxes(title_text="R", row=1, col=2)

        _fig.update_yaxes(title_text="Y", row=1, col=1)
        _fig.update_yaxes(title_text="Z", row=1, col=2)

        colorscale = "Blackbody"  # Perceptually uniform colorscale

        # Compute histogram data for X vs Y
        hist_xy, x_edges_xy, y_edges_xy = np.histogram2d(
            self._x, self._y, weights=self._weights, bins=500
        )
        hist_xy_log = np.log10(hist_xy + 1)  # Apply logarithmic scaling

        # Compute histogram data for R vs Z
        hist_rz, x_edges_rz, y_edges_rz = np.histogram2d(
            self._r, self._z, weights=self._weights, bins=500
        )
        hist_rz_log = np.log10(hist_rz + 1)  # Apply logarithmic scaling

        # Add 2D histogram for X vs Y
        _fig.add_trace(
            go.Heatmap(
                x=x_edges_xy[:-1],
                y=y_edges_xy[:-1],
                z=hist_xy_log.T,
                colorscale=colorscale,
                colorbar=dict(
                    exponentformat="power",
                    x=0.4
                ),
            ),
            row=1,
            col=1,
        )

        # Add 2D histogram for R vs Z
        _fig.add_trace(
            go.Heatmap(
                x=x_edges_rz[:-1],
                y=y_edges_rz[:-1],
                z=hist_rz_log.T,
                colorscale=colorscale,
                colorbar=dict(
                    exponentformat="power",
                    x=1.0
                ),
            ),
            row=1,
            col=2,
        )

        # Adjust axis ranges (optional, based on dataset)
        _fig.update_xaxes(range=[-9000, 9000], row=1, col=1)
        _fig.update_yaxes(range=[-9000, 9000], row=1, col=1)

        _fig.update_xaxes(range=[0, 15000], row=1, col=2)
        _fig.update_yaxes(range=[-6000, 1950], row=1, col=2)

        # Set figure dimensions
        _fig.update_layout(height=700, width=1400)

        # save the file (reference cdn instead of including plotly.js)
        _fig.write_html(path, include_plotlyjs="cdn")

########################################################################################################################
