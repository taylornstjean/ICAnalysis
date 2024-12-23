import numpy as np
import matplotlib.pyplot as plt


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

    def populate(self, charge: np.ndarray, zenith: np.ndarray):
        self._ax[0].hist(np.log10(charge), bins=self._bins[0], alpha=0.5)
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
        self._ax.set_ylim(1e-8, 2e-6)

    def populate(self, primary_energy: np.ndarray, weight: np.ndarray):
        self._ax.hist(primary_energy, weights=weight, bins=self._bins, histtype="step", color="blue")

    def save(self, path):
        plt.tight_layout()
        self._fig.savefig(path, dpi=300, format="png")