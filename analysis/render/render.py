import numpy as np
import matplotlib.pyplot as plt


class Histogram:

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