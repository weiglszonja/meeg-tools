"""
This module contains functions that can be used to perform connectivity analysis
in the sensor space using MNE-Python.
https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity.html
https://mne.tools/mne-connectivity/stable/auto_examples/sensor_connectivity.html#sphx-glr-auto-examples-sensor-connectivity-py
"""
import numpy as np
from matplotlib import pyplot as plt

from mne import Epochs
from mne.connectivity import spectral_connectivity
from mne.preprocessing import compute_current_source_density


def compute_connectivity(epochs: Epochs, **kwargs) -> np.ndarray:
    """
    Compute channel level connectivity matrix from Epochs instance.
    Returns the computed connectivity matrix (n_freqs, n_signals, n_signals).

    Args:
        str spectrum_mode: Valid estimation mode 'fourier' or 'multitaper'
        Epochs epochs: Epochs extracted from a Raw instance
        str method: connectivity estimation method
        int n_jobs: number of epochs to process in parallel
    :return: np.ndarray con: The computed connectivity matrix with a shape of
    (n_freqs, n_signals, n_signals).

    See Also
    --------
    For frequency-decomposition and frequency bin reference:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfftfreq.html
    """
    # spacing between frequency bins
    spacing = epochs.info["sfreq"] / epochs.get_data().shape[-1]

    low_cutoff = tuple(band[0] for band in kwargs["freqs"])
    high_cutoff = tuple(
        band[1] - spacing if band != max(kwargs["freqs"]) else band[1]
        for band in kwargs["freqs"]
    )

    epochs_csd = compute_surface_laplacian(epochs=epochs, verbose=False)
    con, _, _, _, _ = spectral_connectivity(
        data=epochs_csd,
        method=kwargs["method"],
        sfreq=epochs.info["sfreq"],
        mode=kwargs["mode"],
        fmin=low_cutoff,
        fmax=high_cutoff,
        faverage=kwargs["faverage"],
        n_jobs=kwargs["n_jobs"],
        verbose=True,
    )

    # from shape of (n_signals, n_signals, n_freqs) to
    # (n_freqs, n_signals, n_signals)
    con = np.transpose(con, (2, 0, 1))
    con = abs(con)

    return con


def compute_surface_laplacian(epochs: Epochs, verbose: bool = True) -> Epochs:
    """
    Performs the surface Laplacian transform on the Epochs instance
    For more information about the transform parameters please refer to
    Cohen, M. X. (2014). Analyzing neural time series data: theory and practice
    . MIT press. For more information about this function in particular,
    visit the MNE documentation at:
    https://mne.tools/dev/generated/mne.preprocessing.compute_current_source_density.html
    Parameters
    ----------
    epochs: the epochs to be transformed
    verbose: whether to visualize the power spectral densities before and after
    the Laplacian transform

    Returns
    -------
    Raw instance
    """

    epochs_csd = compute_current_source_density(epochs.copy())

    if verbose:
        fig, axes_subplot = plt.subplots(
            nrows=2, ncols=1, sharex="all", sharey="all", dpi=220
        )

        epochs.plot_psd(ax=axes_subplot[0], show=False, fmax=60)
        epochs_csd.plot_psd(ax=axes_subplot[1], show=False, fmax=60)
        fig.show()

    return epochs_csd
