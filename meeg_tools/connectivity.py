"""
This module contains functions that can be used to perform connectivity analysis
in the sensor space using MNE-Python.
https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity.html
https://mne.tools/mne-connectivity/stable/auto_examples/sensor_connectivity.html#sphx-glr-auto-examples-sensor-connectivity-py
"""
from collections import Iterable

from matplotlib import pyplot as plt

from mne import Epochs
from mne_connectivity import spectral_connectivity_epochs, SpectralConnectivity
from mne.preprocessing import compute_current_source_density


def compute_connectivity(epochs: Epochs, config: dict) -> SpectralConnectivity:
    """
    Compute channel level connectivity matrix from Epochs instance.
    Returns the computed connectivity matrix (n_freqs, n_signals, n_signals).

    :return: np.ndarray con: The computed connectivity matrix with a shape of
    (n_freqs, n_signals, n_signals).

    See Also
    --------
    For frequency-decomposition and frequency bin reference:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfftfreq.html
    """
    # spacing between frequency bins
    # spacing = np.round(epochs.info["sfreq"] / epochs.get_data().shape[-1], 2)
    freq_bands = sorted(config["bands"].values())

    low_cutoff = tuple(freq[0] for freq in freq_bands)
    high_cutoff = tuple(freq[1] for freq in freq_bands)
    # high_cutoff = tuple(
    #     freq[1] - spacing if freq != max(freq_bands) else freq[1]
    #     for freq in freq_bands
    # )

    if config["use_laplace"]:
        epochs_csd = compute_surface_laplacian(epochs=epochs, verbose=False)
    else:
        epochs_csd = epochs.copy()

    con = spectral_connectivity_epochs(
        epochs_csd,
        method=config["method"],
        mode='multitaper',
        sfreq=epochs.info["sfreq"],
        fmin=low_cutoff,
        fmax=high_cutoff,
        faverage=True,
        tmin=config["tmin"],
        tmax=config["tmax"],
        n_jobs=11,
        verbose=False)
    # con, _, _, _, _ = spectral_connectivity(
    #     data=epochs_csd,
    #     method=kwargs["method"],
    #     sfreq=epochs.info["sfreq"],
    #     mode=kwargs["mode"],
    #     fmin=low_cutoff,
    #     fmax=high_cutoff,
    #     faverage=kwargs["faverage"],
    #     n_jobs=kwargs["n_jobs"],
    #     verbose=True,
    # )
    #
    # # from shape of (n_signals, n_signals, n_freqs) to
    # # (n_freqs, n_signals, n_signals)
    # con = np.transpose(con, (2, 0, 1))
    # con = abs(con)

    if all(isinstance(el, Iterable) for el in con.attrs["freqs_used"]):
        con.attrs["freqs_used"] = [freq for freq_arr in con.attrs["freqs_used"]
                                   for freq in freq_arr]

    # make sure all connectivity values are positive
    con.xarray.values = abs(con.xarray.values)

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
