[![Pylint](https://github.com/weiglszonja/eeg-preprocessing/actions/workflows/workflow.yml/badge.svg)](https://github.com/weiglszonja/eeg-preprocessing/actions/workflows/workflow.yml)
[![PyPI version](https://badge.fury.io/py/eeg-preprocessing.svg)](https://badge.fury.io/py/eeg-preprocessing)
[![GitHub license](https://img.shields.io/github/license/weiglszonja/eeg-preprocessing)](https://github.com/weiglszonja/eeg-preprocessing/blob/master/LICENSE)

# eeg-preprocessing

A semiautomatic framework for preprocessing EEG data.

# Overview

The eeg-preprocessing package serves as a cookbook for preprocessing EEG
signals in a semiautomatic and reproducible way. The general use-case of the
package is to use it from a Jupyter notebook. The
`tutorials` folder contains a sample notebook that demonstrates data operations
such as loading and writing the data along with the transformation steps that
are described in the Background section.

# Installation

Install the latest version from PyPI into an existing environment:

```bash
$ pip install eeg_preprocessing
```
Since this project is under development, I would recommend installing it
from source in editable mode with pip:

```bash
$ git clone https://github.com/weiglszonja/eeg-preprocessing.git
$ cd eeg-preprocessing
$ pip install -e .
```

Alternatively, you can install it in a new virtual environment with conda:

```bash
$ git clone https://github.com/weiglszonja/eeg-preprocessing.git
$ conda env create -f eeg-preprocessing/make_conda_env.yml
$ source activate eeg_preprocessing
```

# Background

Electroencephalography (EEG) measures neural activity by recording electrical
signals at the level of populations of neurons. The signals that are recorded
from multiple sensors are inherently contaminated by noise. Preprocessing aims
to attenuate noise in the EEG data without removing meaningful signals in the
process.

The eeg-preprocessing package aims to serve as a semiautomatic and reproducible
framework for preprocessing EEG signals prior to time-frequency-based analyses.
It minimizes the manual steps required to clean the data based on visual
inspection and reduces the number of choices that depend on the researcher for
rejecting segments of data or interpolation of electrodes. This package
utilizes modules from mne-Python (Gramfort et al., 2013), a popular open-source
Python package for working with neurophysiological data. For automated
rejection of bad data spans and interpolation of bad electrodes it uses the
Autoreject (Jas et al., 2017) and the Random Sample Consensus (RANSAC) (
Bigdely-Shamlo et al., 2015) packages.

The general use-case of the package is to use it from a Jupyter notebook.
The `tutorials` folder contains a sample notebook that demonstrates data
operations such as loading and writing the data along with the transformation
steps that are described below.

In order to remove high-frequency artefacts and low-frequency drifts, a
zero-phase band-pass filter (0.5 - 45 Hz) is applied to the continuous data
using mne-Python. This temporal filter adapts the filter length and transition
band size based on the cutoff frequencies. The lower and upper cutoff
frequencies can be changed in the configuration file (config.py) located at the
utils folder.

Subsequently, the filtered data is segmented into nonoverlapping segments (
epochs) to facilitate analyses. The default duration of epochs is one seconds,
however it can be changed in the configuration file.

The removal of bad data segments is done in three steps. First, epochs are
rejected based on a global threshold on the z-score (> 3) of the epoch variance
and amplitude range. To further facilitate the signal-to-noise ratio,
independent components analysis (ICA) is applied to the epochs. ICA is a
source-separation technique that decomposes the data into a set of components
that are unique sources of variance in the data. The number of components and
the algorithm to use can be specified in the configuration file. The default
method is the infomax algorithm that finds independent signals by maximizing
entropy as described by (Bell & Sejnowski, 1995), (Nadal & Parga, 1999).
Components containing blink artefacts are automatically identified using
mne-Python. The interactive visualization of ICA sources lets the user decide
which components should be rejected based on their topographies, time-courses
or frequency spectra. The number of components that were removed from the data
are documented in the “description” field of the epochs instance “info”
structure. The final step of epoch rejection is to apply Autoreject (Jas et
al., 2017) on the ICA cleaned data. Autoreject uses unsupervised learning to
estimate the rejection threshold for the epochs. In order to reduce computation
time that increases with the number of segments and channels, autoreject can be
fitted on a representative subset of epochs (25% of total epochs). Once the
parameters are learned, the solution can be applied to any data that contains
channels that were used during fit.

The final step of preprocessing is to find and interpolate outlier channels.
The Random Sample Consensus (RANSAC) algorithm (Fischler & Bolles, 1981)
selects a random subsample of good channels to make predictions of each channel
in small non-overlapping 4 seconds long time windows. It uses a method of
spherical splines (Perrin et al., 1989) to interpolate the bad sensors.

 Additionally, the EEG reference can be changed to a “virtual reference” that 
 is the average of all channels using mne-Python.


# Usage

The `tutorials` folder contains a sample jupyter notebook that demonstrates the
preprocessing pipeline. You can follow the instructions in the notebook. Note
that the custom method for loading raw EEG data expects BrainVision (.vhdr) and
EDF (.edf) files. However, importing data from other formats can be done with
mne-Python.
See [this](https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html)
documentation for further details. you can use the data loading utilities from
mne-Python.

```bash
$ jupyter notebook tutorials/run_preprocessing_tutorial.ipynb
```

Or you can import the methods into your own project and setup the pipeline with
a Python script:

```python
import os

from eeg_preprocessing.preprocessing import prepare_epochs_for_ica, run_ica, \
    run_autoreject, run_ransac
from eeg_preprocessing.utils.raw import read_raw_measurement
from eeg_preprocessing.utils.epochs import create_epochs


def run_pipeline(source: str):
    target_path = os.path.join(source, 'preprocessed')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    files = [file for file in os.listdir(source) if
             file.endswith(('.edf', '.vhdr', '.fif.gz'))]

    for file in files:
        raw = read_raw_measurement(raw_file_path=os.path.join(source, file),
                                   add_info=False)
        print(raw.info)

        # create epochs from filtered continuous data
        epochs = create_epochs(raw=raw)

        # initial rejection of bad epochs
        epochs_faster = prepare_epochs_for_ica(epochs=epochs)

        ica = run_ica(epochs=epochs_faster)
        # uncomment the line below to visualize ICA sources
        # block=True halts the execution of code until the plot is closed
        #ica.plot_sources(epochs_faster, start=0, stop=10, block=True)
        ica.apply(epochs_faster)
        epochs_faster.info['description'] = f'n_components: {len(ica.exclude)}'

        reject_log = run_autoreject(epochs_faster, n_jobs=11, subset=False)
        epochs_autoreject = epochs_faster.copy().drop(reject_log.bad_epochs,
                                                      reason='AUTOREJECT')

        epochs_ransac = run_ransac(epochs_autoreject)

        # set average reference
        epochs_ransac.set_eeg_reference('average', projection=True)

        # save clean epochs
        fid = epochs_autoreject.info['fid']
        epochs_clean_fname = f'{fid}_ICA_autoreject_ransac'
        postfix = '-epo.fif.gz'
        epochs_ransac.save(
            os.path.join(target_path, f'{epochs_clean_fname}{postfix}'),
            overwrite=False)


if __name__ == '__main__':
    run_pipeline(source='/Volumes/crnl-memo-hd/EEG')

```

## Contribution

This project is under development; comments are all welcome and encouraged!
Suggestions related to this project can be made with opening an
[issue](https://github.com/weiglszonja/eeg-preprocessing/issues/new)
at the issue tracker of the project. Contributions and enhancements to the code
can be made by forking the project first; committing changes to the forked
project and then opening a pull request from the forked branch to the master
branch of eeg-preprocessing.


## References

Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to
blind separation and blind deconvolution. Neural Computation, 7(6), 1129–1159.
Bigdely-Shamlo, N., Mullen, T., Kothe, C.,

Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline: standardized
preprocessing for large-scale EEG analysis. In Frontiers in Neuroinformatics (
Vol. 9). https://doi.org/10.3389/fninf.2015.00016

Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus. In
Communications of the ACM (Vol. 24, Issue 6, pp. 381–395)
. https://doi.org/10.1145/358669.358692

Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D.,
Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (
2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7,
267.

Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017).
Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159,
417–429.

Nadal, J.-P., & Parga, N. (1999). SENSORY CODING: INFORMATION MAXIMIZATION AND
REDUNDANCY REDUCTION. In Neuronal Information Processing (pp. 164–171)
. https://doi.org/10.1142/9789812818041_0008

Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F. (1989). Spherical
splines for scalp potential and current density mapping. Electroencephalography
and Clinical Neurophysiology, 72(2), 184–187.
