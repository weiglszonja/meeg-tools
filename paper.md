---
title: 'eeg-preprocessing: A framework for semiautomatic preprocessing of EEG data'
tags:
  - Python
  - Jupyter Notebook
  - EEG
  - preprocessing
  - pipeline
  - mne-Python
  - autoreject
  - RANSAC
authors:
  - name: Anna Szonja Weigl^[first author]
    orcid: 0000-0001-6615-1360
    affiliation: 1 
  - name: Dezso Nemeth
    orcid: 0000-0002-9629-5856
    affiliation: "1, 2, 3"
affiliations:
 - name: Lyon Neuroscience Research Center (CRNL), INSERM U1028, CNRS UMR5292, Université de Lyon 1, Université de Lyon, Lyon, France
   index: 1
 - name: Institute of Psychology, ELTE Eötvös Loránd University, Budapest, Hungary
   index: 2
 - name: Brain, Memory and Language Research Group, Institute of Cognitive Neuroscience and Psychology, Research Centre for Natural Sciences, Budapest, Hungary
   index: 3
date: 17 June 2021
bibliography: paper.bib

---

# Summary

Electroencephalography (EEG) measures neural activity by recording electrical
signals at the level of populations of neurons. The signals that are recorded
from multiple sensors are inherently contaminated by noise. Preprocessing aims
to attenuate noise in the EEG data without removing meaningful signals in the
process. Here, we present a semiautomatic pipeline which can prepare the data 
for functional connectivity or event related potential (ERP) analyses.

# Statement of need

The `eeg-preprocessing` package aims to serve as a semiautomatic and reproducible
framework for preprocessing EEG signals prior to time-frequency-based analyses.
It aids researchers from the (cognitive) neuroscience community to preprocess
signals without having a lot of expertise in working with EEG signals.
It reduces the manual steps required to clean the data based on visual
inspection, and the subjectivity in rejecting segments of data or 
interpolation of electrodes. This package
utilizes modules from `MNE-Python` [@10.3389/fnins.2013.00267], a popular open-source
Python package for working with neurophysiological data. For automated
rejection of bad data spans and interpolation of bad electrodes it uses the
`Autoreject` [@Jas2017] and the Random Sample Consensus (`RANSAC`)
[@10.3389/fninf.2015.00016] packages. The general use-case of the package is 
to use it from a `Jupyter Notebook`. The `tutorials` folder contains a Jupyter 
notebook that demonstrates data operations such as loading and writing the data
along with the transformation steps that are described in the Background section.


# Background

In order to remove high-frequency artefacts and low-frequency drifts, a
zero-phase band-pass filter (0.5 - 45 Hz) is applied to the continuous data
using `MNE-Python` [@10.3389/fnins.2013.00267]. This temporal filter adapts the filter 
length and transition band size based on the cutoff frequencies. 
The lower and upper cutoff frequencies can be changed in the configuration 
file (`config.py`) located at the utils folder.

Subsequently, the filtered data is segmented into non-overlapping segments (
epochs) to facilitate analyses. The default duration of epochs is one seconds,
however it can be changed in the configuration file.

The removal of bad data segments is performed in three steps. First, epochs are
rejected based on a global threshold on the z-score (> 3) of the epoch variance
and amplitude range. To further facilitate the signal-to-noise ratio,
independent components analysis (ICA) is applied to the epochs. ICA is a
source-separation technique that decomposes the data into a set of components
that are unique sources of variance in the data. The number of components and
the algorithm to use can be specified in the configuration file. The default
method is the infomax algorithm that finds independent signals by maximizing
entropy as described by [@Bell1995; @doi:10.1142/9789812818041_0008].
Components containing blink artefacts are automatically identified using
`MNE-Python`. The interactive visualization of ICA sources lets the user decide
which components should be rejected based on their topographies, time-courses
or frequency spectra. The number of components that were removed from the data
are documented in the “description” field of the epochs instance “info”
structure. The final step of epoch rejection is to apply `Autoreject` [@Jas2017]
on the ICA cleaned data. `Autoreject` uses unsupervised learning to
estimate the rejection threshold for the epochs. In order to reduce computation
time that increases with the number of segments and channels, autoreject can be
fitted on a representative subset of epochs (25% of total epochs). Once the
parameters are learned, the solution can be applied to any data that contains
channels that were used during fit.

The final step of preprocessing is to find and interpolate outlier channels.
The Random Sample Consensus (`RANSAC`) algorithm [@10.1145/358669.358692]
selects a random subsample of good channels to make predictions of each channel
in small non-overlapping 4 seconds long time windows. It uses a method of
spherical splines [@Perrin1989] to interpolate the bad sensors.

 Additionally, the EEG reference can be changed to a “virtual reference” that 
 is the average of all channels using `MNE-Python`.

# Acknowledgements

This research was supported by the National Brain Research Program 
(project 2017-1.2.1-NKP-2017-00002); Hungarian Scientific Research Fund 
(NKFIH-OTKA K 128016, to D.N.); IDEXLYON Fellowship of the University of 
Lyon as part of the Programme Investissements d'Avenir (ANR-16-IDEX-0005) 
(to D.N.). We thank Zsofia Zavecz for her useful comments and suggestions. 

# References