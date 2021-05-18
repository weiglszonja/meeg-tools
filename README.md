# eeg-preprocessing
A semiautomatic framework for preprocessing EEG data.

# Setup
Clone the repository and setup a conda environment:
```bash
$ git clone https://github.com/weiglszonja/eeg-preprocessing.git
$ conda env create -f eeg-preprocessing/make_conda_env.yml
$ source activate eeg_preprocessing
```
Alternatively, install into an existing environment:
```bash
$ pip install eeg_preprocessing
```

# Usage
If you cloned the repository, there is a jupyter notebook with a pipeline setup 
that can be used for preprocessing data:
```bash
$ jupyter notebook notebooks/run_preprocessing.ipynb
```
Or you can import the methods into your own project and setup the pipeline for your own needs:
```python
from eeg_preprocessing import preprocessing
```
