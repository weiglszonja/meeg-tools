import os
import unittest

import numpy as np

from mne import datasets
from mne.io import read_raw


class TestEpochs(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_file_path = os.path.join(
            datasets.sample.data_path(), "MEG", "sample", "sample_audvis_raw.fif"
        )

        self.raw = read_raw(fname=self.raw_file_path)

    def test_create_epochs_without_annotations(self):
        from meeg_tools.utils.epochs import create_epochs
        from meeg_tools.utils.config import settings

        settings["epochs"]["duration"] = 2

        epochs = create_epochs(self.raw)
        assert (epochs.events[1][0] - epochs.events[0][0]) == int(
            epochs.info["sfreq"] * settings["epochs"]["duration"]
        )

        epochs.load_data()
        # check epochs are not overlapping
        n_epochs = epochs.get_data().shape[0]
        for epoch_num in range(n_epochs - 1):
            assert not np.array_equal(
                epochs.get_data()[epoch_num, ...], epochs.get_data()[epoch_num + 1, ...]
            )

        assert int(len(self.raw) / self.raw.info["sfreq"] / 2) == len(epochs)
        assert epochs.event_id == {"1": 1}

    def test_create_epochs_from_events(self):
        from meeg_tools.utils.epochs import create_epochs_from_events
        from meeg_tools.utils.config import settings

        settings["epochs"]["start_time"] = -0.2
        settings["epochs"]["end_time"] = 0.5

        epochs = create_epochs_from_events(self.raw, event_ids=[1, 2, 3])
        epochs.load_data()

        # check epochs are not overlapping
        n_epochs = epochs.get_data().shape[0]
        for epoch_num in range(n_epochs - 1):
            assert not np.array_equal(
                epochs.get_data()[epoch_num, ...], epochs.get_data()[epoch_num + 1, ...]
            )

        assert len(epochs) == 218
        assert epochs.event_id == {"1": 1, "2": 2, "3": 3}
        assert np.round(epochs[0].times[0], 2) == -0.2
        assert np.round(epochs[0].times[-1], 2) == 0.5
