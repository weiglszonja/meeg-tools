import os
import unittest

import numpy as np
from mne import datasets, Annotations
from mne.io import read_raw


class TestEpochs(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_file_path = os.path.join(datasets.sample.data_path(), 'MEG',
                                          'sample', 'sample_audvis_raw.fif')

        self.raw = read_raw(fname=self.raw_file_path)

    def test_create_epochs_without_annotations(self):
        from eeg_preprocessing.utils.epochs import create_epochs
        from eeg_preprocessing.utils.config import settings

        settings['epochs']['duration'] = 2

        epochs = create_epochs(self.raw)
        assert (epochs.events[1][0] - epochs.events[0][0]) == int(
            epochs.info['sfreq'] * settings['epochs']['duration'])

        # check epochs are not overlapping
        n_epochs = epochs.get_data().shape[0]
        for epoch_num in range(n_epochs - 1):
            assert not np.array_equal(
                epochs.get_data()[epoch_num, ...],
                epochs.get_data()[epoch_num + 1, ...])

        assert int(len(self.raw) / self.raw.info['sfreq'] / 2) == len(epochs)
        assert epochs.event_id == {'1': 1}

    def test_get_events_from_annotations(self):
        from eeg_preprocessing.utils.epochs import get_events_from_annotations
        from eeg_preprocessing.utils.config import settings

        settings['epochs']['duration'] = 1

        annotations = Annotations(onset=[3, 10, 10.5, 11, 22],
                                  duration=[1, 0.5, 0.25, 1, 1],
                                  description=[1, 2, 3, 1, 2])

        self.raw.set_annotations(annotations)

        events = get_events_from_annotations(self.raw)

        assert len(events) == 21
        assert len(events[events[..., 2] == 1]) == 18
        assert len(events[events[..., 2] == 2]) == 2
        assert len(events[events[..., 2] == 3]) == 1

    def test_create_epochs_with_annotations(self):
        from eeg_preprocessing.utils.epochs import create_epochs
        from eeg_preprocessing.utils.config import settings

        settings['epochs']['duration'] = 1

        annotations = Annotations(onset=[3, 10, 10.5, 11, 22],
                                  duration=[1, 0.5, 0.25, 1, 1],
                                  description=[1, 2, 3, 1, 2])

        self.raw.set_annotations(annotations)

        epochs = create_epochs(self.raw)

        # check epochs are not overlapping
        n_epochs = epochs.get_data().shape[0]
        for epoch_num in range(n_epochs - 1):
            assert not np.array_equal(
                epochs.get_data()[epoch_num, ...],
                epochs.get_data()[epoch_num + 1, ...])

        assert len(epochs) == 21
        assert epochs.event_id == {'1': 1, '2': 2, '3': 3}
        assert len(epochs[0]) == 1

    def test_create_epochs_without_events_to_exclude(self):
        from eeg_preprocessing.utils.epochs import create_epochs
        from eeg_preprocessing.utils.config import settings

        settings['epochs']['duration'] = 1

        annotations = Annotations(onset=[3, 10, 10.5, 11, 22],
                                  duration=[1, 0.5, 0.25, 1, 1],
                                  description=[1, 2, 3, 1, 2])

        self.raw.set_annotations(annotations)

        epochs = create_epochs(self.raw, events_to_exclude=[2, 5])

        assert len(epochs) == 19
        assert epochs.event_id == {'1': 1, '3': 3}
