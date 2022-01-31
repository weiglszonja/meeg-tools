import os
import unittest

from mne import datasets
from mne.io import read_raw

from meeg_tools.utils.epochs import create_epochs


class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_file_path = os.path.join(
            datasets.sample.data_path(), "MEG", "sample", "sample_audvis_raw.fif"
        )

        self.raw = read_raw(fname=self.raw_file_path)
        self.epochs = create_epochs(self.raw).load_data().pick_types(eeg=True)

    def test_prepare_epochs_for_ica(self):
        from meeg_tools.preprocessing import prepare_epochs_for_ica

        assert not any(self.epochs.drop_log)
        epochs_faster = prepare_epochs_for_ica(epochs=self.epochs)

        bad_epochs_indices = [
            drop_ind for drop_ind, drop in enumerate(epochs_faster.drop_log) if drop
        ]
        assert len(bad_epochs_indices) == 24
        assert bad_epochs_indices == [
            14,
            30,
            60,
            100,
            113,
            123,
            188,
            218,
            219,
            220,
            221,
            231,
            235,
            236,
            237,
            238,
            242,
            243,
            244,
            267,
            268,
            269,
            270,
            271,
        ]

    def test_run_ica(self):
        from meeg_tools.preprocessing import run_ica
        from meeg_tools.utils.config import settings

        settings["ica"]["n_components"] = 32
        settings["ica"]["decim"] = 3
        ica = run_ica(epochs=self.epochs)

        assert ica.exclude == [0]
        assert ica.n_components == 32
        assert "eog" not in self.epochs.get_channel_types()

    def test_run_autoreject(self):
        from meeg_tools.preprocessing import run_autoreject

        reject_log = run_autoreject(epochs=self.epochs, subset=False)
        epochs_autoreject = self.epochs.copy().drop(
            reject_log.report, reason="AUTOREJECT"
        )

        assert len(epochs_autoreject) < len(self.epochs)

    def test_get_noisy_channels(self):
        from meeg_tools.preprocessing import get_noisy_channels

        bads = get_noisy_channels(epochs=self.epochs, with_ransac=False)

        assert len(bads) == 9

    def test_interpolate_bad_channels(self):
        from meeg_tools.preprocessing import interpolate_bad_channels

        bads = ["EEG 025", "EEG 019"]

        epochs = interpolate_bad_channels(epochs=self.epochs, bads=bads)

        assert "EEG 025, EEG 019" in epochs.info["description"]

        epochs = interpolate_bad_channels(epochs=self.epochs, bads=[])

        assert "EEG 025, EEG 019" not in epochs.info["description"]
