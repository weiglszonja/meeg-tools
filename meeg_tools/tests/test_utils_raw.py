import os
import unittest

from mne import datasets


class TestRaw(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_file_path = os.path.join(
            datasets.sample.data_path(), "MEG", "sample", "sample_audvis_raw.fif"
        )

    def test_read_raw_measurement(self) -> None:
        from meeg_tools.utils.raw import read_raw_measurement

        raw = read_raw_measurement(raw_file_path=self.raw_file_path)
        assert raw.info["fid"] == "sample_audvis_raw"
        assert bool(raw.get_montage())

    def test_filter_raw(self) -> None:
        from mne.io import read_raw
        from meeg_tools.utils.raw import filter_raw
        from meeg_tools.utils.config import settings

        settings["bandpass_filter"]["low_freq"] = 1
        settings["bandpass_filter"]["high_freq"] = 30

        raw = read_raw(fname=self.raw_file_path)

        raw_filtered = filter_raw(raw)

        assert raw_filtered.info["highpass"] == 1.0
        assert raw_filtered.info["lowpass"] == 30.0

        settings["bandpass_filter"]["low_freq"] = 5
        settings["bandpass_filter"]["high_freq"] = 45

        raw_filtered = filter_raw(raw)

        assert raw_filtered.info["highpass"] == 5.0
        assert raw_filtered.info["lowpass"] == 45.0

        assert raw.info["highpass"] != raw_filtered.info["highpass"]
        assert raw.info["lowpass"] != raw_filtered.info["lowpass"]
