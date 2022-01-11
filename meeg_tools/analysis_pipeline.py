"""
This module contains functions to perform time-frequency analysis on EEG/MEG dataset.
"""
import os
from pathlib import Path

import yaml
import pandas as pd
from tqdm import tqdm
from mne import read_epochs
from mne.utils import logger

from meeg_tools.time_frequency import compute_power, save_to_hdf5

# file path to configuration file
CONFIGURATION_FILE_PATH = "utils/config.yml"
# file path to preprocessed epochs
EPOCHS_FILE_PATH = "/Users/weian/Downloads/epochs_asrt"
# postfix to lookup when collecting the files
EPOCHS_FILE_POSTFIX = "autoreject_ransac-epo.fif.gz"


def run_tfr_pipeline(source: Path, target: Path, conf: dict):
    def _run_tfr(epochs_tfr):
        if mode == "power":
            power = compute_power(epochs=epochs_tfr, config=conf["power"])
            power.comment = file_path
            save_to_hdf5(power=power)
        else:
            pass

    files = sorted(list(source.rglob(f"*{EPOCHS_FILE_POSTFIX}")))

    if not len(files):
        logger.warning(
            f"There are no files with the expected {EPOCHS_FILE_POSTFIX}."
            f"Doing nothing ..."
        )
        return

    pbar = tqdm(sorted(files))
    for file in pbar:
        pbar.set_description("Processing %s" % file.stem)
        fid = "_".join(str(file.stem.replace(" ", "_")).split("_")[:3])

        epochs = read_epochs(os.path.join(source, file), preload=False, verbose=0)
        epochs.load_data().pick_types(**{config["analysis"]["picks"]: True})
        # make sure average ref is applied
        if not epochs.info["custom_ref_applied"]:
            epochs.load_data().set_eeg_reference("average")

        condition_names = conf["analysis"]["conditions"]
        mode = conf["analysis"]["mode"]
        if condition_names:
            if isinstance(epochs.metadata, pd.DataFrame):
                epochs.metadata = epochs.metadata.astype(str)
                if all(name in epochs.metadata.columns for name in condition_names):
                    for condition_values, _ in epochs.metadata.groupby(condition_names):
                        query = [
                            f'{a} == "{b}"'
                            for a, b in zip(condition_names, condition_values)
                        ]
                        query_str = " & ".join(query)
                        epochs_query = epochs[query_str]

                        file_name = f"{fid}_{'_'.join(condition_values)}"
                        file_path = target / file_name
                        if file_exists(file_path=f"{file_path}_{mode}-tfr.h5"):
                            pbar.update(1)
                            continue

                        _run_tfr(epochs_tfr=epochs_query)

        else:
            file_path = target / fid
            if file_exists(f'{file_path}_{conf["analysis"]["mode"]}-tfr.h5'):
                pbar.update(1)
                continue

            _run_tfr(epochs_tfr=epochs)

    pbar.close()


def read_config(file_name: str):
    cfg = {}
    if os.path.exists(file_name):
        with open(file_name, "r") as config_file:
            cfg = yaml.safe_load(config_file)
    else:
        logger.error(f"Configuration file ({file_name}) does not exist!")
    return cfg


def file_exists(file_path: str) -> bool:
    return os.path.exists(file_path) and not config["analysis"]["overwrite"]


if __name__ == "__main__":
    logger.info(f"\nReading configuration from {CONFIGURATION_FILE_PATH} file ...\n")
    config = read_config(CONFIGURATION_FILE_PATH)
    logger.info(yaml.dump(config))
    logger.info(
        f'\nRunning EEG/MEG {config["analysis"]["mode"]} analysis pipeline for '
        f'{config["analysis"]["picks"]} channels\n'
    )

    pipeline_in = Path(EPOCHS_FILE_PATH)
    if len(config["analysis"]["conditions"]):
        conditions = "-".join(config["analysis"]["conditions"]).upper()
        pipeline_out = (
            pipeline_in.parent
            / config["analysis"]["picks"]
            / config["analysis"]["mode"]
            / conditions
        )
    else:
        pipeline_out = (
            pipeline_in.parent
            / config["analysis"]["picks"]
            / config["analysis"]["mode"]
        )
    logger.info(f'Source directory is "{pipeline_in}"')
    logger.info(f'Output files are created at "{pipeline_out}"')

    if len(config["analysis"]["conditions"]):
        logger.info(
            f'Current run computes {config["analysis"]["mode"]} '
            f'for each {", ".join(config["analysis"]["conditions"])} '
            f"found in epochs metadata."
        )

    run_tfr_pipeline(source=pipeline_in, target=pipeline_out, conf=config)
