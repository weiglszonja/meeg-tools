"""
This module contains functions to perform time-frequency analysis on EEG/MEG dataset.
"""
import os
from pathlib import Path
import re

import yaml
import pandas as pd
from tqdm import tqdm
from mne import read_epochs
from mne.utils import logger

from meeg_tools.connectivity import compute_connectivity
from meeg_tools.time_frequency import compute_power, save_to_hdf5, compute_erp

# file path to configuration file
CONFIGURATION_FILE_PATH = "utils/config.yml"
# file path to preprocessed epochs
EPOCHS_FILE_PATH = "/Users/weian/Downloads/Raw_data/preprocessed/epochs_asrt_longer"
# postfix to lookup when collecting the files
EPOCHS_FILE_POSTFIX = "autoreject_ransac-epo.fif.gz"


def run():
    logger.info(f"\nReading configuration from {CONFIGURATION_FILE_PATH} file ...\n")
    config = read_config(CONFIGURATION_FILE_PATH)
    logger.info(yaml.dump(config))
    logger.info(
        f'\nRunning EEG/MEG {config["analysis"]["mode"]} analysis pipeline for '
        f'{config["analysis"]["picks"]} channels\n'
    )

    pipeline_in = Path(EPOCHS_FILE_PATH)
    current_folder_name = pipeline_in.name.replace("epochs",
                                                   config["analysis"]["mode"])
    if len(config["analysis"]["conditions"]):
        conditions = "-".join(config["analysis"]["conditions"]).upper()
        pipeline_out = (
                pipeline_in.parent
                / config["analysis"]["mode"]
                / current_folder_name
                / conditions
        )
    else:
        pipeline_out = (
                pipeline_in.parent
                / config["analysis"]["mode"]
                / current_folder_name
        )
    logger.info(f'Source directory is "{pipeline_in}"')
    logger.info(f'Output files are created at "{pipeline_out}"')

    if len(config["analysis"]["conditions"]):
        logger.info(
            f'Current run computes {config["analysis"]["mode"]} '
            f'for each {", ".join(config["analysis"]["conditions"])} '
            f"found in epochs metadata."
        )

    if not os.path.exists(pipeline_out):
        os.makedirs(pipeline_out)

    run_tfr_pipeline(source=pipeline_in, target=pipeline_out, conf=config)


def run_tfr_pipeline(source: Path, target: Path, conf: dict):
    def _file_exists(file_path: str) -> bool:
        return os.path.exists(file_path) and not conf["analysis"]["overwrite"]

    def _run_tfr(epochs_tfr):
        if mode == "power":
            power = compute_power(epochs=epochs_tfr, config=conf["power"])
            power.comment = file_path
            save_to_hdf5(power=power)
        elif mode == "erp":
            erp = compute_erp(epochs=epochs_tfr,
                              config=conf["erp"])
            erp.comment = str(file_path)
            erp.save(f"{file_path}_{mode}-{conf[mode]['postfix']}")
        elif mode == "con":
            con = compute_connectivity(epochs=epochs_tfr,
                                       config=conf["con"])
            con.attrs.update(comment=str(file_path),
                             ch_names=epochs_tfr.info["ch_names"],
                             sfreq=epochs_tfr.info["sfreq"],
                             ch_types=conf["analysis"]["picks"],)
            con.save(f"{file_path}_{conf[mode]['method']}-{conf[mode]['postfix']}")
        else:
            pass

    files = sorted(list(source.rglob(f"*{EPOCHS_FILE_POSTFIX}")))

    if not len(files):
        logger.warning(
            f"There are no files with the expected {EPOCHS_FILE_POSTFIX}."
            f"Doing nothing ..."
        )
        return

    condition_names = conf["analysis"]["conditions"]
    mode = conf["analysis"]["mode"]
    metadata = pd.DataFrame(columns=['fid', 'n_epochs'] + condition_names)
    pbar = tqdm(sorted(files))
    for file in pbar:
        pbar.set_description("Processing %s" % file.stem)
        fid = "_".join(str(file.stem.replace(" ", "_")).split("_")[:3])

        epochs = read_epochs(os.path.join(source, file), preload=False, verbose=0)
        epochs.load_data().pick_types(**{conf["analysis"]["picks"]: True})
        # make sure average ref is applied
        if not epochs.info["custom_ref_applied"]:
            epochs.load_data().set_eeg_reference("average")

        if condition_names:
            if isinstance(epochs.metadata, pd.DataFrame):
                epochs.metadata = epochs.metadata.astype(str)
                if all(name in epochs.metadata.columns for name in condition_names):
                    for condition_values, _ in epochs.metadata.groupby(condition_names):
                        if isinstance(condition_values, str):
                            condition_values = [condition_values]
                        query = [
                            f'{a} == "{b}"'
                            for a, b in zip(condition_names, condition_values)
                        ]
                        query_str = " & ".join(query)
                        epochs_query = epochs[query_str]

                        data = dict(fid=fid,
                                    n_epochs=len(epochs_query),
                                    **dict(zip(condition_names, condition_values)),
                                    **conf[mode])
                        metadata = metadata.append(data,
                                                   ignore_index=True)

                        file_name = f"{fid}_{'_'.join(condition_values)}"

                        floats = re.findall("[-+]?\d*\.\d+", file_name)
                        if floats:
                            for num in floats:
                                file_name = file_name.replace(num, str(int(float(num))))

                        file_path = target / file_name
                        if _file_exists(file_path=f"{file_path}_{mode}-{conf[mode]['postfix']}"):
                            pbar.update(1)
                            continue

                        _run_tfr(epochs_tfr=epochs_query)

        else:
            for event_id in epochs.event_id:
                epochs_per_event_id = epochs[event_id]

                metadata = metadata.append(dict(fid=fid,
                                                n_epochs=len(epochs_per_event_id),
                                                event_id=event_id,
                                                **conf[mode]),
                                           ignore_index=True)

                file_path = target / f"{fid}_{event_id}"
                if _file_exists(f"{file_path}_{mode}-{conf[mode]['postfix']}"):
                    pbar.update(1)
                    continue

                _run_tfr(epochs_tfr=epochs_per_event_id)

    pbar.close()

    metadata_file_path = os.path.join(target, f"{mode}_metadata.csv")
    metadata.to_csv(metadata_file_path, index=False)

    logger.info(f"Metadata file can be found at:\n{metadata_file_path}")

    logger.info("\n[PIPELINE FINISHED]")


def read_config(file_name: str):
    cfg = {}
    if os.path.exists(file_name):
        with open(file_name, "r") as config_file:
            cfg = yaml.safe_load(config_file)
    else:
        logger.error(f"Configuration file ({file_name}) does not exist!")
    return cfg


if __name__ == "__main__":
    run()
