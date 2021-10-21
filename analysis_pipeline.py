import argparse
import os
from pathlib import Path

import pandas as pd
from mne import read_epochs
from meeg_tools.time_frequency import compute_power

from tqdm import tqdm
from mne.utils import logger

from meeg_tools.utils.config import analysis

# file path to preprocessed epochs
epochs_path = '/Volumes/crnl-memo-hd/preprocessed/epochs_asrt'
# postfix to lookup when collecting the files
postfix = 'autoreject_ransac-epo.fif.gz'


def run_tfr_pipeline(source: Path, target: Path):
    files = sorted(list(source.rglob(f'*{postfix}')))

    if not len(files):
        logger.warning(f"There are no files with the expected {postfix}."
                       f"Doing nothing ...")
        return

    pbar = tqdm(sorted(files))
    for file in pbar:
        pbar.set_description("Processing %s" % file.stem)
        fid = '_'.join(str(file.stem.replace(' ', '_')).split('_')[:3])

        epochs = read_epochs(os.path.join(source, file), preload=False)
        # make sure average ref is applied
        if not epochs.info['custom_ref_applied']:
            epochs.load_data().set_eeg_reference('average')

        if analysis['filter_by']:
            if isinstance(epochs.metadata, pd.DataFrame):
                epochs.metadata = epochs.metadata.astype(str)
                if all(condition in epochs.metadata.columns for condition in analysis['filter_by']):
                    for meta_ind, _ in epochs.metadata.groupby(analysis['filter_by']):
                        query_str = ' & '.join([f'{a} == "{b}"' for a, b in zip(analysis['filter_by'], meta_ind)])
                        file_name = f"{fid}_{'_'.join(meta_ind)}"
                        epochs_query = epochs[query_str]
                        epochs_query.info['fid'] = target / file_name

                        if _file_exists(file_path=f'{epochs_query.info["fid"]}_{analysis["method"]}-tfr.h5'):
                            pbar.update(1)
                            continue

                        compute_power(epochs=epochs_query)

        else:
            file_path = target / fid
            epochs.info['fid'] = file_path
            if _file_exists(f'{file_path}_{analysis["method"]}-tfr.h5'):
                pbar.update(1)
                continue

            compute_power(epochs=epochs)

    pbar.close()


def _file_exists(file_path: str) -> bool:
    if 'overwrite' in analysis:
        return os.path.exists(file_path) and not analysis['overwrite']
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running EEG/MEG analysis pipeline\n')
    parser.add_argument("--channel_type", type=str, default='eeg', help="Type of channel to analyze, default is 'eeg'.")
    parser.add_argument("--analysis_type", type=str, default='power', help="Type of analysis pipeline to run, default"
                                                                           "is 'power'. Supported fields are 'power',"
                                                                           "'con', and 'erp'.")
    parser.add_argument("--filter_by", type=list, default=['epoch', 'triplet'], help="Filter trials based on metadata,"
                                                                                     "to disable filtering use []")
    parser.add_argument("--overwrite", type=bool, default=False, help="Whether to overwrite existing files, default is"
                                                                      "not to overwrite (False).")

    args = parser.parse_args()

    logger.info(f'\nRunning EEG/MEG {args.analysis_type} analysis pipeline for {args.channel_type} channels\n')
    if len(args.filter_by):
        logger.info(f'Current run computes {args.analysis_type} for each {", ".join(args.filter_by)} found in epochs.')
    pipeline_in = Path(epochs_path)
    pipeline_out = pipeline_in.parent / args.channel_type / args.analysis_type
    logger.info(f'Source directory is "{pipeline_in}"')
    logger.info(f'Output files are created at "{pipeline_out}"')

    # Update config with current run arguments
    analysis['method'] = args.analysis_type
    analysis['picks'] = args.channel_type
    analysis['filter_by'] = args.filter_by
    analysis['overwrite'] = args.overwrite

    if 'power' in args.analysis_type:
        run_tfr_pipeline(source=pipeline_in, target=pipeline_out)
    elif 'con' in args.analysis_type:
        raise NotImplementedError(f'{args.analysis_type} pipeline function has not been added yet.')
    elif 'erp' in args.analysis_type:
        raise NotImplementedError(f'{args.analysis_type} pipeline function has not been added yet.')
    else:
        raise NotImplementedError(f'The {args.analysis_type} method is not yet implemented,'
                                  f' supported methods are "power", "con", and "erp".')
