"""Training script for custom sleep spindle dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import os
import pickle
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

project_root = os.path.abspath(".")
sys.path.append(project_root)

from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys

RESULT_PATH = os.path.join(project_root, "results")


if __name__ == "__main__":
    # ----- Experiments settings
    this_date       = datetime.datetime.now().strftime("%Y%m%d")
    task_mode       = constants.N2_RECORD
    experiment_name = "%s_custom_ss_train" % this_date

    # ----- N2 times per recording (in seconds)
    n2_times = {
        "ln": [
            (2095.0, 2288.0),
            (2349.0, 2632.0)
        ],
        "sd": [
            (607.0, 616.0),
            (644.0, 677.0),
            (778.0, 798.0),
            (855.0, 1248.0)
        ],
    }

    # ----- Model configuration 
    model_config = {
        pkeys.MODEL_VERSION:   constants.V2_TIME,
        pkeys.BORDER_DURATION: pkeys.DEFAULT_BORDER_DURATION_V2_TIME,
    }

    # ----- Load dataset
    print("Loading custom dataset")
    dataset = load_dataset(
        constants.CUSTOM_SS_NAME,
        load_checkpoint=False,
        n2_times=n2_times,
        verbose=True
    )

    # Filter subjects: use only channels with annotations
    channels_with_spindles = [
        'ln_EEG C3', 'ln_EEG C4', 'ln_EEG Cz', 'ln_EEG F3', 'ln_EEG F4', 
        'ln_EEG F7', 'ln_EEG F8', 'ln_EEG Fp1', 'ln_EEG Fp2', 'ln_EEG Fz',
        'sd_EEG C3', 'sd_EEG C4', 'sd_EEG Cz', 'sd_EEG F3', 'sd_EEG F4', 
        'sd_EEG F7', 'sd_EEG F8', 'sd_EEG Fp1', 'sd_EEG Fp2', 'sd_EEG Fz', 
        'sd_EEG T3', 'sd_EEG T4'
    ]
    valid_ids = [sid for sid in dataset.all_ids if sid in channels_with_spindles]

    # ----- Split train/val/test
    np.random.seed(42)
    np.random.shuffle(valid_ids)

    n_total = len(valid_ids)
    n_test  = max(2, n_total // 5)  # ~20% test
    n_val   = max(2, n_total // 5)  # ~20% val
    n_train = n_total - n_test - n_val

    train_ids = valid_ids[:n_train]
    val_ids   = valid_ids[n_train:n_train + n_val]
    test_ids  = valid_ids[n_train + n_val:]

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"Train IDs: {train_ids}")
    print(f"Val IDs: {val_ids}")
    print(f"Test IDs: {test_ids}")

    # ----- Compute global std
    dataset.global_std = dataset.compute_global_std(train_ids + valid_ids)
    print(f"Global STD: {dataset.global_std:.4f}")

    # ----- Create data feeders
    wich_expert = 1
    data_train  = FeederDataset(dataset, train_ids, task_mode, which_expert=wich_expert)
    data_val    = FeederDataset(dataset, val_ids, task_mode, which_expert=wich_expert)
    data_test   = FeederDataset(dataset, test_ids, task_mode, which_expert=wich_expert)

    # ----- Setup parameters
    params = copy.deepcopy(pkeys.default_params)
    params.update(model_config)

    # Data argumentation (scaled by global_std)
    da_unif_noise_uv = pkeys.DEFAULT_AUG_INDEP_UNIFORM_NOISE_INTENSITY_MICROVOLTS
    params[pkeys.AUG_INDEP_UNIFORM_NOISE_INTENSITY] = da_unif_noise_uv / dataset.global_std

    da_random_waves = copy.deepcopy(pkeys.DEFAULT_AUG_RANDOM_WAVES_PARAMS_SPINDLE)
    for da_id in range(len(da_random_waves)):
        da_random_waves[da_id]["max_amplitude"] = (
            da_random_waves[da_id]["max_amplitude_microvolts"] / dataset.global_std
        )
        da_random_waves[da_id].pop("max_amplitude_microvolts")
    params[pkeys.AUG_RANDOM_WAVES_PARAMS]      = da_random_waves
    params[pkeys.AUG_RANDOM_ANTI_WAVES_PARAMS] = pkeys.DEFAULT_AUG_RANDOM_ANTI_WAVES_PARAMS_SPINDLE

    # ----- Epochs reduction for small dimension dataset
    params[pkeys.MAX_EPOCHS] = 100

    # ----- Train
    folder_name = model_config[pkeys.MODEL_VERSION]
    base_dir    = os.path.join(
        f"{experiment_name}_{task_mode}_{constants.CUSTOM_SS_NAME}",
        folder_name,
        "fold0"
    )
    log_dir     = os.path.join(RESULT_PATH, base_dir)
    print(f"Training dir: {log_dir}")

    model = WaveletBLSTM(params=params, logdir=log_dir)
    model.fit(data_train, data_val, verbose=True)

    # ----- Predict and Save
    save_dir = os.path.abspath(
        os.path.join(RESULT_PATH, f"predictions_{constants.CUSTOM_SS_NAME}", base_dir)
    )
    checks.ensure_directory(save_dir)

    feeders_dict = {
        constants.TRAIN_SUBSET: data_train, 
        constants.VAL_SUBSET:   data_val, 
        constants.TEST_SUBSET:  data_test,
    }

    for set_name, data_inference, in feeders_dict.items():
        print(f"Predicting {set_name}...", flush=True)
        prediction = model.predict_dataset(data_inference, verbose=True)
        filename = os.path.join(save_dir, f"prediction_{task_mode}_{set_name}.pkl")
        with open(filename, "wb") as handle:
            pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nPredictions saved at {save_dir}")
    print("Training complete!")