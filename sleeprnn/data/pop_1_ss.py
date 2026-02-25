"""pop_1_ss.py: Dataset class for testing the algorithm on the dataset 'pop_1' - multi-channel support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import re

import numpy as np

# Use MNE for reading EDF files
try:
    import mne
    mne.set_log_level('ERROR')
    USE_MNE = True
except ImportError:
    import pyedflib
    USE_MNE = False
    print("Warning: MNE not installed, falling back to pyedflib")

from sleeprnn.common import constants
from sleeprnn.data import utils
from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import (
    KEY_EEG,
    KEY_N2_PAGES,
    KEY_ALL_PAGES,
    KEY_MARKS,
    KEY_HYPNOGRAM,
)

PATH_POP1_RELATIVE = "pop_1"

# Channels to exclude (same as custom_ss)
CHANNELS_TO_EXCLUDE = {'EEGA1', 'EEGA2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2', 
                       'eega1', 'eega2', 'oculo', 'mk', 'ecg', 'emg1', 'emg2'}


class Pop1SS(Dataset):
    """Dataset class for pop_1 sleep spindle detection with multi-channel support.
    
    Each channel is treated as a separate subject, allowing the model to detect
    spindles on all EEG channels simultaneously.
    
    Expected directory structure:
        resources/datasets/pop_1/
            recordings/
                file_1.edf
                file_2.edf
                ...
    """

    def __init__(
        self,
        params=None,
        load_checkpoint=False,
        verbose=True,
        **kwargs
    ):
        """Constructor.
        
        Args:
            params: Dictionary of parameters
            load_checkpoint: Whether to load from checkpoint
            verbose: Print progress
        """
        # Dataset parameters
        self.page_duration = 20  # Page duration in seconds (same as MASS)
        
        # Sleep state identifiers
        self.state_ids = np.array(["1", "2", "3", "4", "R", "W", "?"])
        self.unknown_id = "?"
        self.n2_id = "2"
        
        # Sleep spindle characteristics
        self.min_ss_duration = 0.5
        self.max_ss_duration = 3.0
        
        # Discover available recordings
        self.recording_files = self._discover_recordings()
        
        # Discover available channels per recording
        self.available_channels = self._discover_channels()
        
        # Create IDs combining patient + channel (e.g., "file_1_C3")
        all_ids = []
        for patient_id in self.recording_files.keys():
            for ch in self.available_channels.get(patient_id, []):
                all_ids.append(f"{patient_id}_{ch}")
        
        if verbose:
            print(f"Discovered {len(all_ids)} patient-channel combinations:")
            print(f"  {len(self.recording_files)} patients")
            channels_per_patient = {pid: len(self.available_channels.get(pid, [])) 
                                   for pid in self.recording_files.keys()}
            print(f"  Channels per patient: min={min(channels_per_patient.values())}, "
                  f"max={max(channels_per_patient.values())}, "
                  f"avg={np.mean(list(channels_per_patient.values())):.1f}")
        
        super(Pop1SS, self).__init__(
            dataset_dir=PATH_POP1_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name="pop_1_ss",
            all_ids=all_ids,
            event_name=constants.SPINDLE,
            hypnogram_sleep_labels=["2"],
            hypnogram_page_duration=self.page_duration,
            n_experts=1,
            params=params,
            verbose=verbose,
        )
        
        self.global_std = None

        # All subjects are for testing (no ground truth annotations)
        self.test_ids = np.array(self.all_ids)
        self.train_ids = np.array([])
        
        if verbose:
            print(f"Test IDs: {len(self.test_ids)} subjects")

    def _discover_recordings(self):
        """Discovers available EDF recordings in the pop_1 directory."""
        base_path = os.path.join("resources", "datasets", PATH_POP1_RELATIVE, "recordings")
        
        if not os.path.exists(base_path):
            print(f"Warning: Directory not found: {base_path}")
            return {}
        
        recording_files = {}
        for filename in os.listdir(base_path):
            if filename.endswith('.edf'):
                # Remove .edf extension to get patient ID
                patient_id = filename[:-4]
                recording_files[patient_id] = os.path.join(base_path, filename)
        
        return recording_files

    def _discover_channels(self):
        """Discovers EEG channels available in each EDF file."""
        available = {}
        
        for patient_id, edf_path in self.recording_files.items():
            if os.path.exists(edf_path):
                try:
                    if USE_MNE:
                        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                        all_channels = raw.ch_names
                    else:
                        with pyedflib.EdfReader(edf_path) as f:
                            all_channels = f.getSignalLabels()
                    
                    # Filter out excluded channels
                    valid_channels = [
                        ch for ch in all_channels 
                        if ch not in CHANNELS_TO_EXCLUDE 
                        and ch.lower() not in [x.lower() for x in CHANNELS_TO_EXCLUDE]
                    ]
                    available[patient_id] = valid_channels
                except Exception as e:
                    print(f"Warning: Could not read {edf_path}: {e}")
                    available[patient_id] = []
            else:
                print(f"Warning: File not found: {edf_path}")
                available[patient_id] = []
        
        return available

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data = {}
        
        # Cache for already read signals (to avoid re-reading the same EDF)
        signal_cache = {}
        
        start_time = time.time()
        
        for i, subject_id in enumerate(self.all_ids):
            parts = subject_id.rsplit("_", 1)
            patient_id = parts[0]
            channel = parts[1] if len(parts) > 1 else parts[0]
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"\nLoading {subject_id} (patient={patient_id}, channel={channel})")
            
            edf_path = self.recording_files[patient_id]
            
            # 1. Read EEG signal (with cache)
            cache_key = (patient_id, channel)
            if cache_key not in signal_cache:
                signal, fs_original = self._read_eeg_channel(edf_path, channel)
                signal_cache[cache_key] = (signal, fs_original)
            else:
                signal, fs_original = signal_cache[cache_key]
            
            # 2. No annotations for pop_1 (inference only)
            marks = np.array([]).reshape(0, 2)
            
            # 3. Create hypnogram (assume all N2 for inference)
            hypnogram, n2_pages, all_pages = self._create_hypnogram(signal)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  N2 pages: {n2_pages.shape[0]}, Total pages: {all_pages.shape[0]}")
            
            # Save data
            ind_dict = {
                KEY_EEG: signal.astype(np.float32),
                KEY_N2_PAGES: n2_pages.astype(np.int16),
                KEY_ALL_PAGES: all_pages.astype(np.int16),
                KEY_HYPNOGRAM: hypnogram,
                f"{KEY_MARKS}_1": marks.astype(np.int32),
            }
            data[subject_id] = ind_dict
            
            if (i + 1) % 10 == 0:
                print(f"  Loaded ({i+1}/{len(self.all_ids)}). "
                      f"Time: {time.time() - start_time:.1f}s")
        
        print(f"\nTotal loading time: {time.time() - start_time:.1f}s")
        return data

    def _read_eeg_channel(self, edf_path, channel_name):
        """Reads a single channel from the EDF file."""
        if USE_MNE:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            
            # Find channel (case-insensitive)
            channel_idx = None
            for idx, name in enumerate(raw.ch_names):
                if name == channel_name or name.lower() == channel_name.lower():
                    channel_idx = idx
                    channel_name = name  # Use exact name
                    break
            
            if channel_idx is None:
                raise ValueError(f"Channel {channel_name} not found in {edf_path}")
            
            # Read only this channel
            raw.pick_channels([channel_name])
            raw.load_data()
            
            signal = raw.get_data()[0]  # Shape: (n_samples,)
            fs_original = int(raw.info['sfreq'])
            
            # MNE returns in Volts, convert to microvolts
            signal = signal * 1e6
        else:
            # Fallback to pyedflib
            with pyedflib.EdfReader(edf_path) as f:
                channel_names = f.getSignalLabels()
                
                channel_idx = None
                for idx, name in enumerate(channel_names):
                    if name == channel_name or name.lower() == channel_name.lower():
                        channel_idx = idx
                        break
                
                if channel_idx is None:
                    raise ValueError(f"Channel {channel_name} not found in {edf_path}")
                
                signal = f.readSignal(channel_idx)
                fs_original = int(f.samplefrequency(channel_idx))
        
        # Apply band-pass filter (0.3-35 Hz)
        signal = utils.broad_filter(signal, fs_original)
        
        # Resample to 200 Hz if needed
        if self.fs != fs_original:
            signal = utils.resample_signal(signal, fs_old=fs_original, fs_new=self.fs)
        
        return signal, fs_original

    def _create_hypnogram(self, signal):
        """Creates hypnogram assuming all N2 (for inference)."""
        page_size = int(self.page_duration * self.fs)
        total_pages = int(signal.shape[0] / page_size)
        
        # Assume all N2 for inference
        hypnogram = np.array([self.n2_id] * total_pages)
        
        # Find N2 pages
        n2_pages = np.where(hypnogram == self.n2_id)[0].astype(np.int16)
        
        # Exclude first and last page for safety (borders)
        if len(n2_pages) > 2:
            n2_pages = n2_pages[1:-1]
        
        all_pages = n2_pages.copy()
        
        return hypnogram, n2_pages, all_pages

    def get_subjects_by_patient(self, patient_id):
        """Returns all subject_ids for a given patient."""
        return [sid for sid in self.all_ids if sid.startswith(f"{patient_id}_")]

    def get_patient_id_from_subject(self, subject_id):
        """Extracts patient ID from subject_id."""
        return subject_id.rsplit("_", 1)[0]