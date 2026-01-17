"""custom_ss.py: Dataset class for custom sleep spindle data - multi-channel support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd

# Usa MNE invece di pyedflib (più tollerante con EDF standard)
try:
    import mne
    mne.set_log_level('ERROR')  # Silenzia i warning di MNE
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

PATH_CUSTOM_RELATIVE = "custom"

# Canali da escludere
CHANNELS_TO_EXCLUDE = {'EEGA1', 'EEGA2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2', 
                       'eega1', 'eega2', 'oculo', 'mk', 'ecg', 'emg1', 'emg2'}

# Default N2 times per recording (in seconds)
DEFAULT_N2_TIMES = {
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


class CustomSS(Dataset):
    """Dataset class for custom sleep spindle annotations with multi-channel support.
    
    Each channel is treated as a separate subject, allowing the model to learn
    from all EEG channels simultaneously.
    
    Expected directory structure:
        resources/datasets/custom/
            recordings/
                ln_24-3-23.edf
                sd_7-02-23.edf
            annotations/
                calcluated_start_end_time_ln.csv
                calculated_start_end_time_sd.csv
    """

    def __init__(
        self,
        params=None,
        load_checkpoint=False,
        verbose=True,
        n2_times=None,  # Dizionario con tempi N2 per ogni recording
        **kwargs
    ):
        """Constructor.
        
        Args:
            params: Dictionary of parameters
            load_checkpoint: Whether to load from checkpoint
            verbose: Print progress
            n2_times: Dict with N2 start/end times for each subject.
                      Format: {"ln": [(start1, end1), (start2, end2), ...],
                               "sd": [(start1, end1), ...]}
                      Times in seconds.
                      If None, uses DEFAULT_N2_TIMES.
        """
        # Tempi N2 (usa default se non forniti)
        if n2_times is not None:
            self.n2_times = n2_times
        else:
            self.n2_times = DEFAULT_N2_TIMES
        
        # Parametri del dataset
        self.page_duration = 20  # Durata pagina in secondi (come MASS)
        
        # Identificatori stati sonno
        self.state_ids = np.array(["1", "2", "3", "4", "R", "W", "?"])
        self.unknown_id = "?"
        self.n2_id = "2"
        
        # Caratteristiche sleep spindles
        self.min_ss_duration = 0.5
        self.max_ss_duration = 3.0
        
        # Recording base IDs
        self.recording_ids = ["ln", "sd"]
        
        # Prima, scopri quali canali sono disponibili
        self.available_channels = self._discover_channels()
        
        # Crea IDs combinando recording + canale (es: "ln_C3", "ln_C4", "sd_C3", etc.)
        all_ids = []
        for rec_id in self.recording_ids:
            for ch in self.available_channels.get(rec_id, []):
                all_ids.append(f"{rec_id}_{ch}")
        
        if verbose:
            print(f"Discovered {len(all_ids)} subject-channel combinations:")
            for rec_id in self.recording_ids:
                channels = self.available_channels.get(rec_id, [])
                print(f"  {rec_id}: {len(channels)} channels - {channels}")
        
        super(CustomSS, self).__init__(
            dataset_dir=PATH_CUSTOM_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name="custom_ss",
            all_ids=all_ids,
            event_name=constants.SPINDLE,
            hypnogram_sleep_labels=["2"],
            hypnogram_page_duration=self.page_duration,
            n_experts=1,
            params=params,
            verbose=verbose,
        )
        
        self.global_std = None

        self.test_ids = np.array([sid for sid in self.all_ids if sid.startswith("sd_")])
        self.train_ids = np.array([sid for sid in self.all_ids if sid.startswith("ln_")])
        
        if verbose:
            print(f"Train IDs ({len(self.train_ids)}): {list(self.train_ids)}")
            print(f"Test IDs ({len(self.test_ids)}): {list(self.test_ids)}")

    def _discover_channels(self):
        """Scopre i canali EEG disponibili in ogni file EDF."""
        available = {}
        
        # Build path manually since dataset_dir is not set yet
        base_path = os.path.join("resources", "datasets", PATH_CUSTOM_RELATIVE)
        
        file_mapping = {
            "ln": os.path.join(base_path, "recordings", "ln_24-3-23.edf"),
            "sd": os.path.join(base_path, "recordings", "sd_7-02-23.edf"),
        }
        
        for rec_id, edf_path in file_mapping.items():
            if os.path.exists(edf_path):
                try:
                    if USE_MNE:
                        # Usa MNE per leggere i canali
                        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                        all_channels = raw.ch_names
                    else:
                        # Fallback a pyedflib
                        with pyedflib.EdfReader(edf_path) as f:
                            all_channels = f.getSignalLabels()
                    
                    # Filtra canali da escludere
                    valid_channels = [
                        ch for ch in all_channels 
                        if ch not in CHANNELS_TO_EXCLUDE 
                        and ch.lower() not in [x.lower() for x in CHANNELS_TO_EXCLUDE]
                    ]
                    available[rec_id] = valid_channels
                except Exception as e:
                    print(f"Warning: Could not read {edf_path}: {e}")
                    available[rec_id] = []
            else:
                print(f"Warning: File not found: {edf_path}")
                available[rec_id] = []
        
        return available

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data = {}
        
        file_mapping = {
            "ln": {
                "edf": os.path.join(self.dataset_dir, "recordings", "ln_24-3-23.edf"),
                "annot": os.path.join(self.dataset_dir, "annotations", "calcluated_start_end_time_ln.csv"),
            },
            "sd": {
                "edf": os.path.join(self.dataset_dir, "recordings", "sd_7-02-23.edf"),
                "annot": os.path.join(self.dataset_dir, "annotations", "calculated_start_end_time_sd.csv"),
            },
        }
        
        # Cache per i segnali già letti (per non rileggere lo stesso EDF)
        signal_cache = {}
        annotation_cache = {}
        
        start_time = time.time()
        
        for i, subject_id in enumerate(self.all_ids):
            # Parse subject_id: "ln_C3" -> rec_id="ln", channel="C3"
            parts = subject_id.split("_", 1)
            rec_id = parts[0]
            channel = parts[1] if len(parts) > 1 else parts[0]
            
            print(f"\nLoading {subject_id} (recording={rec_id}, channel={channel})")
            
            paths = file_mapping[rec_id]
            
            # 1. Leggi il segnale EEG (con cache)
            cache_key = (rec_id, channel)
            if cache_key not in signal_cache:
                signal, fs_original = self._read_eeg_channel(paths["edf"], channel)
                signal_cache[cache_key] = (signal, fs_original)
            else:
                signal, fs_original = signal_cache[cache_key]
            
            # 2. Leggi le annotazioni per questo canale (con cache)
            if rec_id not in annotation_cache:
                annotation_cache[rec_id] = self._read_all_annotations(paths["annot"], fs_original)
            
            marks = self._get_channel_annotations(annotation_cache[rec_id], channel)
            print(f"  Spindle annotations for {channel}: {marks.shape[0]}")
            
            # 3. Crea ipnogramma basato sui tempi N2 forniti
            hypnogram, n2_pages, all_pages = self._create_hypnogram(signal, rec_id)
            print(f"  N2 pages: {n2_pages.shape[0]}, Total pages: {all_pages.shape[0]}")
            
            # Salva i dati
            ind_dict = {
                KEY_EEG: signal.astype(np.float32),
                KEY_N2_PAGES: n2_pages.astype(np.int16),
                KEY_ALL_PAGES: all_pages.astype(np.int16),
                KEY_HYPNOGRAM: hypnogram,
                f"{KEY_MARKS}_1": marks.astype(np.int32),
            }
            data[subject_id] = ind_dict
            
            print(f"  Loaded ({i+1}/{len(self.all_ids)}). "
                  f"Time: {time.time() - start_time:.1f}s")
        
        return data

    def _read_eeg_channel(self, edf_path, channel_name):
        """Legge un singolo canale dal file EDF."""
        full_path = os.path.join("resources", "datasets", edf_path)
        
        if USE_MNE:
            # Usa MNE
            raw = mne.io.read_raw_edf(full_path, preload=False, verbose=False)
            
            # Trova il canale (case-insensitive)
            channel_idx = None
            for idx, name in enumerate(raw.ch_names):
                if name == channel_name or name.lower() == channel_name.lower():
                    channel_idx = idx
                    channel_name = name  # Usa il nome esatto
                    break
            
            if channel_idx is None:
                raise ValueError(f"Channel {channel_name} not found in {edf_path}")
            
            # Leggi solo questo canale
            raw.pick_channels([channel_name])
            raw.load_data()
            
            signal = raw.get_data()[0]  # Shape: (n_samples,)
            fs_original = int(raw.info['sfreq'])
            
            # MNE restituisce in Volts, converti in microvolts
            signal = signal * 1e6
        else:
            # Fallback a pyedflib
            with pyedflib.EdfReader(full_path) as f:
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
        
        # Applica filtro passa-banda (0.3-35 Hz)
        signal = utils.broad_filter(signal, fs_original)
        
        # Ricampiona a 200 Hz se necessario
        if self.fs != fs_original:
            signal = utils.resample_signal(signal, fs_old=fs_original, fs_new=self.fs)
        
        return signal, fs_original

    def _read_all_annotations(self, annot_path, fs_original):
        """Legge tutte le annotazioni dal CSV."""
        full_path = os.path.join("resources", "datasets", annot_path)
        df = pd.read_csv(full_path)
        
        print(f"  Annotation file columns: {df.columns.tolist()}")
        
        # Trova le colonne start/end
        start_col = None
        end_col = None
        channel_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'start' in col_lower:
                start_col = col
            elif 'end' in col_lower:
                end_col = col
            elif 'channel' in col_lower or 'chan' in col_lower:
                channel_col = col
        
        if start_col is None or end_col is None:
            raise ValueError(f"Cannot find start/end columns in {annot_path}. "
                           f"Available: {df.columns.tolist()}")
        
        return {
            'df': df,
            'start_col': start_col,
            'end_col': end_col,
            'channel_col': channel_col,
            'fs_original': fs_original
        }

    def _get_channel_annotations(self, annot_data, channel_name):
        """Estrae le annotazioni per un canale specifico."""
        df = annot_data['df']
        start_col = annot_data['start_col']
        end_col = annot_data['end_col']
        channel_col = annot_data['channel_col']
        
        # Filtra per canale se la colonna esiste
        if channel_col is not None:
            # Estrai solo il nome del canale senza "EEG " prefix
            # EDF: "EEG Fp1" -> CSV: "Fp1"
            channel_short = channel_name.replace("EEG ", "").strip()
            
            # Match case-insensitive
            csv_channels_lower = df[channel_col].str.lower().str.strip()
            channel_short_lower = channel_short.lower()
            
            mask = csv_channels_lower == channel_short_lower
            df_filtered = df[mask]
            
            if len(df_filtered) == 0:
                # Non stampare warning per canali che non hanno annotazioni (es. A1, A2, O1, O2, etc.)
                return np.array([]).reshape(0, 2)
        else:
            # Se non c'è colonna canale, usa tutte le annotazioni
            df_filtered = df
        
        starts = df_filtered[start_col].values
        ends = df_filtered[end_col].values
        
        # Converti da secondi a samples (a 200 Hz)
        marks_time = np.stack([starts, ends], axis=1)
        marks = np.round(marks_time * self.fs).astype(np.int32)
        
        # Filtra per durata valida
        durations = (marks[:, 1] - marks[:, 0]) / self.fs
        valid = (durations >= self.min_ss_duration) & (durations <= self.max_ss_duration)
        marks = marks[valid]
        
        return marks

    def _create_hypnogram(self, signal, rec_id):
        """Crea ipnogramma basato sui tempi N2 forniti."""
        page_size = int(self.page_duration * self.fs)
        total_pages = int(signal.shape[0] / page_size)
        
        # Inizializza tutto come "unknown"
        hypnogram = np.array([self.unknown_id] * total_pages)
        
        # Se abbiamo i tempi N2, marcali
        if rec_id in self.n2_times and len(self.n2_times[rec_id]) > 0:
            for start_sec, end_sec in self.n2_times[rec_id]:
                start_page = int(start_sec / self.page_duration)
                end_page = int(end_sec / self.page_duration)
                
                # Clamp ai limiti validi
                start_page = max(0, min(start_page, total_pages - 1))
                end_page = max(0, min(end_page, total_pages - 1))
                
                hypnogram[start_page:end_page + 1] = self.n2_id
        else:
            # Se non abbiamo tempi N2, assume tutto N2 (per test)
            print(f"    Warning: No N2 times for {rec_id}, assuming all N2")
            hypnogram[:] = self.n2_id
        
        # Trova pagine N2
        n2_pages = np.where(hypnogram == self.n2_id)[0].astype(np.int16)
        
        # Escludi prima e ultima pagina per sicurezza (bordi)
        if len(n2_pages) > 2:
            n2_pages = n2_pages[1:-1]
        
        all_pages = n2_pages.copy()
        
        return hypnogram, n2_pages, all_pages

    def get_subjects_by_recording(self, rec_id):
        """Ritorna tutti i subject_ids per una data registrazione."""
        return [sid for sid in self.all_ids if sid.startswith(f"{rec_id}_")]