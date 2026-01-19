"""Extract detected spindles from pop_1 predictions."""

import os
import pickle
import numpy as np
import pandas as pd

project_root = os.path.abspath(".")
import sys
sys.path.append(project_root)

from sleeprnn.common import constants
from sleeprnn.detection import det_utils
from sleeprnn.helpers.reader import load_dataset

# Paths
RESULTS_PATH = os.path.join(project_root, "results")
PRED_PATH = os.path.join(
    RESULTS_PATH,
    "predictions_pop_1_ss",
    "20260118_from_20260118_standard_train_fixed_e1_n2_train_custom_ss_ensemble_to_e1_n2_train_pop_1_ss",
    "v2_time",
    "fold0",
    "prediction_n2_test.pkl"
)

# Optimal threshold from custom_ss training
OPTIMAL_THRESHOLD = 0.20  # Using the median/mode value from the training

# Load predictions only (lightweight)
print(f"Loading predictions from {PRED_PATH}...")
with open(PRED_PATH, "rb") as f:
    probabilities = pickle.load(f)

print(f"Found predictions for {len(probabilities)} subjects")
print(f"Using optimal threshold: {OPTIMAL_THRESHOLD}")

# Postprocessing parameters (same as used during training)
params = {
    'ss_min_separation': 0.3,  # seconds
    'ss_min_duration': 0.3,    # seconds
    'ss_max_duration': 5.0,    # seconds
}

# Sampling frequency
fs = 200  # Hz

# Extract spindles
output_dir = os.path.join(project_root, "results", "pop1_detected_spindles")
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("EXTRACTING SPINDLES")
print("="*60)

all_results = []

for i, (subject_id, prob_array) in enumerate(probabilities.items()):
    # Threshold probabilities to get binary predictions
    binary_pred = (prob_array >= OPTIMAL_THRESHOLD).astype(int)
    
    # Find contiguous regions of 1s (detected events)
    padded = np.pad(binary_pred, (1, 1), mode='constant', constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    stamps = np.column_stack([starts, ends]) if len(starts) > 0 else np.array([])
    
    if len(stamps) == 0:
        continue
    
    # Apply duration and separation filters
    durations = stamps[:, 1] - stamps[:, 0]
    
    # Filter by minimum duration
    min_dur_samples = int(params['ss_min_duration'] * fs)
    valid_duration = durations >= min_dur_samples
    
    # Filter by maximum duration if specified
    if params['ss_max_duration'] is not None:
        max_dur_samples = int(params['ss_max_duration'] * fs)
        valid_duration = valid_duration & (durations <= max_dur_samples)
    
    stamps = stamps[valid_duration]
    
    if len(stamps) == 0:
        continue
    
    # Filter by minimum separation between events
    if len(stamps) > 1:
        min_sep_samples = int(params['ss_min_separation'] * fs)
        separations = stamps[1:, 0] - stamps[:-1, 1]
        valid_separation = np.concatenate([[True], separations >= min_sep_samples])
        stamps = stamps[valid_separation]
    
    if len(stamps) == 0:
        continue
    
    # Convert from samples to seconds
    start_times = stamps[:, 0] / fs
    end_times = stamps[:, 1] / fs
    durations = end_times - start_times
    
    # Parse subject_id: "PATIENT_EEG CHANNEL" -> patient, channel
    parts = subject_id.split("_")
    patient_id = "_".join(parts[:-1])  # Everything except last part
    channel = parts[-1]  # Last part is channel
    
    for j in range(len(stamps)):
        all_results.append({
            'patient_id': patient_id,
            'channel': channel,
            'spindle_id': j + 1,
            'start_time_sec': start_times[j],
            'end_time_sec': end_times[j],
            'duration_sec': durations[j],
            'start_sample': stamps[j, 0],
            'end_sample': stamps[j, 1]
        })
    
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(probabilities)} subjects")

print(f"\nTotal processed: {len(probabilities)} subjects")

# Create main DataFrame
df = pd.DataFrame(all_results)

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total subjects analyzed: {len(probabilities)}")

if len(df) > 0:
    print(f"Patients with spindles: {df['patient_id'].nunique()}")
    print(f"Total spindles detected: {len(df)}")
    print(f"Spindles per patient (avg): {len(df) / df['patient_id'].nunique():.1f}")
    print(f"\nDuration statistics (seconds):")
    print(f"  Min: {df['duration_sec'].min():.2f}")
    print(f"  Max: {df['duration_sec'].max():.2f}")
    print(f"  Mean: {df['duration_sec'].mean():.2f}")
    print(f"  Median: {df['duration_sec'].median():.2f}")
    
    # Save individual CSV per patient
    patients_dir = os.path.join(output_dir, "by_patient")
    os.makedirs(patients_dir, exist_ok=True)
    
    print(f"\n" + "="*60)
    print("SAVING INDIVIDUAL PATIENT FILES")
    print("="*60)
    
    for patient_id in df['patient_id'].unique():
        patient_df = df[df['patient_id'] == patient_id].copy()
        
        # Sort by channel and start time
        patient_df = patient_df.sort_values(['channel', 'start_time_sec'])
        
        # Reset spindle_id per patient
        patient_df['spindle_id'] = range(1, len(patient_df) + 1)
        
        # Save patient file
        patient_file = os.path.join(patients_dir, f"{patient_id}_spindles.csv")
        patient_df.to_csv(patient_file, index=False)
        
        print(f"  {patient_id}: {len(patient_df)} spindles across {patient_df['channel'].nunique()} channels")
    
    print(f"\n✅ Saved {df['patient_id'].nunique()} patient files to {patients_dir}/")
    
    # Save overall summary
    all_spindles_file = os.path.join(output_dir, "all_detected_spindles.csv")
    df.to_csv(all_spindles_file, index=False)
    print(f"✅ Saved all {len(df)} spindles to {all_spindles_file}")
    
    # Per-patient summary statistics
    patient_summary = df.groupby('patient_id').agg({
        'spindle_id': 'count',
        'duration_sec': ['mean', 'std', 'min', 'max'],
        'channel': 'nunique'
    })
    patient_summary.columns = ['n_spindles', 'avg_duration', 'std_duration', 
                                'min_duration', 'max_duration', 'n_channels']
    patient_summary = patient_summary.round(2)
    
    patient_summary_file = os.path.join(output_dir, "patient_summary.csv")
    patient_summary.to_csv(patient_summary_file)
    print(f"✅ Saved patient summary to {patient_summary_file}")
    
    # Per-channel summary (across all patients)
    channel_summary = df.groupby('channel').agg({
        'spindle_id': 'count',
        'duration_sec': 'mean',
        'patient_id': 'nunique'
    }).rename(columns={
        'spindle_id': 'n_spindles', 
        'duration_sec': 'avg_duration',
        'patient_id': 'n_patients'
    }).round(2)
    
    channel_summary_file = os.path.join(output_dir, "channel_summary.csv")
    channel_summary.to_csv(channel_summary_file)
    print(f"✅ Saved channel summary to {channel_summary_file}")
    
else:
    print("\n⚠️  No spindles detected!")
    print("This suggests the model probabilities are very low.")
    print("Possible reasons:")
    print("  1. Domain shift between training (custom_ss) and test (pop_1_ss) data")
    print("  2. Different signal characteristics in pop_1_ss")
    print("  3. Model may need retraining or threshold adjustment")