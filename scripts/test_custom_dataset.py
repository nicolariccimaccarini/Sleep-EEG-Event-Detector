"""Test loading the custom dataset."""

import os
import sys

project_root = os.path.abspath(".")
sys.path.append(project_root)

from sleeprnn.common import constants
from sleeprnn.helpers.reader import load_dataset


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

print("=" * 60)
print("Loading Custom Dataset")
print("=" * 60)

dataset = load_dataset(
    constants.CUSTOM_SS_NAME,
    load_checkpoint=False,
    verbose=True,
    n2_times=n2_times
)

print("\n" + "=" * 60)
print("Dataset Info")
print("=" * 60)

print(f"Dataset name: {dataset.dataset_name}")
print(f"Number of subjects (channel combinations): {len(dataset.all_ids)}")
print(f"Subject IDs: {dataset.all_ids}")
print(f"Sampling frequency: {dataset.fs} Hz")
print(f"Event type: {dataset.event_name}")

print("\n" + "=" * 60)
print("Computing Global STD")
print("=" * 60)
dataset.global_std = dataset.compute_global_std(dataset.all_ids)
print(f"Global STD: {dataset.global_std:.4f}")

print("\n" + "=" * 60)
print("Per-Subject Info")
print("=" * 60)

for subject_id in dataset.all_ids[:5]:  # Primi 5
    signal = dataset.get_subject_signal(subject_id)
    marks = dataset.get_subject_stamps(subject_id, which_expert=1)
    n2_pages = dataset.get_subject_pages(subject_id, pages_subset=constants.N2_RECORD)
    
    duration_sec = len(signal) / dataset.fs
    duration_min = duration_sec / 60
    
    print(f"\n{subject_id}:")
    print(f"  Signal: {len(signal)} samples ({duration_min:.1f} min)")
    print(f"  N2 pages: {len(n2_pages)}")
    print(f"  Spindle annotations: {len(marks)}")
    
    if len(marks) > 0:
        durations = (marks[:, 1] - marks[:, 0]) / dataset.fs
        print(f"  Spindle durations: min={durations.min():.2f}s, max={durations.max():.2f}s, mean={durations.mean():.2f}s")

print("\n" + "=" * 60)
print("Annotation Summary")
print("=" * 60)
total_spindles = 0
channels_with_spindles = []
for subject_id in dataset.all_ids:
    marks = dataset.get_subject_stamps(subject_id, which_expert=1)
    if len(marks) > 0:
        total_spindles += len(marks)
        channels_with_spindles.append(subject_id)

print(f"Total spindle annotations: {total_spindles}")
print(f"Channels with spindles: {len(channels_with_spindles)}")
if channels_with_spindles:
    print(f"  {channels_with_spindles}")

print("\n" + "=" * 60)
print("Dataset loaded successfully!")
print("=" * 60)