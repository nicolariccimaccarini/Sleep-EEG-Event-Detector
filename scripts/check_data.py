"""Script to check your data structure."""

import os
import sys
import pandas as pd
import pyedflib

project_root = os.path.abspath(".")
sys.path.append(project_root)

DATA_PATH = os.path.join(project_root, "resources", "datasets", "custom")

# Check annotations
print("=" * 60)
print("ANNOTATIONS")
print("=" * 60)

for annot_file in ["calcluated_start_end_time_ln.csv", "calculated_start_end_time_sd.csv"]:
    path = os.path.join(DATA_PATH, "annotations", annot_file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n{annot_file}:")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Shape: {df.shape}")
        print(f"  Unique channels: {df['Channel'].unique().tolist()}")
        
        # Calcola durate
        df['Duration'] = df['End_Time(s)'] - df['Start_Time(s)']
        print(f"  Duration stats: min={df['Duration'].min():.3f}s, max={df['Duration'].max():.3f}s, mean={df['Duration'].mean():.3f}s")
        
        # Conta annotazioni per canale
        print(f"  Annotations per channel:")
        for ch in sorted(df['Channel'].unique()):
            count = len(df[df['Channel'] == ch])
            print(f"    {ch}: {count}")
        
        # Mostra annotazioni anomale (durata > 3s o < 0.5s)
        anomalies = df[(df['Duration'] > 3.0) | (df['Duration'] < 0.5)]
        if len(anomalies) > 0:
            print(f"  WARNING: {len(anomalies)} anomalous annotations (duration <0.5s or >3s):")
            print(anomalies.head(10).to_string())
    else:
        print(f"File not found: {path}")

# Check EDF files
print("\n" + "=" * 60)
print("EDF FILES")
print("=" * 60)

for edf_file in ["ln_24-3-23.edf", "sd_7-02-23.edf"]:
    path = os.path.join(DATA_PATH, "recordings", edf_file)
    if os.path.exists(path):
        with pyedflib.EdfReader(path) as f:
            print(f"\n{edf_file}:")
            print(f"  Duration: {f.getFileDuration()} seconds ({f.getFileDuration()/60:.1f} minutes)")
            print(f"  Channels ({len(f.getSignalLabels())}):")
            for i, ch in enumerate(f.getSignalLabels()):
                print(f"    {ch}: {f.samplefrequency(i)} Hz")
    else:
        print(f"File not found: {path}")