#!/usr/bin/env python3
"""
Compare your BMI160 right-instep raw recording against the gaitmap
SensorPositionComparison2019 dataset.

Default dataset target:
    participant = 4d91
    test        = normal_10
    side        = right
    sensor      = r_instep

This script follows your existing project style:
    import pipeline_core as core
    trial = core.load_trial(...)

It creates:
    1) Dataset axis-wise acc/gyr plot
    2) Your BMI160 axis-wise acc/gyr plot
    3) Acceleration norm comparison
    4) Gyroscope norm comparison
    5) Z-score normalized norm comparison
    6) Summary CSV with sampling rates and signal statistics

Important:
    - Your BMI160 acceleration is assumed to be in g.
    - Your BMI160 gyroscope is assumed to be in deg/s.
    - gaitmap dataset acceleration is treated as m/s^2.
    - gaitmap dataset gyroscope is treated as rad/s.
    - Your gyroscope is converted from deg/s to rad/s for comparison.
    - Axis-wise plots are qualitative only because casing/sensor axes may differ.
    - Norm plots are the main fair visual comparison.

Example:
    python compare_bmi160_with_sensorposition_dataset.py ^
      --data_folder "path/to/sensorpositoncomparison-v1.0.0-beta" ^
      --core_dir "path/to/foot-imu-mtc/algorithm" ^
      --my_csv imu_recordings\\imu_20260502_190000_RAW.csv

If --my_csv is omitted, the script uses the newest CSV in imu_recordings/,
preferring files ending in *_RAW.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


G = 9.80665


def latest_csv(recordings_dir: Path) -> Path:
    """Find newest walking CSV, preferring RAW files."""
    raw_candidates = sorted(recordings_dir.glob("*_RAW.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if raw_candidates:
        return raw_candidates[0]

    candidates = sorted(recordings_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if "TEMP" not in p.name.upper()]
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No CSV files found in {recordings_dir.resolve()}")


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return x * 0.0
    return (x - np.nanmean(x)) / s


def crop_by_time(df: pd.DataFrame, time_col: str, start_s: float, duration_s: float) -> pd.DataFrame:
    end_s = start_s + duration_s
    out = df[(df[time_col] >= start_s) & (df[time_col] <= end_s)].copy()
    if len(out) < 10:
        raise ValueError(
            f"Crop produced too few samples. start={start_s}, duration={duration_s}, rows={len(out)}"
        )
    out[time_col] = out[time_col] - out[time_col].iloc[0]
    return out.reset_index(drop=True)


def load_my_bmi160_csv(path: Path, signal_mode: str = "raw") -> pd.DataFrame:
    """Load user's XIAO/BMI160 CSV and standardize units.

    Output columns:
        time_s
        acc_x_mps2, acc_y_mps2, acc_z_mps2
        gyr_x_radps, gyr_y_radps, gyr_z_radps
        acc_norm_mps2
        gyr_norm_radps
    """
    df = pd.read_csv(path)

    required = ["seq", "t_ms", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"My CSV missing required columns: {missing}. Found: {list(df.columns)}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[required].isna().any(axis=1).any():
        bad = int(df[required].isna().any(axis=1).sum())
        raise ValueError(f"My CSV contains {bad} rows with missing/non-numeric values.")

    # Sequence integrity check.
    seq = df["seq"].astype(int).to_numpy()
    expected = np.arange(seq.min(), seq.max() + 1)
    missing_seq = sorted(set(expected.tolist()) - set(seq.tolist()))
    if missing_seq:
        raise ValueError(f"My CSV has missing seq values. First few missing: {missing_seq[:20]}")

    # Choose raw columns if requested and available.
    if signal_mode == "raw" and all(c in df.columns for c in ["ax_raw_g", "ay_raw_g", "az_raw_g", "gx_raw_dps", "gy_raw_dps", "gz_raw_dps"]):
        ax_col, ay_col, az_col = "ax_raw_g", "ay_raw_g", "az_raw_g"
        gx_col, gy_col, gz_col = "gx_raw_dps", "gy_raw_dps", "gz_raw_dps"
    else:
        ax_col, ay_col, az_col = "ax_g", "ay_g", "az_g"
        gx_col, gy_col, gz_col = "gx_dps", "gy_dps", "gz_dps"

    out = pd.DataFrame()
    out["time_s"] = (df["t_ms"].astype(float) - float(df["t_ms"].iloc[0])) / 1000.0

    out["acc_x_mps2"] = df[ax_col].astype(float) * G
    out["acc_y_mps2"] = df[ay_col].astype(float) * G
    out["acc_z_mps2"] = df[az_col].astype(float) * G

    out["gyr_x_radps"] = np.deg2rad(df[gx_col].astype(float))
    out["gyr_y_radps"] = np.deg2rad(df[gy_col].astype(float))
    out["gyr_z_radps"] = np.deg2rad(df[gz_col].astype(float))

    out["acc_norm_mps2"] = np.linalg.norm(out[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy(float), axis=1)
    out["gyr_norm_radps"] = np.linalg.norm(out[["gyr_x_radps", "gyr_y_radps", "gyr_z_radps"]].to_numpy(float), axis=1)

    out.attrs["source_csv"] = str(path)
    out.attrs["signal_mode"] = signal_mode
    out.attrs["sample_count"] = int(len(out))
    if len(out) > 1:
        duration = float(out["time_s"].iloc[-1] - out["time_s"].iloc[0])
        out.attrs["effective_hz"] = float((len(out) - 1) / duration) if duration > 0 else np.nan
    else:
        out.attrs["effective_hz"] = np.nan

    return out


def load_dataset_trial(
    *,
    data_folder: Path,
    core_dir: Optional[Path],
    participant: str,
    test: str,
    side: str,
    sensor: str,
    padding_s: float,
) -> Tuple[pd.DataFrame, float]:
    """Load gaitmap SensorPositionComparison2019 trial using user's core.load_trial."""
    if core_dir is not None:
        sys.path.insert(0, str(core_dir.resolve()))

    import pipeline_core as core

    trial = core.load_trial(
        data_folder=str(data_folder),
        participant=participant,
        test=test,
        side=side,
        padding_s=padding_s,
        sensor=sensor,
        toe_marker="r_toe" if side.lower().startswith("r") else "l_toe",
    )

    imu = trial.imu_df.copy()
    fs_hz = float(trial.fs_hz)

    # imu columns are acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z.
    required = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    missing = [c for c in required if c not in imu.columns]
    if missing:
        raise ValueError(f"Dataset IMU missing columns: {missing}. Found: {list(imu.columns)}")

    out = pd.DataFrame()

    # In the gaitmap example, the IMU index is a time axis. If not, fall back to fs_hz.
    try:
        idx = imu.index.to_numpy(dtype=float)
        if len(idx) > 1 and np.all(np.diff(idx) > 0) and np.nanmedian(np.diff(idx)) < 0.1:
            out["time_s"] = idx - idx[0]
        else:
            out["time_s"] = np.arange(len(imu), dtype=float) / fs_hz
    except Exception:
        out["time_s"] = np.arange(len(imu), dtype=float) / fs_hz

    # Dataset units from gaitmap example:
    # acceleration m/s^2, gyro rad/s.
    out["acc_x_mps2"] = imu["acc_x"].astype(float).to_numpy()
    out["acc_y_mps2"] = imu["acc_y"].astype(float).to_numpy()
    out["acc_z_mps2"] = imu["acc_z"].astype(float).to_numpy()

    out["gyr_x_radps"] = np.deg2rad(imu["gyr_x"].astype(float).to_numpy())
    out["gyr_y_radps"] = np.deg2rad(imu["gyr_y"].astype(float).to_numpy())
    out["gyr_z_radps"] = np.deg2rad(imu["gyr_z"].astype(float).to_numpy())

    out["acc_norm_mps2"] = np.linalg.norm(out[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy(float), axis=1)
    out["gyr_norm_radps"] = np.linalg.norm(out[["gyr_x_radps", "gyr_y_radps", "gyr_z_radps"]].to_numpy(float), axis=1)

    out.attrs["participant"] = participant
    out.attrs["test"] = test
    out.attrs["side"] = side
    out.attrs["sensor"] = sensor
    out.attrs["fs_hz"] = fs_hz
    out.attrs["sample_count"] = int(len(out))

    return out, fs_hz


def plot_axiswise(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

    axes[0].plot(df["time_s"], df["acc_x_mps2"], label="acc_x")
    axes[0].plot(df["time_s"], df["acc_y_mps2"], label="acc_y")
    axes[0].plot(df["time_s"], df["acc_z_mps2"], label="acc_z")
    axes[0].set_ylabel("Acceleration [m/s²]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["time_s"], df["gyr_x_radps"], label="gyr_x")
    axes[1].plot(df["time_s"], df["gyr_y_radps"], label="gyr_y")
    axes[1].plot(df["time_s"], df["gyr_z_radps"], label="gyr_z")
    axes[1].set_ylabel("Gyroscope [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_norm_compare(dataset: pd.DataFrame, mine: pd.DataFrame, title: str, out_path: Path, column: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dataset["time_s"], dataset[column], label="Dataset 4d91 normal_10 r_instep", linewidth=1.2)
    ax.plot(mine["time_s"], mine[column], label="My BMI160 right instep", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_zscore_norm_compare(dataset: pd.DataFrame, mine: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

    axes[0].plot(dataset["time_s"], zscore(dataset["acc_norm_mps2"].to_numpy()), label="Dataset acc norm z-score")
    axes[0].plot(mine["time_s"], zscore(mine["acc_norm_mps2"].to_numpy()), label="My BMI160 acc norm z-score")
    axes[0].set_ylabel("Acceleration norm [z-score]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(dataset["time_s"], zscore(dataset["gyr_norm_radps"].to_numpy()), label="Dataset gyro norm z-score")
    axes[1].plot(mine["time_s"], zscore(mine["gyr_norm_radps"].to_numpy()), label="My BMI160 gyro norm z-score")
    axes[1].set_ylabel("Gyro norm [z-score]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle("Normalized norm comparison: waveform shape only")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summary_stats(name: str, df: pd.DataFrame, fs_hz: float) -> Dict[str, float | str]:
    duration_s = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if len(df) > 1 else np.nan
    return {
        "signal": name,
        "samples": int(len(df)),
        "duration_s": duration_s,
        "fs_hz": float(fs_hz),
        "acc_norm_mean_mps2": float(df["acc_norm_mps2"].mean()),
        "acc_norm_std_mps2": float(df["acc_norm_mps2"].std(ddof=1)),
        "gyr_norm_mean_radps": float(df["gyr_norm_radps"].mean()),
        "gyr_norm_std_radps": float(df["gyr_norm_radps"].std(ddof=1)),
        "acc_norm_max_mps2": float(df["acc_norm_mps2"].max()),
        "gyr_norm_max_radps": float(df["gyr_norm_radps"].max()),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare raw BMI160 right-instep walking data against SensorPositionComparison2019 r_instep."
    )
    p.add_argument("--data_folder", required=True, type=Path, help="Path to sensorpositioncomparison-v1.0.0-beta folder.")
    p.add_argument("--core_dir", default=Path("."), type=Path, help="Folder containing pipeline_core.py and legacy script.")
    p.add_argument("--my_csv", default=None, type=Path, help="Your BMI160 CSV. If omitted, newest imu_recordings/*_RAW.csv is used.")
    p.add_argument("--recordings_dir", default=Path("imu_recordings"), type=Path)

    p.add_argument("--participant", default="4d91")
    p.add_argument("--test", default="normal_10")
    p.add_argument("--side", default="right")
    p.add_argument("--sensor", default="r_instep")
    p.add_argument("--padding_s", default=3.0, type=float)

    p.add_argument("--my_signal", default="raw", choices=["raw", "calibrated"], help="Use raw columns if available, or calibrated columns.")
    p.add_argument("--dataset_start_s", default=3.0, type=float, help="Start time for dataset crop.")
    p.add_argument("--my_start_s", default=1.0, type=float, help="Start time for your recording crop.")
    p.add_argument("--duration_s", default=8.0, type=float, help="Duration of comparison window.")
    p.add_argument("--out_dir", default=Path("raw_instep_comparison_plots"), type=Path)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    my_csv = args.my_csv if args.my_csv is not None else latest_csv(args.recordings_dir)
    print(f"[MY] Using CSV: {my_csv}")

    mine_full = load_my_bmi160_csv(my_csv, signal_mode=args.my_signal)
    my_fs = float(mine_full.attrs.get("effective_hz", np.nan))

    dataset_full, dataset_fs = load_dataset_trial(
        data_folder=args.data_folder,
        core_dir=args.core_dir,
        participant=args.participant,
        test=args.test,
        side=args.side,
        sensor=args.sensor,
        padding_s=args.padding_s,
    )

    print(f"[DATASET] participant={args.participant}, test={args.test}, side={args.side}, sensor={args.sensor}")
    print(f"[DATASET] fs_hz={dataset_fs:.3f}, samples={len(dataset_full)}")
    print(f"[MY] effective_fs_hz={my_fs:.3f}, samples={len(mine_full)}")

    dataset = crop_by_time(dataset_full, "time_s", args.dataset_start_s, args.duration_s)
    mine = crop_by_time(mine_full, "time_s", args.my_start_s, args.duration_s)

    # Save processed windows for inspection.
    dataset_csv = args.out_dir / "dataset_4d91_normal10_r_instep_window.csv"
    my_window_csv = args.out_dir / "my_bmi160_right_instep_window.csv"
    dataset.to_csv(dataset_csv, index=False)
    mine.to_csv(my_window_csv, index=False)

    # Plots.
    plot_axiswise(
        dataset,
        f"Dataset: {args.participant} {args.test} {args.sensor}",
        args.out_dir / "01_dataset_axiswise_acc_gyr.png",
    )
    plot_axiswise(
        mine,
        f"My BMI160: right instep ({args.my_signal})",
        args.out_dir / "02_my_bmi160_axiswise_acc_gyr.png",
    )
    plot_norm_compare(
        dataset,
        mine,
        "Acceleration norm comparison",
        args.out_dir / "03_acc_norm_comparison.png",
        "acc_norm_mps2",
        "Acceleration norm [m/s²]",
    )
    plot_norm_compare(
        dataset,
        mine,
        "Gyroscope norm comparison",
        args.out_dir / "04_gyr_norm_comparison.png",
        "gyr_norm_radps",
        "Gyroscope norm [rad/s]",
    )
    plot_zscore_norm_compare(
        dataset,
        mine,
        args.out_dir / "05_zscore_norm_shape_comparison.png",
    )

    summary = pd.DataFrame([
        summary_stats("dataset_4d91_normal10_r_instep_window", dataset, dataset_fs),
        summary_stats("my_bmi160_right_instep_window", mine, my_fs),
    ])
    summary_path = args.out_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    notes_path = args.out_dir / "interpretation_notes.txt"
    notes_path.write_text(
        "Raw visual comparison notes:\\n"
        "- Main fair comparison: acceleration norm and gyroscope norm.\\n"
        "- Axis-wise plots are qualitative only because casing axes and gaitmap coordinate axes may differ.\\n"
        "- Your BMI160 acceleration was converted from g to m/s^2.\\n"
        "- Your BMI160 gyroscope was converted from deg/s to rad/s.\\n"
        "- Dataset target: participant 4d91, normal_10, right side, r_instep.\\n"
        "- For professor review: show 03_acc_norm_comparison.png, 04_gyr_norm_comparison.png, and 05_zscore_norm_shape_comparison.png first.\\n",
        encoding="utf-8",
    )

    print("\\nSaved outputs:")
    for path in [
        args.out_dir / "01_dataset_axiswise_acc_gyr.png",
        args.out_dir / "02_my_bmi160_axiswise_acc_gyr.png",
        args.out_dir / "03_acc_norm_comparison.png",
        args.out_dir / "04_gyr_norm_comparison.png",
        args.out_dir / "05_zscore_norm_shape_comparison.png",
        summary_path,
        notes_path,
    ]:
        print(f"  {path.resolve()}")

    print("\\nRecommendation:")
    print("  Use 03, 04, and 05 for the main visual comparison.")
    print("  Use axis-wise plots only as supporting figures because the sensor coordinate frames may not match exactly.")


if __name__ == "__main__":
    main()
