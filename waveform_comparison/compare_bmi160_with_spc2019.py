#!/usr/bin/env python3
"""
Compare your BMI160 right-instep recording with SensorPositionComparison2019
using the gaitmap sensor-frame convention.

Default target:
    Dataset participant = 4d91
    Test                = normal_10
    Side                = right
    Sensor              = r_instep

Major fixes vs old script:
    1. Dataset gyro is treated as deg/s, not rad/s.
       gaitmap package convention uses acceleration in m/s² and angular velocity in deg/s.

    2. Your BMI160 gyro is kept in deg/s.
       No rad/s conversion is applied.

    3. Your BMI160 casing/sensor axes are transformed into a gaitmap-style sensor frame.
       Based on your Stage 0 results:
           Front_x_plus  = +BMI160 x points toe/front
           Back_x_minus  = -BMI160 x points heel/back
           Left_y_plus   = +BMI160 y points casing-left
           Right_y_minus = -BMI160 y points casing-right
           Top_z_plus    = +BMI160 z points top/outward
           Bottom_z_minus= -BMI160 z points bottom/toward shoe

       Based on the provided gaitmap sensor-frame reference:
           gaitmap x ≈ top/outward/superior direction
           gaitmap y ≈ left-right / ML direction
           gaitmap z ≈ toe-heel / PA direction

       Default BMI160-to-gaitmap mapping:
           gaitmap_x = +bmi160_z
           gaitmap_y = +bmi160_y
           gaitmap_z = +bmi160_x

       This is controlled by:
           --my_axis_map "+z,+y,+x"

       If signs look inverted, try:
           --my_axis_map "+z,-y,+x"
           --my_axis_map "+z,+y,-x"
           --my_axis_map "+z,-y,-x"

    4. Dataset walking window is selected from gait events by default.
       Fallback is robust active-window detection from gyro norm.

    5. Your walking window is auto-selected using robust gyro activity.
       You can override manually with --my_start_s.

Outputs:
    raw_instep_comparison_plots_v2/
      00_window_selection_diagnostics.png
      01_dataset_axiswise_gaitmap_frame.png
      02_my_bmi160_axiswise_gaitmap_frame.png
      03_acc_norm_comparison.png
      04_gyr_norm_comparison_degps.png
      05_zscore_norm_shape_comparison.png
      06_axis_overlay_gaitmap_frame.png
      comparison_summary.csv
      interpretation_notes.txt

Example:
    python compare_bmi160_with_sensorposition_dataset_v2.py ^
      --data_folder "path/to/sensorpositoncomparison-v1.0.0-beta" ^
      --core_dir "path/to/foot-imu-mtc/algorithm" ^
      --my_csv "imu_recordings\\imu_20260502_191304_RAW.csv"

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


G = 9.80665


# -----------------------------
# Generic helpers
# -----------------------------

def latest_csv(recordings_dir: Path) -> Path:
    raw_candidates = sorted(
        recordings_dir.glob("*_RAW.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if raw_candidates:
        return raw_candidates[0]

    candidates = sorted(
        recordings_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
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


def parse_axis_map(axis_map: str) -> List[Tuple[float, str]]:
    """
    Parse "+z,+y,+x" into:
        [(+1, "z"), (+1, "y"), (+1, "x")]
    Meaning:
        output x = +input z
        output y = +input y
        output z = +input x
    """
    parts = [p.strip().lower() for p in axis_map.split(",")]
    if len(parts) != 3:
        raise ValueError("--my_axis_map must have exactly 3 comma-separated entries, e.g. '+z,+y,+x'")

    out = []
    used = set()
    for p in parts:
        if len(p) != 2 or p[0] not in "+-" or p[1] not in "xyz":
            raise ValueError(f"Invalid axis-map entry: {p}. Use entries like +x, -y, +z.")
        sign = 1.0 if p[0] == "+" else -1.0
        axis = p[1]
        if axis in used:
            raise ValueError(f"Axis {axis} used more than once in --my_axis_map={axis_map}")
        used.add(axis)
        out.append((sign, axis))

    return out


def apply_axis_map(df: pd.DataFrame, axis_map: str, prefix_acc: str = "acc", prefix_gyr: str = "gyr") -> pd.DataFrame:
    """
    Transform input columns:
        acc_x_mps2, acc_y_mps2, acc_z_mps2
        gyr_x_degps, gyr_y_degps, gyr_z_degps

    into gaitmap-style:
        acc_x_mps2, acc_y_mps2, acc_z_mps2
        gyr_x_degps, gyr_y_degps, gyr_z_degps
    according to axis_map.
    """
    mapping = parse_axis_map(axis_map)
    out = pd.DataFrame()
    out["time_s"] = df["time_s"].to_numpy(dtype=float)

    in_acc = {
        "x": df["acc_x_mps2"].to_numpy(dtype=float),
        "y": df["acc_y_mps2"].to_numpy(dtype=float),
        "z": df["acc_z_mps2"].to_numpy(dtype=float),
    }
    in_gyr = {
        "x": df["gyr_x_degps"].to_numpy(dtype=float),
        "y": df["gyr_y_degps"].to_numpy(dtype=float),
        "z": df["gyr_z_degps"].to_numpy(dtype=float),
    }

    out_axes = ["x", "y", "z"]
    for out_axis, (sign, in_axis) in zip(out_axes, mapping):
        out[f"acc_{out_axis}_mps2"] = sign * in_acc[in_axis]
        out[f"gyr_{out_axis}_degps"] = sign * in_gyr[in_axis]

    add_norms(out)
    return out


def add_norms(df: pd.DataFrame) -> None:
    df["acc_norm_mps2"] = np.linalg.norm(
        df[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy(dtype=float),
        axis=1,
    )
    df["gyr_norm_degps"] = np.linalg.norm(
        df[["gyr_x_degps", "gyr_y_degps", "gyr_z_degps"]].to_numpy(dtype=float),
        axis=1,
    )


def crop_by_time(df: pd.DataFrame, start_s: float, duration_s: float) -> pd.DataFrame:
    end_s = start_s + duration_s
    out = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()
    if len(out) < 10:
        raise ValueError(f"Crop too short: start={start_s:.3f}s, duration={duration_s:.3f}s, rows={len(out)}")
    out["time_s"] = out["time_s"] - out["time_s"].iloc[0]
    return out.reset_index(drop=True)


def robust_active_start(
    df: pd.DataFrame,
    fs_hz: float,
    duration_s: float,
    prefer_after_s: float = 0.0,
    signal_col: str = "gyr_norm_degps",
) -> float:
    """
    Find active walking window using clipped gyro norm.

    Why clipped?
        A single impact/turn spike should not dominate the window selection.
    """
    if len(df) < 10:
        return 0.0

    t = df["time_s"].to_numpy(dtype=float)
    g = df[signal_col].to_numpy(dtype=float)

    valid = np.isfinite(g)
    if not np.any(valid):
        return 0.0

    g = np.where(valid, g, 0.0)

    # Ignore initial standing if requested.
    eligible = t >= float(prefer_after_s)
    if np.sum(eligible) < max(10, int(fs_hz * duration_s * 0.5)):
        eligible[:] = True

    # Clip very high spikes so continuous walking is preferred over one impact.
    p90 = np.nanpercentile(g[eligible], 90)
    p95 = np.nanpercentile(g[eligible], 95)
    clip_at = max(p90, 0.7 * p95)
    if not np.isfinite(clip_at) or clip_at <= 0:
        clip_at = np.nanmax(g[eligible])

    g_score = np.clip(g, 0, clip_at)

    window_n = max(5, int(round(duration_s * fs_hz)))
    if window_n >= len(g_score):
        return float(max(0.0, t[0]))

    kernel = np.ones(window_n, dtype=float) / float(window_n)
    score = np.convolve(g_score, kernel, mode="valid")

    # Penalize windows that start before prefer_after_s.
    start_times = t[: len(score)]
    score = np.where(start_times >= prefer_after_s, score, -np.inf)

    if not np.isfinite(score).any():
        idx = 0
    else:
        idx = int(np.nanargmax(score))

    return float(t[idx])


def event_based_dataset_start(
    events_gt_imu: Optional[pd.DataFrame],
    fs_hz: float,
    duration_s: float,
    stride_start: int = 0,
    pre_margin_s: float = 0.25,
) -> Optional[float]:
    """
    Pick dataset crop from gait events.

    Uses IC if available. Events are assumed to be in IMU samples,
    matching your core.load_trial output.
    """
    if events_gt_imu is None or len(events_gt_imu) == 0:
        return None

    df = events_gt_imu.copy()

    if "ic" not in df.columns:
        return None

    ic = pd.to_numeric(df["ic"], errors="coerce").dropna().to_numpy(dtype=float)
    ic = np.sort(ic[np.isfinite(ic)])
    if len(ic) == 0:
        return None

    stride_start = int(np.clip(stride_start, 0, len(ic) - 1))
    start_s = float(ic[stride_start] / fs_hz - pre_margin_s)
    return max(0.0, start_s)


# -----------------------------
# Data loading
# -----------------------------

def load_my_bmi160_csv(path: Path, signal_mode: str = "raw") -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["seq", "t_ms", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"My CSV missing required columns: {missing}. Found: {list(df.columns)}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[required].isna().any(axis=1).any():
        bad = int(df[required].isna().any(axis=1).sum())
        raise ValueError(f"My CSV contains {bad} rows with missing/non-numeric required values.")

    df["seq"] = df["seq"].astype(int)
    df["t_ms"] = df["t_ms"].astype(int)

    seq = df["seq"].to_numpy(dtype=int)
    expected = np.arange(seq.min(), seq.max() + 1)
    missing_seq = sorted(set(expected.tolist()) - set(seq.tolist()))
    if missing_seq:
        raise ValueError(f"My CSV has missing seq values. First few: {missing_seq[:20]}")

    # If a CALIBRATED file is passed but signal_mode=raw, use preserved raw columns.
    if signal_mode == "raw" and all(
        c in df.columns for c in ["ax_raw_g", "ay_raw_g", "az_raw_g", "gx_raw_dps", "gy_raw_dps", "gz_raw_dps"]
    ):
        ax_col, ay_col, az_col = "ax_raw_g", "ay_raw_g", "az_raw_g"
        gx_col, gy_col, gz_col = "gx_raw_dps", "gy_raw_dps", "gz_raw_dps"
    else:
        ax_col, ay_col, az_col = "ax_g", "ay_g", "az_g"
        gx_col, gy_col, gz_col = "gx_dps", "gy_dps", "gz_dps"

    out = pd.DataFrame()
    out["time_s"] = (df["t_ms"].astype(float) - float(df["t_ms"].iloc[0])) / 1000.0

    # BMI160: acceleration in g -> m/s².
    out["acc_x_mps2"] = df[ax_col].astype(float).to_numpy() * G
    out["acc_y_mps2"] = df[ay_col].astype(float).to_numpy() * G
    out["acc_z_mps2"] = df[az_col].astype(float).to_numpy() * G

    # BMI160: gyro already deg/s.
    out["gyr_x_degps"] = df[gx_col].astype(float).to_numpy()
    out["gyr_y_degps"] = df[gy_col].astype(float).to_numpy()
    out["gyr_z_degps"] = df[gz_col].astype(float).to_numpy()

    add_norms(out)

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
) -> Tuple[pd.DataFrame, float, Optional[pd.DataFrame]]:
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

    required = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    missing = [c for c in required if c not in imu.columns]
    if missing:
        raise ValueError(f"Dataset IMU missing columns: {missing}. Found: {list(imu.columns)}")

    out = pd.DataFrame()

    # The loaded data has a correct time axis. If index is seconds, use it.
    # Otherwise fall back to sample index / fs_hz.
    try:
        idx = imu.index.to_numpy(dtype=float)
        if len(idx) > 1 and np.all(np.diff(idx) > 0) and np.nanmedian(np.diff(idx)) < 0.1:
            out["time_s"] = idx - idx[0]
        else:
            out["time_s"] = np.arange(len(imu), dtype=float) / fs_hz
    except Exception:
        out["time_s"] = np.arange(len(imu), dtype=float) / fs_hz

    # Dataset/gaitmap convention: acceleration m/s², gyro deg/s.
    out["acc_x_mps2"] = imu["acc_x"].astype(float).to_numpy()
    out["acc_y_mps2"] = imu["acc_y"].astype(float).to_numpy()
    out["acc_z_mps2"] = imu["acc_z"].astype(float).to_numpy()

    out["gyr_x_degps"] = imu["gyr_x"].astype(float).to_numpy()
    out["gyr_y_degps"] = imu["gyr_y"].astype(float).to_numpy()
    out["gyr_z_degps"] = imu["gyr_z"].astype(float).to_numpy()

    add_norms(out)

    out.attrs["participant"] = participant
    out.attrs["test"] = test
    out.attrs["side"] = side
    out.attrs["sensor"] = sensor
    out.attrs["fs_hz"] = fs_hz
    out.attrs["sample_count"] = int(len(out))

    return out, fs_hz, trial.events_gt_imu


# -----------------------------
# Plotting
# -----------------------------

def plot_axiswise(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

    axes[0].plot(df["time_s"], df["acc_x_mps2"], label="acc_x")
    axes[0].plot(df["time_s"], df["acc_y_mps2"], label="acc_y")
    axes[0].plot(df["time_s"], df["acc_z_mps2"], label="acc_z")
    axes[0].set_ylabel("Acceleration [m/s²]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["time_s"], df["gyr_x_degps"], label="gyr_x")
    axes[1].plot(df["time_s"], df["gyr_y_degps"], label="gyr_y")
    axes[1].plot(df["time_s"], df["gyr_z_degps"], label="gyr_z")
    axes[1].set_ylabel("Gyroscope [deg/s]")
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
    ax.plot(mine["time_s"], mine[column], label="My BMI160 right instep transformed", linewidth=1.2)
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

    axes[1].plot(dataset["time_s"], zscore(dataset["gyr_norm_degps"].to_numpy()), label="Dataset gyro norm z-score")
    axes[1].plot(mine["time_s"], zscore(mine["gyr_norm_degps"].to_numpy()), label="My BMI160 gyro norm z-score")
    axes[1].set_ylabel("Gyro norm [z-score]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle("Normalized norm comparison: waveform shape only")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_axis_overlay(dataset: pd.DataFrame, mine: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14, 9))

    axes_list = ["x", "y", "z"]
    for i, ax_name in enumerate(axes_list):
        ax = axes[i, 0]
        ax.plot(dataset["time_s"], zscore(dataset[f"acc_{ax_name}_mps2"].to_numpy()), label=f"Dataset acc_{ax_name}")
        ax.plot(mine["time_s"], zscore(mine[f"acc_{ax_name}_mps2"].to_numpy()), label=f"My acc_{ax_name}")
        ax.set_ylabel(f"acc_{ax_name} z")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        ax = axes[i, 1]
        ax.plot(dataset["time_s"], zscore(dataset[f"gyr_{ax_name}_degps"].to_numpy()), label=f"Dataset gyr_{ax_name}")
        ax.plot(mine["time_s"], zscore(mine[f"gyr_{ax_name}_degps"].to_numpy()), label=f"My gyr_{ax_name}")
        ax.set_ylabel(f"gyr_{ax_name} z")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle("Axis-wise overlay after BMI160 → gaitmap-frame transformation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_window_diagnostics(
    dataset_full: pd.DataFrame,
    mine_full: pd.DataFrame,
    dataset_start_s: float,
    my_start_s: float,
    duration_s: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(dataset_full["time_s"], dataset_full["gyr_norm_degps"], label="Dataset full gyro norm")
    axes[0].axvspan(dataset_start_s, dataset_start_s + duration_s, alpha=0.25, label="selected window")
    axes[0].set_ylabel("Dataset gyro norm [deg/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(mine_full["time_s"], mine_full["gyr_norm_degps"], label="My full gyro norm")
    axes[1].axvspan(my_start_s, my_start_s + duration_s, alpha=0.25, label="selected window")
    axes[1].set_ylabel("My gyro norm [deg/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle("Window-selection diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Summary
# -----------------------------

def summary_stats(name: str, df: pd.DataFrame, fs_hz: float, start_s: float, axis_map: str = "") -> Dict[str, float | str]:
    duration_s = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if len(df) > 1 else np.nan
    return {
        "signal": name,
        "selected_start_s_before_crop": float(start_s),
        "samples": int(len(df)),
        "duration_s": duration_s,
        "fs_hz": float(fs_hz),
        "axis_map": axis_map,
        "acc_norm_mean_mps2": float(df["acc_norm_mps2"].mean()),
        "acc_norm_std_mps2": float(df["acc_norm_mps2"].std(ddof=1)),
        "acc_norm_max_mps2": float(df["acc_norm_mps2"].max()),
        "gyr_norm_mean_degps": float(df["gyr_norm_degps"].mean()),
        "gyr_norm_std_degps": float(df["gyr_norm_degps"].std(ddof=1)),
        "gyr_norm_max_degps": float(df["gyr_norm_degps"].max()),
    }


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare BMI160 right-instep raw data with SensorPositionComparison2019 r_instep using gaitmap sensor-frame reference."
    )

    p.add_argument("--data_folder", required=True, type=Path)
    p.add_argument("--core_dir", default=Path("."), type=Path)
    p.add_argument("--my_csv", default=None, type=Path)
    p.add_argument("--recordings_dir", default=Path("imu_recordings"), type=Path)

    p.add_argument("--participant", default="4d91")
    p.add_argument("--test", default="normal_10")
    p.add_argument("--side", default="right")
    p.add_argument("--sensor", default="r_instep")
    p.add_argument("--padding_s", default=3.0, type=float)

    p.add_argument("--my_signal", default="raw", choices=["raw", "calibrated"])
    p.add_argument("--my_axis_map", default="+x,+y,+z",
                   help="Map BMI160 axes to gaitmap axes. Default: gaitmap_x=+BMI160_z, gaitmap_y=+BMI160_y, gaitmap_z=+BMI160_x")

    p.add_argument("--duration_s", default=10.0, type=float)

    p.add_argument("--dataset_window", default="events", choices=["events", "active", "manual"])
    p.add_argument("--dataset_event_stride_start", default=0, type=int)
    p.add_argument("--dataset_start_s", default=None, type=float)

    p.add_argument("--my_window", default="active", choices=["active", "manual"])
    p.add_argument("--my_start_s", default=None, type=float)

    p.add_argument("--out_dir", default=Path("raw_instep_comparison_plots_v2"), type=Path)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    my_csv = args.my_csv if args.my_csv is not None else latest_csv(args.recordings_dir)
    print(f"[MY] Using CSV: {my_csv}")

    # Load my raw/corrected BMI160 data in native BMI160 frame.
    mine_native = load_my_bmi160_csv(my_csv, signal_mode=args.my_signal)
    my_fs = float(mine_native.attrs.get("effective_hz", np.nan))

    # Transform BMI160 physical casing frame into gaitmap-style sensor frame.
    mine_full = apply_axis_map(mine_native, args.my_axis_map)

    # Load dataset already in gaitmap coordinate system.
    dataset_full, dataset_fs, events_gt_imu = load_dataset_trial(
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
    print(f"[MY] BMI160 -> gaitmap axis map: {args.my_axis_map}")

    # Pick dataset crop.
    dataset_start_s = args.dataset_start_s
    if args.dataset_window == "manual":
        if dataset_start_s is None:
            raise ValueError("--dataset_window manual requires --dataset_start_s")
    elif args.dataset_window == "events":
        dataset_start_s = event_based_dataset_start(
            events_gt_imu=events_gt_imu,
            fs_hz=dataset_fs,
            duration_s=args.duration_s,
            stride_start=args.dataset_event_stride_start,
        )
        if dataset_start_s is None:
            print("[DATASET] Event-based window failed; falling back to active gyro window.")
            dataset_start_s = robust_active_start(dataset_full, dataset_fs, args.duration_s, prefer_after_s=0.0)
    else:
        dataset_start_s = robust_active_start(dataset_full, dataset_fs, args.duration_s, prefer_after_s=0.0)

    # Pick my crop.
    my_start_s = args.my_start_s
    if args.my_window == "manual":
        if my_start_s is None:
            raise ValueError("--my_window manual requires --my_start_s")
    else:
        my_start_s = robust_active_start(mine_full, my_fs, args.duration_s, prefer_after_s=0.0)

    print(f"[WINDOW] dataset_start_s={dataset_start_s:.3f}, my_start_s={my_start_s:.3f}, duration_s={args.duration_s:.3f}")

    dataset = crop_by_time(dataset_full, float(dataset_start_s), args.duration_s)
    mine = crop_by_time(mine_full, float(my_start_s), args.duration_s)

    # Save processed windows.
    dataset_window_csv = args.out_dir / "dataset_4d91_normal10_r_instep_window_gaitmap_frame.csv"
    my_window_csv = args.out_dir / "my_bmi160_right_instep_window_transformed_gaitmap_frame.csv"
    dataset.to_csv(dataset_window_csv, index=False)
    mine.to_csv(my_window_csv, index=False)

    # Plots.
    plot_window_diagnostics(
        dataset_full,
        mine_full,
        float(dataset_start_s),
        float(my_start_s),
        args.duration_s,
        args.out_dir / "00_window_selection_diagnostics.png",
    )

    plot_axiswise(
        dataset,
        f"Dataset gaitmap frame: {args.participant} {args.test} {args.sensor}",
        args.out_dir / "01_dataset_axiswise_gaitmap_frame.png",
    )

    plot_axiswise(
        mine,
        f"My BMI160 transformed to gaitmap frame ({args.my_signal}, map={args.my_axis_map})",
        args.out_dir / "02_my_bmi160_axiswise_gaitmap_frame.png",
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
        args.out_dir / "04_gyr_norm_comparison_degps.png",
        "gyr_norm_degps",
        "Gyroscope norm [deg/s]",
    )

    plot_zscore_norm_compare(
        dataset,
        mine,
        args.out_dir / "05_zscore_norm_shape_comparison.png",
    )

    plot_axis_overlay(
        dataset,
        mine,
        args.out_dir / "06_axis_overlay_gaitmap_frame.png",
    )

    summary = pd.DataFrame([
        summary_stats(
            "dataset_4d91_normal10_r_instep_gaitmap_frame",
            dataset,
            dataset_fs,
            float(dataset_start_s),
            axis_map="already gaitmap frame on loading",
        ),
        summary_stats(
            "my_bmi160_right_instep_transformed_to_gaitmap_frame",
            mine,
            my_fs,
            float(my_start_s),
            axis_map=args.my_axis_map,
        ),
    ])
    summary_path = args.out_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    notes_path = args.out_dir / "interpretation_notes.txt"
    notes_path.write_text(
        "Raw visual comparison notes\\n"
        "===========================\\n\\n"
        "Dataset target: 4d91 / normal_10 / right / r_instep.\\n"
        "Dataset data are loaded through core.load_trial(), which uses SensorPositionComparison2019Mocap.\\n"
        "The gaitmap dataset loader transforms foot-mounted sensors into the gaitmap coordinate system.\\n\\n"
        "Unit handling:\\n"
        "- Acceleration: dataset m/s²; BMI160 g converted to m/s².\\n"
        "- Gyroscope: dataset deg/s; BMI160 deg/s.\\n"
        "- This v2 script no longer treats dataset gyro as rad/s.\\n\\n"
        "Axis handling:\\n"
        f"- BMI160 -> gaitmap axis map used: {args.my_axis_map}\\n"
        "- Default map assumes BMI160 +z is top/outward, +y is casing-left, +x is toe/front.\\n"
        "- Norm plots are still the safest comparison because exact sensor placement and shoe alignment can differ.\\n\\n"
        "Recommended figures for professor review:\\n"
        "1. 00_window_selection_diagnostics.png\\n"
        "2. 03_acc_norm_comparison.png\\n"
        "3. 04_gyr_norm_comparison_degps.png\\n"
        "4. 05_zscore_norm_shape_comparison.png\\n"
        "5. 06_axis_overlay_gaitmap_frame.png only if discussing axis mapping.\\n",
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    for path in [
        args.out_dir / "00_window_selection_diagnostics.png",
        args.out_dir / "01_dataset_axiswise_gaitmap_frame.png",
        args.out_dir / "02_my_bmi160_axiswise_gaitmap_frame.png",
        args.out_dir / "03_acc_norm_comparison.png",
        args.out_dir / "04_gyr_norm_comparison_degps.png",
        args.out_dir / "05_zscore_norm_shape_comparison.png",
        args.out_dir / "06_axis_overlay_gaitmap_frame.png",
        summary_path,
        notes_path,
    ]:
        print(f"  {path.resolve()}")

    print("\nMain review order:")
    print("  00: confirm the selected walking windows.")
    print("  03/04: compare raw norm magnitudes.")
    print("  05: compare waveform shape only.")
    print("  06: inspect axis-level similarity after BMI160 -> gaitmap mapping.")


if __name__ == "__main__":
    main()
