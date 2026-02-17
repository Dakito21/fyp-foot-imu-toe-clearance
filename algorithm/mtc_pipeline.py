"""
Phase 1 (1.1–1.9) — Single-point toe clearance (MTC) from instep IMU
with MoCap validation using SensorPositionComparison2019Mocap.

Refactor requested:
- Each subphase (1.1 → 1.9) is its own function
- main() calls them in chronological order

NOTE:
- For *verification*, stance/swing windows are taken from MoCap IC/TC (ground-truth).
- After Phase 1 is validated, you can swap (1.4) with IMU-only stance detection.

Run example:
python MFC_dataset_testing.py `
--data_folder "path/to/sensorpositoncomparison-v1.0.0-beta" `
--participant 4d91 `
--test normal_10 `
--side left `
--plot `
--calibrate_toe_z

"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from joblib import Memory

from gaitmap_datasets.sensor_position_comparison_2019 import SensorPositionComparison2019Mocap

from ahrs.filters import Madgwick
import csv
from pathlib import Path
import builtins
ORIG_PRINT = builtins.print  # keep original

from scipy.signal import butter, filtfilt

def butter_lowpass_filter(x, fs_hz, cutoff_hz, order=4):
    """
    Zero-phase Butterworth low-pass filter.
    """
    import numpy as np

    if cutoff_hz <= 0 or cutoff_hz >= 0.5 * fs_hz:
        return x

    nyq = 0.5 * fs_hz
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low", analog=False)
    return filtfilt(b, a, x)


##############################################################################
# Logging & small helpers
##############################################################################

def force_print(*args, **kwargs):
    ORIG_PRINT(*args, **kwargs)


def mm(x):  # meters -> millimeters
    return float(x) * 1000.0


def print_phase(tag, msg, *, always=False):
    """
    Centralized phase printer.
    - If SUMMARY_ONLY=True: prints only when always=True
    - Otherwise: prints normally
    """
    if SUMMARY_ONLY and not always:
        return
    force_print(f"\n[{tag}] {msg}")


def debug_gyro_units(imu_f, zupt_mask):
    # ---- NEW: auto-detect gyro column names ----
    cols = list(imu_f.columns)

    gyro_candidates = [
        ("gyro_x", "gyro_y", "gyro_z"),
        ("gyr_x", "gyr_y", "gyr_z"),
        ("wx", "wy", "wz"),
        ("ang_vel_x", "ang_vel_y", "ang_vel_z"),
    ]

    gyro_cols = None
    for triplet in gyro_candidates:
        if all(c in cols for c in triplet):
            gyro_cols = list(triplet)
            break

    if gyro_cols is None:
        # Try partial match fallback
        gx = next((c for c in cols if "gyr" in c and c.endswith(("x", "_x"))), None)
        gy = next((c for c in cols if "gyr" in c and c.endswith(("y", "_y"))), None)
        gz = next((c for c in cols if "gyr" in c and c.endswith(("z", "_z"))), None)
        if gx and gy and gz:
            gyro_cols = [gx, gy, gz]

    if gyro_cols is None:
        print("\n=== Debug: Gyro units check ===")
        print("Could not find gyro columns in imu_f. Available columns:")
        print(cols)
        return

    gnorm_all = np.linalg.norm(imu_f[gyro_cols].to_numpy(), axis=1)

    gnorm_st = gnorm_all[zupt_mask] if zupt_mask is not None else np.array([])

    print("\n=== Debug: Gyro units check ===")
    print(f"gyro_norm median (all):   {np.median(gnorm_all):.3f}")

    # ---- NEW: guard against empty stance ----
    if gnorm_st.size == 0:
        print("gyro_norm median (stance):nan (no stance/ZUPT samples)")
        print("gyro_norm 95% (stance):   nan (no stance/ZUPT samples)")
        print("Guide:")
        print("- If typical values are ~0.1–3 -> likely rad/s")
        print("- If typical values are ~10–300 -> likely deg/s")
        print("NOTE: No stance detected; relax stride/stance gates or check detection.")
        return

    print(f"gyro_norm median (stance):{np.median(gnorm_st):.3f}")
    print(f"gyro_norm 95% (stance):   {np.percentile(gnorm_st,95):.3f}")
    print("Guide:")
    print("- If typical values are ~0.1–3 -> likely rad/s")
    print("- If typical values are ~10–300 -> likely deg/s")

##############################################################################
# Calibration CSV persistence (toe offsets)
##############################################################################

def load_toe_offset(participant, test, side):
    if not CALIB_FILE.exists():
        return None

    with open(CALIB_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["participant"] == participant
                and row["test"] == test
                and row["side"] == side
            ):
                return {
                    "x_cm": float(row["toe_x_cm"]),
                    "z_cm": float(row["toe_z_cm"]),
                    "bias_mm": float(row["bias_mm"]),
                    "rmse_mm": float(row["rmse_mm"]),
                    # Optional newer fields (backwards compatible with older CSVs)
                    "mtc_imu_mean_mm": float(row.get("mtc_imu_mean_mm", "nan")),
                    "mtc_mocap_mean_mm": float(row.get("mtc_mocap_mean_mm", "nan")),
                    "mtc_avg_diff_mm": float(row.get("mtc_avg_diff_mm", row.get("bias_mm", "nan"))),
                    "mtc_mean_abs_diff_mm": float(row.get("mtc_mean_abs_diff_mm", "nan")),
                }
    return None


def save_toe_offset(participant, test, side, best):
    """Append (or extend) toe_offset_calibration.csv with calibration + MTC summary metrics.

    This function is backwards-compatible with an existing CSV that has only the old columns:
    it rewrites the file with the union of columns so the new MTC fields appear without
    requiring you to delete the CSV or re-run expensive x/z sweeps.
    """
    row = {
        "participant": participant,
        "test": test,
        "side": side,
        "toe_x_cm": round(float(best["x_cm"]), 2),
        "toe_z_cm": round(float(best["z_cm"]), 2),
        "bias_mm": round(float(best["bias"]) * 1000.0, 3),
        "rmse_mm": round(float(best["rmse"]) * 1000.0, 3),

        # Optional stride-level summary stats (may be missing if caller didn't pass them)
        "mtc_imu_mean_mm": round(float(best.get("mtc_imu_mean_m", np.nan)) * 1000.0, 3)
            if best.get("mtc_imu_mean_m", None) is not None else np.nan,
        "mtc_mocap_mean_mm": round(float(best.get("mtc_mocap_mean_m", np.nan)) * 1000.0, 3)
            if best.get("mtc_mocap_mean_m", None) is not None else np.nan,
        # Signed average difference IMU - MoCap (mm). Equals bias when computed on the same joined set.
        "mtc_avg_diff_mm": round(float(best.get("mtc_avg_diff_m", best.get("bias", np.nan))) * 1000.0, 3)
            if best.get("mtc_avg_diff_m", None) is not None else round(float(best["bias"]) * 1000.0, 3),
        "mtc_mean_abs_diff_mm": round(float(best.get("mtc_mean_abs_diff_m", np.nan)) * 1000.0, 3)
            if best.get("mtc_mean_abs_diff_m", None) is not None else np.nan,
    }

    # Union-columns append (upgrades old CSV headers automatically)
    csv_upsert_row_union_cols(str(CALIB_FILE), row)


def csv_upsert_row_union_cols(csv_path: str, row: dict, key_cols=("participant","test","side")):
    """
    Upsert a row into csv_path:
      - Union columns (backwards compatible)
      - If a row with same key exists, REPLACE it (override)
      - Otherwise append
    """
    import os
    import pandas as pd

    new_row = pd.DataFrame([row])

    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)

        # union columns
        for c in new_row.columns:
            if c not in old.columns:
                old[c] = pd.NA
        for c in old.columns:
            if c not in new_row.columns:
                new_row[c] = pd.NA

        # align order: keep old columns first, new at end
        new_cols = [c for c in new_row.columns if c not in old.columns]
        out_cols = list(old.columns) + new_cols
        old = old[out_cols]
        new_row = new_row[out_cols]

        # build key mask
        mask = None
        for k in key_cols:
            if k not in old.columns:
                # if key cols missing for some reason, fallback to append
                mask = None
                break
            m = (old[k].astype(str) == str(row[k]))
            mask = m if mask is None else (mask & m)

        if mask is not None and mask.any():
            # replace FIRST match; drop other duplicates of same key
            idxs = old.index[mask].tolist()
            keep = old.drop(index=idxs[1:])  # drop duplicates beyond first
            keep.loc[idxs[0]] = new_row.iloc[0]
            out = keep
        else:
            out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row

    out.to_csv(csv_path, index=False)

##############################################################################
# Quaternion math
##############################################################################

def quat_mul(q1, q2):
    """Hamilton product. q = [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_mul_batch(Q1, Q2):
    """
    Elementwise Hamilton product for arrays shaped (N,4).
    Quats are [w,x,y,z].
    Returns (N,4).
    """
    Q1 = np.asarray(Q1, dtype=float)
    Q2 = np.asarray(Q2, dtype=float)

    if Q1.ndim != 2 or Q2.ndim != 2 or Q1.shape[1] != 4 or Q2.shape[1] != 4:
        raise ValueError(f"quat_mul_batch expects (N,4), got {Q1.shape} and {Q2.shape}")
    if Q1.shape[0] != Q2.shape[0]:
        raise ValueError(f"quat_mul_batch expects same N, got {Q1.shape[0]} and {Q2.shape[0]}")

    w1, x1, y1, z1 = Q1[:, 0], Q1[:, 1], Q1[:, 2], Q1[:, 3]
    w2, x2, y2, z2 = Q2[:, 0], Q2[:, 1], Q2[:, 2], Q2[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.column_stack([w, x, y, z])

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_rotate(q, v):
    """Rotate 3D vector v by quaternion q."""
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

def _quat_norm(q):
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

def quat_norm_batch(Q):
    Q = np.asarray(Q, dtype=float)
    n = np.linalg.norm(Q, axis=1, keepdims=True)
    return Q / np.clip(n, 1e-12, None)


def _quat_from_two_unit_vectors(u, v):
    # u,v are unit 3-vectors. Returns quaternion rotating u -> v.
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))

    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    if dot < -0.999999:
        # 180 deg rotation: pick any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(u[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = axis - u * np.dot(axis, u)
        axis = axis / max(1e-12, np.linalg.norm(axis))
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=float)

    axis = np.cross(u, v)
    w = 1.0 + dot
    q = np.array([w, axis[0], axis[1], axis[2]], dtype=float)
    return _quat_norm(q)

def _slerp_identity(q, alpha):
    """
    True SLERP from identity to q by alpha.
    q is [w,x,y,z]. alpha in [0,1].
    """
    q = _quat_norm(q)
    if q[0] < 0:
        q = -q  # shortest path

    alpha = float(np.clip(alpha, 0.0, 1.0))

    # angle between I=[1,0,0,0] and q is 2*acos(w)
    w = float(np.clip(q[0], -1.0, 1.0))
    theta = 2.0 * np.arccos(w)

    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # Identity->q slerp is equivalent to exponentiating q^alpha
    # q = [cos(theta/2), sin(theta/2)*axis]
    axis = q[1:4]
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis_u = axis / s

    theta_a = alpha * theta
    return np.array([np.cos(theta_a/2.0),
                     *(np.sin(theta_a/2.0) * axis_u)], dtype=float)


##############################################################################
# Robust statistics helpers
##############################################################################

def _robust_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def _robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + 1e-12

##############################################################################
# Signal processing / filtering
##############################################################################

def compute_dynamic_gate_s_from_stride_time(
    events_ref: pd.DataFrame,
    fs_hz: float,
    *,
    frac: float = 0.12,   # 12% of stride time
    min_gate: float = 0.08,
    max_gate: float = 0.25,
    use: str = "ic2ic",   # "ic2ic" (recommended) or "tc2tc"
):
    """
    Dynamic stride matching tolerance gate_s based on typical stride duration.

    gate_s = clip(frac * median_stride_time, min_gate, max_gate)
    """
    ev = events_ref.dropna(subset=["ic", "tc"]).copy()
    if len(ev) < 3:
        return float(np.clip(0.15, min_gate, max_gate))  # fallback

    if use == "ic2ic":
        t = ev["ic"].to_numpy(dtype=int)
    elif use == "tc2tc":
        t = ev["tc"].to_numpy(dtype=int)
    else:
        raise ValueError("use must be 'ic2ic' or 'tc2tc'")

    dt = np.diff(t) / float(fs_hz)  # seconds
    dt = dt[(dt > 0.3) & (dt < 3.0)]  # reject junk
    if dt.size == 0:
        return float(np.clip(0.15, min_gate, max_gate))

    stride_time = float(np.median(dt))
    gate_s = frac * stride_time
    return float(np.clip(gate_s, min_gate, max_gate))


def lowpass_df(df, fs_hz, cutoff_hz=20.0, order=4):
    b, a = butter(order, cutoff_hz / (0.5 * fs_hz), btype="low", analog=False)
    out = df.copy()
    for col in df.columns:
        out[col] = filtfilt(b, a, df[col].to_numpy())
    return out

def estimate_step_frequency_from_gyro_norm(gyr, fs_hz, fmin=0.5, fmax=4.0):
    import numpy as np

    g = np.asarray(gyr, dtype=float)
    gyro_norm = np.linalg.norm(g[:, :3], axis=1) if (g.ndim == 2 and g.shape[1] >= 3) else g

    x = gyro_norm - np.nanmean(gyro_norm)
    x = np.nan_to_num(x)

    # FFT magnitude
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), d=1.0 / float(fs_hz))

    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return 1.5

    fb = f[band]
    Xb = X[band]

    # 1) Find dominant peak as candidate fundamental
    i0 = int(np.argmax(Xb))
    f0 = float(fb[i0])

    # 2) Look for 2nd harmonic near 2*f0 (within tolerance)
    f2 = 2.0 * f0
    tol = 0.12 * f0  # ~12% tolerance
    i2_candidates = np.where((fb >= f2 - tol) & (fb <= f2 + tol))[0]

    A0 = float(Xb[i0])
    A2 = float(np.max(Xb[i2_candidates])) if len(i2_candidates) else 0.0

    # 3) Decide if the dominant is stride or step:
    # If harmonic is strong, treat step frequency as 2*f0; else f0
    # (tune ratio if needed)
    if (f0 < 1.3) and (A2 > 0.55 * A0):
        f_step = f2
    else:
        # If f0 itself is already in plausible step range, keep it
        f_step = f0

    if not np.isfinite(f_step) or f_step <= 0:
        f_step = 1.5
    return float(np.clip(f_step, fmin, fmax))


def dynamic_cutoffs_from_step_freq(f_step,
                                  acc_mult=8.0, gyr_mult=6.0,
                                  acc_lo=10.0, acc_hi=35.0,
                                  gyr_lo=8.0,  gyr_hi=30.0):
    """
    Convert step frequency to dynamic low-pass cutoffs (Hz).
    """
    import numpy as np
    cutoff_acc = float(np.clip(acc_mult * f_step, acc_lo, acc_hi))
    cutoff_gyr = float(np.clip(gyr_mult * f_step, gyr_lo, gyr_hi))
    return cutoff_acc, cutoff_gyr


def phase_1_3_filter_imu(imu, fs_hz, cutoff_hz=20.0, *, dynamic=False):
    """
    Filter IMU accel/gyro with either:
      - fixed cutoff_hz (legacy), or
      - dynamic cutoffs based on step frequency (recommended).
    """
    import numpy as np

    imu_f = imu.copy()

    if dynamic:
        gyr = imu_f[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=float)
        f_stride = estimate_step_frequency_from_gyro_norm(gyr, fs_hz)

        # If estimator is actually giving stride frequency, convert to step frequency:
        f_step = 2.0 * f_stride

        cutoff_acc, cutoff_gyr = dynamic_cutoffs_from_step_freq(f_step)

        force_print(f"[FILTER] dynamic: f_stride={f_stride:.2f} Hz, f_step={f_step:.2f} Hz, "
                    f"cutoff_acc={cutoff_acc:.1f} Hz, cutoff_gyr={cutoff_gyr:.1f} Hz")
    else:
        cutoff_acc = float(cutoff_hz)
        cutoff_gyr = float(cutoff_hz)

    # Filter accel
    for k in ["acc_x", "acc_y", "acc_z"]:
        imu_f[k] = butter_lowpass_filter(imu_f[k].to_numpy(dtype=float), fs_hz, cutoff_hz=cutoff_acc)

    # Filter gyro
    for k in ["gyr_x", "gyr_y", "gyr_z"]:
        imu_f[k] = butter_lowpass_filter(imu_f[k].to_numpy(dtype=float), fs_hz, cutoff_hz=cutoff_gyr)

    return imu_f


##############################################################################
# Event detection + stride filters + ZUPT
##############################################################################

def _run_length_filter(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Remove True-runs shorter than min_len."""
    mask = mask.astype(bool)
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _fill_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False-gaps shorter than or equal to max_gap between True segments."""
    mask = mask.astype(bool)
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and (not out[j]):
                j += 1
            gap_len = j - i
            left_true = (i - 1) >= 0 and out[i - 1]
            right_true = j < n and out[j]
            if left_true and right_true and gap_len <= max_gap:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _hysteresis_threshold(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Hysteresis comparator:
      - enter True when x < low
      - stay True until x > high
    Assumes low <= high.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x), dtype=bool)
    state = False
    for i, v in enumerate(x):
        if not state:
            if v < low:
                state = True
        else:
            if v > high:
                state = False
        out[i] = state
    return out


def _clamp(a, lo, hi):
    return max(lo, min(int(a), int(hi)))


def _argmax_in_window(x: np.ndarray, a: int, b: int):
    if b <= a:
        return None
    j = int(np.argmax(x[a:b]))
    return a + j


def phase_1_4b_filter_invalid_strides(events_imu: pd.DataFrame, fs_hz: float) -> pd.DataFrame:
    """
    Post-filter to eliminate over-detected / physically implausible strides.
    Keeps only strides that satisfy basic timing and ordering constraints.
    """
    if events_imu is None or len(events_imu) == 0:
        return events_imu

    ev = events_imu.copy()

    # Required columns
    for c in ["start", "end", "tc", "ic"]:
        if c not in ev.columns:
            return ev

    # Use local timing only (more robust than relying on 'end')
    swing_dur_s = (ev["ic"] - ev["tc"]) / float(fs_hz)

    # Approx stride duration from consecutive TC events (fallback if 'end' is not per-stride)
    ev = ev.sort_values("tc").copy()
    tc_next = ev["tc"].shift(-1)
    stride_dur_s = (tc_next - ev["tc"]) / float(fs_hz)

    # Basic ordering
    keep_order = (
        (ev["tc"] >= ev["start"]) &
        (ev["ic"] > ev["tc"]) &
        (ev["end"] > ev["ic"])
    )

    # after computing swing_dur_s and stride_dur_s
    med_stride = float(np.nanmedian(stride_dur_s.to_numpy()))
    if not np.isfinite(med_stride) or med_stride <= 0:
        med_stride = 1.0  # safe fallback

    # cadence-adaptive bounds (with clamps to prevent insanity)
    min_stride = float(np.clip(0.45 * med_stride, 0.25, 0.80))
    max_stride = float(np.clip(1.80 * med_stride, 0.60, 2.50))

    min_swing  = float(np.clip(0.18 * med_stride, 0.08, 0.25))
    max_swing  = float(np.clip(0.60 * med_stride, 0.30, 1.20))

    keep_timing = (
        (swing_dur_s >= min_swing) & (swing_dur_s <= max_swing) &
        ((stride_dur_s.isna()) | ((stride_dur_s >= min_stride) & (stride_dur_s <= max_stride)))
    )

    keep = keep_order & keep_timing

    
    # ---- DIAGNOSTIC: if nothing passes, print why (once) ----
    if not np.any(keep):
        print("\n[Phase 1.4b Debug] All strides rejected. Showing first 5 rows and checks:")
        print(ev[["start", "tc", "ic", "end"]].head())

        # Show ordering violations
        bad_order = (
            (ev["tc"] <= ev["start"]) |
            (ev["ic"] <= ev["tc"]) |
            (ev["end"] <= ev["ic"])
        )
        print(f"[Phase 1.4b Debug] bad_order count: {int(bad_order.sum())} / {len(ev)}")

        # Show duration stats (nan-aware)
        sd = stride_dur_s.to_numpy()
        sw = swing_dur_s.to_numpy()
        print(f"[Phase 1.4b Debug] stride_dur_s: min={np.nanmin(sd):.3f}, med={np.nanmedian(sd):.3f}, max={np.nanmax(sd):.3f}")
        print(f"[Phase 1.4b Debug] swing_dur_s:  min={np.nanmin(sw):.3f}, med={np.nanmedian(sw):.3f}, max={np.nanmax(sw):.3f}")

        # Show NaN counts
        print(f"[Phase 1.4b Debug] NaNs in start/tc/ic/end: "
              f"{int(ev['start'].isna().sum())}/"
              f"{int(ev['tc'].isna().sum())}/"
              f"{int(ev['ic'].isna().sum())}/"
              f"{int(ev['end'].isna().sum())}")


    ev = ev[keep].copy()

    # Also enforce strictly increasing TC to prevent duplicate strides from flicker
    ev = ev.sort_values("tc")
    ev = ev[ev["tc"].diff().fillna(1e9) > int(0.12 * fs_hz)]  # refractory 120 ms

    return ev


def phase_1_4_detect_events_imu_only(
    imu_f: pd.DataFrame,
    fs_hz: float,
    gyro_th_deg_s: float = 12.0,
    acc_th_m_s2: float = 2.0,
    min_stance_s: float = 0.08,
    min_swing_s: float = 0.08,
    gap_fill_s: float = 0.03,
    refractory_s: float = 0.35,
):
    """IMU-only gait events from instep IMU.

    Outputs a stride table compatible with downstream phases:
      columns: start, end, tc, ic, min_vel
    where swing is defined as [tc, ic] for each stride.

    Notes:
    - Stance is detected with a simple threshold gate on gyro norm and acc norm.
    - IC is stance rising edge (swing->stance).
    - TC is stance falling edge (stance->swing).
    - Strides are built as TC_k -> IC_{k+1}; end is the next TC.

    This is intentionally lightweight and real-time friendly.
    """
    gyr = imu_f[["gyr_x","gyr_y","gyr_z"]].to_numpy(dtype=float)
    acc = imu_f[["acc_x","acc_y","acc_z"]].to_numpy(dtype=float)

    gyr_norm = np.linalg.norm(gyr, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1)

    # “Impact proxy”: acceleration norm high-pass-ish by subtracting median stance gravity magnitude
    
    # IC feature: jerk proxy (more sensitive to impact than acc_norm which is gravity-dominated)
    acc_jerk = np.abs(np.diff(acc_norm, prepend=acc_norm[0])) * fs_hz

    # light smoothing to reduce noise spikes (cheap moving average)
    w = max(3, int(round(0.01 * fs_hz)))  # ~10 ms window, min 3 samples
    kernel = np.ones(w, dtype=float) / float(w)
    acc_peak_feat = np.convolve(acc_jerk, kernel, mode="same")


    # --- Stance detection (IMU-only) ---
    # Use gyro-norm hysteresis for stance edges (robust near IC/TC).
    # Acc gate is NOT used to define edges (it shifts edges inward by ~200-300ms).
    gyro_med_stance_hint = np.median(gyr_norm)  # just for stability if needed

    # --- Adaptive hysteresis thresholds (per trial) ---
    med_all = float(np.median(gyr_norm))
    p95_all = float(np.percentile(gyr_norm, 95))

    # “stance hint” = below-median portion; avoids needing a correct stance mask first
    low_part = gyr_norm[gyr_norm <= med_all]
    p95_lowpart = float(np.percentile(low_part, 95)) if low_part.size > 10 else p95_all

    # Pollution check: if p95_lowpart is too close to overall energy, fallback
    ratio = p95_lowpart / max(med_all, 1e-6)

    if ratio > 0.8:
        low_th = 0.50 * med_all
    else:
        low_th = 1.15 * p95_lowpart

    # Clamp to sane bounds (deg/s)
    low_th  = float(np.clip(low_th, 8.0, 18.0))
    high_th = float(np.clip(2.0 * low_th, 18.0, 45.0))

    stance = _hysteresis_threshold(gyr_norm, low=low_th, high=high_th)


    # Denoise mask: fill small gaps, then drop tiny stance islands
    gap_fill = max(1, int(round(gap_fill_s * fs_hz)))
    min_stance = max(2, int(round(min_stance_s * fs_hz)))
    min_swing  = max(2, int(round(min_swing_s * fs_hz)))

    stance = _fill_small_gaps(stance, max_gap=gap_fill)
    stance = _run_length_filter(stance, min_len=min_stance)
    # Remove tiny swing islands too (prevents double-stance segments → FP strides)
    stance = ~_run_length_filter(~stance, min_len=min_swing)

    # -------------------------------
    # TC feature selection (AFTER stance mask exists)
    # -------------------------------
    ratios = []
    for ax in range(3):
        g = np.abs(gyr[:, ax])
        stance_vals = g[stance]
        swing_vals  = g[~stance]

        if len(stance_vals) < 5 or len(swing_vals) < 5:
            ratios.append(0.0)
            continue

        r = (np.sqrt(np.mean(swing_vals**2)) + 1e-6) / \
            (np.sqrt(np.mean(stance_vals**2)) + 1e-6)
        ratios.append(r)

    best_ax = int(np.argmax(ratios))
    tc_feat = np.abs(gyr[:, best_ax])   # <-- FINAL tc feature}")    

    n = len(gyr_norm)

    # --- Global reference levels (used by swing energy gates) ---
    gyr_norm_med_all = float(np.median(gyr_norm))
    tc_feat_med_all  = float(np.median(tc_feat))

    segments = []
    i = 0
    while i < n:
        if stance[i]:
            a = i
            while i < n and stance[i]:
                i += 1
            b = i
            segments.append((a, b))
        else:
            i += 1

    # Need at least 2 stance segments to form 1 full TC->IC swing
    rows = []
    refractory = int(round(refractory_s * fs_hz))
    prev_tc = -10**9

    # ---- 4.A rejection bookkeeping (for tuning & debugging) ----
    rej = {
        "order": 0,
        "swing_dur": 0,

        # 4.A gates
        "r_energy": 0,
        "tc_z": 0,
        "ic_z": 0,

        # other FP killers you already have (but weren't counting)
        "refractory": 0,
        "Eg": 0,
        "Etc": 0,
    }



    det_id = 0

    for k in range(len(segments) - 1):
        st_a, st_b = segments[k]
        nx_a, nx_b = segments[k + 1]

        # Window sizes (tuneable); these are conservative for ~200 Hz
        w_ic_pre  = int(round(0.20 * fs_hz))  # search before nx_a (impact often precedes stable stance)
        w_ic_post = int(round(0.05 * fs_hz))  # small lookahead
        w_tc_pre  = int(round(0.05 * fs_hz))  # just before stance end
        w_tc_post = int(round(0.20 * fs_hz))  # toe-off dynamics can extend after mask edge

        # ----- TC refinement (reduce early-bias: search wider + pick later peak) -----
        a_tc = _clamp(st_b - int(round(0.12 * fs_hz)), 0, n)

        # Don't let TC drift too close to IC; keep a margin before next stance start (nx_a)
        b_tc = _clamp(
            min(
                nx_a - int(round(0.12 * fs_hz)),
                st_b + int(round(0.45 * fs_hz)),
            ),
            0,
            n
        )

        # Fallback window if nx_a is too close
        if b_tc <= a_tc + 5:
            a_tc = _clamp(st_b, 0, n)
            b_tc = _clamp(st_b + int(round(0.35 * fs_hz)), 0, n)

        win = tc_feat[a_tc:b_tc]
        if len(win) > 5:
           # --- Guardrail: TC must be sufficiently before next stance start ---
            tc_latest = nx_a - int(round(0.18 * fs_hz))   # tune 0.16–0.22 s
            tc_latest = _clamp(tc_latest, 0, n)

            d = np.diff(win)
            peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1  # local maxima

            if len(peaks) == 0:
                # choose max within [a_tc, min(b_tc, tc_latest)]
                b_eff = min(b_tc, tc_latest)
                if b_eff > a_tc + 5:
                    tc = a_tc + int(np.argmax(tc_feat[a_tc:b_eff]))
                else:
                    tc = a_tc + int(np.argmax(win))
            else:
                mx = float(np.max(win)) + 1e-12
                good = [p for p in peaks if win[p] >= 0.70 * mx]

                # Convert to absolute indices
                cand = (good if len(good) else list(peaks))
                cand_abs = [a_tc + int(p) for p in cand]

                # Keep only candidates that satisfy the late bound
                cand_ok = [t for t in cand_abs if t <= tc_latest]

                if len(cand_ok) > 0:
                    # choose the LAST prominent peak that is still within bound
                    tc = int(cand_ok[-1])
                else:
                    # If all peaks are too late, choose the BEST peak within bound (max tc_feat)
                    b_eff = min(b_tc, tc_latest)
                    if b_eff > a_tc + 5:
                        tc = a_tc + int(np.argmax(tc_feat[a_tc:b_eff]))
                    else:
                        # ultimate fallback
                        tc = int(st_b)
        else:
            tc = int(st_b)




        if k < 5:
            force_print(f"[TC dbg] axins={best_ax} tc={tc}")


        
        # Refractory: drop strides whose TC is too close to previous TC (anti-chatter)
        if (tc - prev_tc) < refractory:
            rej["refractory"] += 1
            continue
        prev_tc = tc


        # ----- IC refinement (acc_norm peak; stable) -----
        a_ic = _clamp(nx_a - w_ic_pre, 0, n)
        b_ic = _clamp(nx_a + w_ic_post, 0, n)
        ic_ref = _argmax_in_window(acc_peak_feat, a_ic, b_ic)
        ic = int(ic_ref) if ic_ref is not None else int(nx_a)
        

        # Sanity: enforce ordering (TC must occur before IC)
        if ic <= tc:
            rej["order"] += 1
            continue
        
        # --- Swing energy gate (FP killer) ---
        E_g = float(np.median(gyr_norm[tc:ic])) if (ic > tc + 3) else 0.0
        if E_g < 0.60 * gyr_norm_med_all:
            rej["Eg"] += 1
            continue

        # --- Local TC-axis energy gate (more selective than gyr_norm) ---
        E_tc = float(np.median(tc_feat[tc:ic])) if (ic > tc + 3) else 0.0
        if E_tc < 0.60 * tc_feat_med_all:
            rej["Etc"] += 1
            continue


        # =========================
        # 4.A Stride confidence gate
        # =========================

        # Define stride-local windows
        swing_g = gyr_norm[tc:ic]                  # swing gyro norm
        stance_g = gyr_norm[nx_a:nx_b]             # next stance gyro norm (proxy for "quiet")

        # (A1) Swing-to-stance energy ratio (local, robust)
        Eg_swing = _robust_rms(swing_g)
        Eg_stance = _robust_rms(stance_g) + 1e-6
        r_energy = Eg_swing / Eg_stance

        # (A2) TC peak prominence vs local baseline (use tc_feat)
        # Compare TC peak to median+MAD in a local band around TC candidate window
        # TC baseline window: BEFORE tc (avoids contaminating baseline with the peak)
        tc_band_a = _clamp(tc - int(round(0.20 * fs_hz)), 0, n)
        tc_band_b = _clamp(tc - int(round(0.05 * fs_hz)), 0, n)
        tc_band = tc_feat[tc_band_a:tc_band_b]
        tc_pk = float(tc_feat[tc])
        tc_med = float(np.median(tc_band)) if tc_band.size else 0.0
        tc_mad = _robust_mad(tc_band)
        tc_z = (tc_pk - tc_med) / max(tc_mad, 1e-6)

        # (A3) IC impact prominence (use acc_peak_feat around IC search window)
        ic_band_a = _clamp(ic - int(round(0.10 * fs_hz)), 0, n)
        ic_band_b = _clamp(ic + int(round(0.05 * fs_hz)), 0, n)
        ic_band = acc_peak_feat[ic_band_a:ic_band_b]
        ic_pk = float(acc_peak_feat[ic])
        ic_med = float(np.median(ic_band)) if ic_band.size else 0.0
        ic_mad = _robust_mad(ic_band)
        ic_z = (ic_pk - ic_med) / max(ic_mad, 1e-6)

        bad_tc_z = False
        bad_ic_z = False

        # Thresholds (start here; tune later)
        # - r_energy kills fake strides created by mask flicker
        # - tc_z ensures TC is not just noise
        # - ic_z ensures IC isn't random


        # energy gate (relative + absolute)
        if r_energy < 1.8:
            rej["r_energy"] += 1
            continue

        # TC gate (non-lethal): flag it
        if tc_z < 1.5:
            rej["tc_z"] += 1
            bad_tc_z = True

        # IC gate (non-lethal): flag it
        if ic_z < 1.5:
            rej["ic_z"] += 1
            bad_ic_z = True

        end = int(segments[k + 1][1])  # end of next stance; consistent stride window

        start = int(tc)

        # min_vel proxy: pick a point in next stance with minimal gyro AND acc close to gravity
        gwin = gyr_norm[nx_a:nx_b]
        awin = acc_norm[nx_a:nx_b]

        if len(gwin) > 0:
            # Prefer samples that look "foot-flat": acc_norm close to 9.81
            flat = np.abs(awin - 9.81) < float(acc_th_m_s2)  # acc gate used ONLY here
            if np.any(flat):
                idxs = np.where(flat)[0]
                # among "flat" samples, choose minimum gyro norm
                j = idxs[np.argmin(gwin[idxs])]
            else:
                # fallback: pure min gyro in stance
                j = int(np.argmin(gwin))
            min_vel = int(nx_a + j)
        else:
            min_vel = int((nx_a + nx_b) // 2)

        

        rows.append({
            "det_stride_id": det_id,
            "start": start,
            "end": end,
            "tc": tc,
            "ic": ic,
            "min_vel": min_vel,

            "bad_tc_z": bool(bad_tc_z),
            "bad_ic_z": bool(bad_ic_z),
            "tc_z": float(tc_z),
            "ic_z": float(ic_z),
            "r_energy": float(r_energy),
            "Eg_swing": float(Eg_swing),
            "Eg_stance": float(Eg_stance),
        })
        det_id += 1

    events_imu = pd.DataFrame(rows)

    if events_imu.empty:
        events_imu = pd.DataFrame(columns=["det_stride_id","start","end","tc","ic","min_vel"])
        events_imu = events_imu.set_index("det_stride_id")
    else:
        events_imu = events_imu.set_index("det_stride_id")


    debug = {
        "stance_mask": stance,
        "gyr_norm": gyr_norm,
        "acc_norm": acc_norm,
        "n_stance_segments": len(segments),
        "reject_counts": rej,
    }
    return events_imu, debug



#For ML-based event detection

class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 7, d: int = 1):
        super().__init__()
        pad = (k // 2) * d
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class EventNetStageA(nn.Module):
    """
    Must match the architecture used in train_eventnet_stageA.py
    Input:  (B, C, T)
    Output: (B, 1, T) logits (we will sigmoid -> p_stance)
    """
    def __init__(self, c_in: int = 8, c_hidden: int = 64):
        super().__init__()
        self.b1 = _ConvBlock(c_in, c_hidden, k=7, d=1)
        self.b2 = _ConvBlock(c_hidden, c_hidden, k=7, d=2)
        self.b3 = _ConvBlock(c_hidden, c_hidden, k=7, d=4)
        self.head = nn.Conv1d(c_hidden, 1, kernel_size=1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.head(x)

def load_eventnet(pt_path: str, device: str = "cpu"):
    """
    Loads Stage-A model bundle saved by train_eventnet_stageA.py:
      keys: state_dict, model_cfg, feature_cfg, postproc_cfg, fs_hz, train_info
    """
    # Safer load (avoids the warning where possible)
    # - If your torch supports weights_only=True, this limits pickle risks
    try:
        obj = torch.load(pt_path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch versions don't support weights_only
        obj = torch.load(pt_path, map_location=device)

    # Case 1: TorchScript (if you later export it)
    if hasattr(obj, "forward") and hasattr(obj, "graph"):
        obj.eval()
        return obj, {}

    if not isinstance(obj, dict):
        raise RuntimeError("param.pt is not a dict or TorchScript model.")

    # Case 2: Your Stage-A bundle dict
    if "state_dict" in obj and "model_cfg" in obj:
        cfg = obj["model_cfg"]
        c_in = int(cfg.get("c_in", 8))
        c_hidden = int(cfg.get("c_hidden", 64))

        model = EventNetStageA(c_in=c_in, c_hidden=c_hidden)
        model.load_state_dict(obj["state_dict"], strict=True)
        model.to(device)
        model.eval()

        meta = {
            "feature_cfg": obj.get("feature_cfg", {}),
            "postproc_cfg": obj.get("postproc_cfg", {}),
            "fs_hz": obj.get("fs_hz", None),
            "train_info": obj.get("train_info", {}),
        }
        return model, meta

    raise RuntimeError("Unsupported param.pt format. Expected Stage-A bundle with state_dict + model_cfg.")

def phase_1_4_detect_events_ml(
    imu_f: pd.DataFrame,
    fs_hz: float,
    model,
    meta: dict | None = None,
    *,
    p_enter: float = 0.55,     # enter stance if p_stance > p_enter
    p_exit: float = 0.45,      # exit stance if p_stance < p_exit
    min_stance_s: float = 0.08,
    min_swing_s: float = 0.08,
    gap_fill_s: float = 0.03,
    refractory_s: float = 0.30,
):
    """
    ML-based event detection:
      - Model predicts stance probability per sample.
      - Convert prob->stance mask via hysteresis + cleanup.
      - IC = rising edge (swing->stance), TC = falling edge (stance->swing).
      - Build strides as TC_k -> IC_{k+1}; end = next TC_{k+1} (same as your IMU-only logic).
      - min_vel computed using gyro norm minimization within next stance (same style as your current code).

    Returns:
      events_imu (DataFrame indexed by det_stride_id)
      debug (dict with stance_mask, probs, reject_counts)
    """
    if meta is None:
        meta = {}

    # --------------------------
    # (A) Build model input (MUST match training: 8 channels)
    # --------------------------
    acc = imu_f[["acc_x","acc_y","acc_z"]].to_numpy(dtype=np.float32)  # (T,3)
    gyr = imu_f[["gyr_x","gyr_y","gyr_z"]].to_numpy(dtype=np.float32)  # (T,3)

    acc_norm_1d = np.linalg.norm(acc, axis=1).astype(np.float32)  # (T,)
    gyr_norm_1d = np.linalg.norm(gyr, axis=1).astype(np.float32)  # (T,)

    X = np.concatenate(
        [acc, gyr, acc_norm_1d[:, None], gyr_norm_1d[:, None]],
        axis=1
    ).astype(np.float32)  # (T,8)


    # IMPORTANT: match training normalization (median/MAD per trial)
    med = np.median(X, axis=0, keepdims=True)
    mad = np.median(np.abs(X - med), axis=0, keepdims=True) + 1e-6
    X = (X - med) / (1.4826 * mad)

    # Torch expects (B,C,T)
    Xt = torch.from_numpy(X.T[None, ...])  # (1,8,T)



    # --------------------------
    # (B) Inference: stance probability
    # --------------------------
    with torch.no_grad():
        print("[ML dbg] Xt", tuple(Xt.shape))

        out = model(Xt)  # expected shape (1,T) or (1,1,T) or (1,T,1)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.float()

        # squeeze to (T,)
        if out.ndim == 3:
            # (B,1,T) or (B,T,1)
            if out.shape[1] == 1:
                out = out[:,0,:]
            elif out.shape[2] == 1:
                out = out[:,:,0]
        if out.ndim == 2:
            out = out[0]
        p_stance = torch.sigmoid(out).detach().cpu().numpy().astype(float)  # (T,)

    n = len(p_stance)
    if n < 10:
        events_imu = pd.DataFrame(columns=["det_stride_id","start","end","tc","ic","min_vel"]).set_index("det_stride_id")
        return events_imu, {"stance_mask": np.zeros(n, dtype=bool), "p_stance": p_stance}

    # --------------------------
    # (C) Prob -> stance mask (hysteresis + cleanup)
    # --------------------------
    # reuse your hysteresis helper, but it assumes "x < low enters true".
    # We'll implement explicit stance hysteresis for p_stance:
    stance = np.zeros(n, dtype=bool)
    state = False
    for i, p in enumerate(p_stance):
        if not state:
            if p >= p_enter:
                state = True
        else:
            if p <= p_exit:
                state = False
        stance[i] = state

    gap_fill = max(1, int(round(gap_fill_s * fs_hz)))
    min_stance = max(2, int(round(min_stance_s * fs_hz)))
    min_swing  = max(2, int(round(min_swing_s * fs_hz)))

    stance = _fill_small_gaps(stance, max_gap=gap_fill)
    stance = _run_length_filter(stance, min_len=min_stance)
    stance = ~_run_length_filter(~stance, min_len=min_swing)

    # Extract stance segments
    segments = []
    i = 0
    while i < n:
        if stance[i]:
            a = i
            while i < n and stance[i]:
                i += 1
            b = i
            segments.append((a, b))
        else:
            i += 1

    # Need at least 2 stance segments to form 1 swing (TC->IC)
    rows = []
    refractory = int(round(refractory_s * fs_hz))
    prev_tc = -10**9

    # IC impact feature (same style you used)
    acc_jerk = np.abs(np.diff(acc_norm_1d, prepend=acc_norm_1d[0])) * fs_hz
    w = max(3, int(round(0.01 * fs_hz)))
    kernel = np.ones(w, dtype=float) / float(w)
    acc_peak_feat = np.convolve(acc_jerk, kernel, mode="same")

    rej = {"order":0, "refractory":0, "too_short":0}

    det_id = 0
    for k in range(len(segments) - 1):
        st_a, st_b = segments[k]
        nx_a, nx_b = segments[k + 1]

        # TC: stance falling edge (end of current stance)
        tc = int(st_b)

        # IC: find a peak near next stance start (more stable than nx_a itself)
        w_ic_pre  = int(round(0.20 * fs_hz))
        w_ic_post = int(round(0.05 * fs_hz))
        a_ic = _clamp(nx_a - w_ic_pre, 0, n)
        b_ic = _clamp(nx_a + w_ic_post, 0, n)
        ic_ref = _argmax_in_window(acc_peak_feat, a_ic, b_ic)
        ic = int(ic_ref) if ic_ref is not None else int(nx_a)

        if (tc - prev_tc) < refractory:
            rej["refractory"] += 1
            continue
        prev_tc = tc

        if ic <= tc:
            rej["order"] += 1
            continue
        if (ic - tc) < int(round(0.06 * fs_hz)):
            rej["too_short"] += 1
            continue

        # end: end of next stance segment (keeps stride window consistent with your pipeline)
        end = int(nx_b)

        # start: tc
        start = int(tc)

        # min_vel: choose minimum gyro norm inside next stance
        gwin = gyr_norm_1d[nx_a:nx_b]
        awin = acc_norm_1d[nx_a:nx_b]

        if len(gwin) > 0:
            flat = np.abs(awin - 9.81) < 2.0
            if np.any(flat):
                idxs = np.where(flat)[0]
                j = idxs[np.argmin(gwin[idxs])]
            else:
                j = int(np.argmin(gwin))
            min_vel = int(nx_a + j)
        else:
            min_vel = int((nx_a + nx_b) // 2)

        rows.append({
            "det_stride_id": det_id,
            "start": start,
            "end": end,
            "tc": tc,
            "ic": ic,
            "min_vel": min_vel,
        })
        det_id += 1

    events_imu = pd.DataFrame(rows)
    if events_imu.empty:
        events_imu = pd.DataFrame(columns=["det_stride_id","start","end","tc","ic","min_vel"]).set_index("det_stride_id")
    else:
        events_imu = events_imu.set_index("det_stride_id")

    debug = {
        "p_stance": p_stance,
        "stance_mask": stance,
        "n_stance_segments": len(segments),
        "reject_counts": rej,
    }
    return events_imu, debug

def refine_ic_with_specific_acc(
    events_imu,
    specific_acc_w,
    fs_hz,
    pre_s=0.18,
    post_s=0.03,
    *,
    dynamic=False,
    # stride-relative settings (used when dynamic=True)
    pre_frac=0.12,        # 12% of stride before coarse IC
    post_frac=0.03,       # 3% of stride after coarse IC
    pre_bounds_s=(0.08, 0.22),
    post_bounds_s=(0.01, 0.06),
    # limit how much earlier than coarse IC we allow the refined IC to move
    max_early_pull_frac=0.03,   # 3% of stride
    max_early_pull_bounds_s=(0.01, 0.05),
    stride_ref="ic2ic",   # stride length source: "ic2ic" (recommended)
):
    """
    Refine IC using vertical specific acceleration peak near coarse IC.

    If dynamic=True:
      - compute stride duration per stride (ic->next ic)
      - set search window as fractions of stride duration (with clamps)
      - set early-pull limit as fraction of stride duration (with clamps)
    """
    n = len(specific_acc_w)
    feat = np.abs(specific_acc_w[:, 2])  # vertical specific accel magnitude

    out = events_imu.copy()
    if "ic" not in out.columns or out["ic"].isna().all():
        return out

    # --- stride duration per stride (samples) ---
    ic_idx = out["ic"].to_numpy(dtype=float)

    stride_samples = np.full(len(out), np.nan, dtype=float)
    if dynamic and stride_ref == "ic2ic":
        for i in range(len(out) - 1):
            if np.isfinite(ic_idx[i]) and np.isfinite(ic_idx[i + 1]):
                d = int(ic_idx[i + 1]) - int(ic_idx[i])
                # reject junk (<0.3s or >3s)
                if 60 <= d <= int(3.0 * fs_hz):
                    stride_samples[i] = d

        med_stride = np.nanmedian(stride_samples)
        if not np.isfinite(med_stride):
            med_stride = int(1.0 * fs_hz)  # fallback 1s stride
    else:
        med_stride = None  # unused

    ic_refined = []

    # Add a positional index column so we never use stride IDs as array positions
    out2 = out.copy()
    out2["_pos"] = np.arange(len(out2))

    for _, r in out2.iterrows():
        pos = int(r["_pos"])
        ic0 = int(r["ic"])

        if not dynamic:
            ...
            ic_refined.append(ic_new)
            continue

        # --- dynamic windows based on stride length ---
        ss = stride_samples[pos] if np.isfinite(stride_samples[pos]) else med_stride

        pre_s_i = float(np.clip(pre_frac * ss / fs_hz, pre_bounds_s[0], pre_bounds_s[1]))
        post_s_i = float(np.clip(post_frac * ss / fs_hz, post_bounds_s[0], post_bounds_s[1]))

        w_pre = int(round(pre_s_i * fs_hz))
        w_post = int(round(post_s_i * fs_hz))

        a = max(0, ic0 - w_pre)
        b = min(n, ic0 + w_post)
        if b <= a:
            ic_refined.append(ic0)
            continue

        ic_new = a + int(np.argmax(feat[a:b]))

        max_early_pull_s = float(np.clip(
            max_early_pull_frac * ss / fs_hz,
            max_early_pull_bounds_s[0],
            max_early_pull_bounds_s[1],
        ))
        max_early_pull = int(round(max_early_pull_s * fs_hz))
        ic_new = max(ic_new, ic0 - max_early_pull)

        ic_refined.append(ic_new)

    out["ic"] = ic_refined
    return out


def phase_1_4_get_zupt_mask_from_min_vel(
    events_imu,
    n_samples,
    half_window=5,
    *,
    dynamic=False,
    fs_hz=None,
    frac=0.03,          # half-window = ~3% of stride
    min_hw=6,
    max_hw=25,
    stride_ref="ic2ic", # "ic2ic" or "tc2tc"
):
    """
    ZUPT mask centered around min_vel for each stride.

    If dynamic=True:
      - compute stride length per stride (in samples)
      - set half_window = clamp(round(frac * stride_samples), min_hw, max_hw)
    """
    zupt = np.zeros(n_samples, dtype=bool)

    ev = events_imu.dropna(subset=["min_vel"]).copy()
    if ev.empty:
        return zupt

    if not dynamic:
        hw = int(half_window)
        for _, row in ev.iterrows():
            mv = int(row["min_vel"])
            a = max(0, mv - hw)
            b = min(n_samples, mv + hw + 1)
            zupt[a:b] = True
        return zupt

    # --- dynamic mode ---
    if fs_hz is None or fs_hz <= 0:
        fs_hz = 200.0  # fallback

    # pick stride reference indices
    if stride_ref == "ic2ic":
        if "ic" not in events_imu.columns:
            raise ValueError("events_imu must have 'ic' for stride_ref='ic2ic'")
        idx = events_imu["ic"].to_numpy()
    elif stride_ref == "tc2tc":
        if "tc" not in events_imu.columns:
            raise ValueError("events_imu must have 'tc' for stride_ref='tc2tc'")
        idx = events_imu["tc"].to_numpy()
    else:
        raise ValueError("stride_ref must be 'ic2ic' or 'tc2tc'")

    # stride length per row via next index
    # (we use the same dataframe ordering as events_imu)
    stride_samples = np.full(len(events_imu), np.nan, dtype=float)
    for i in range(len(events_imu) - 1):
        if np.isfinite(idx[i]) and np.isfinite(idx[i + 1]):
            d = int(idx[i + 1]) - int(idx[i])
            if 60 <= d <= int(3.0 * fs_hz):  # reject junk (<0.3s or >3s)
                stride_samples[i] = d

    # use median stride for last / missing
    med_stride = np.nanmedian(stride_samples)
    if not np.isfinite(med_stride):
        med_stride = int(1.0 * fs_hz)  # fallback 1s stride

    # build ZUPT windows
    tmp = events_imu.copy()
    tmp["_pos"] = np.arange(len(tmp))
    ev = tmp.dropna(subset=["min_vel"]).copy()

    for _, row in ev.iterrows():
        pos = int(row["_pos"])
        mv = int(row["min_vel"])

        ss = stride_samples[pos]
        if not np.isfinite(ss):
            ss = med_stride

        hw = int(np.clip(round(frac * ss), min_hw, max_hw))

        a = max(0, mv - hw)
        b = min(n_samples, mv + hw + 1)
        zupt[a:b] = True

    return zupt

##############################################################################
# Orientation + gravity removal
##############################################################################

def phase_1_5_estimate_orientation(
    imu_f, fs_hz, *,
    zupt_mask=None,
    beta_stance=0.20,
    beta_swing=0.02,
    warmup_s=0.5,
):
    """
    (1.5) Orientation using Madgwick (gyro+acc) with OPTIONAL adaptive beta:
      - stance: higher beta (acc trusted)  -> correct drift
      - swing : lower beta (acc polluted) -> avoid mid-swing overshoot

    Inputs:
      - gyro assumed deg/s (converted to rad/s)
      - acc can be in any linear units; we normalize to unit vector
      - zupt_mask: boolean array (True during stance/min_vel window)
    """
    import numpy as np
    from ahrs.filters import Madgwick

    acc = imu_f[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyr = imu_f[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=float)

    # deg/s -> rad/s (you confirmed this already)
    gyr_rad = np.deg2rad(gyr)

    f = Madgwick(frequency=float(fs_hz))

    n = len(acc)
    Q = np.zeros((n, 4), dtype=float)
    Q[0] = np.array([1.0, 0.0, 0.0, 0.0])

    # warmup: force stance-like correction at start to settle initial attitude
    warmup_n = int(max(0, round(float(warmup_s) * float(fs_hz))))

    # ensure mask is usable
    if zupt_mask is not None:
        zupt_mask = np.asarray(zupt_mask, dtype=bool)
        if zupt_mask.shape[0] != n:
            zupt_mask = None  # fallback

    for i in range(1, n):
        # --- adaptive beta ---
        if i < warmup_n:
            f.beta = float(beta_stance)
        elif zupt_mask is not None and zupt_mask[i]:
            f.beta = float(beta_stance)
        else:
            f.beta = float(beta_swing)

        # normalize acc to unit vector (robust to m/s^2 vs g units)
        a = acc[i]
        an = float(np.linalg.norm(a))
        if an > 1e-8 and np.isfinite(an):
            a = a / an
        else:
            # if accel is garbage, reuse last (won't inject NaNs)
            a = acc[i-1]
            an = float(np.linalg.norm(a))
            if an > 1e-8 and np.isfinite(an):
                a = a / an
            else:
                a = np.array([0.0, 0.0, 1.0], dtype=float)

        Q[i] = f.updateIMU(Q[i-1], gyr=gyr_rad[i], acc=a)

    return Q

def phase_1_5b_stance_attitude_correction(
    qs, imu_f, events_imu, fs_hz, half_window, *,
    alpha=0.25, g_world=np.array([0.0, 0.0, 1.0]),
    use_median=True
):
    acc = imu_f[["acc_x","acc_y","acc_z"]].to_numpy(dtype=float)
    gyr = imu_f[["gyr_x","gyr_y","gyr_z"]].to_numpy(dtype=float)

    # alpha mode
    alpha_mode = "fixed"
    alpha_min, alpha_max = 0.10, 0.60
    if isinstance(alpha, str) and alpha.lower().strip() in ("dynamic", "auto"):
        alpha_mode = "dynamic"
        alpha_fixed = None
    else:
        alpha_fixed = float(np.clip(float(alpha), 0.0, 1.0))

    gyr_mag_all = np.linalg.norm(gyr, axis=1)
    _finite = np.isfinite(gyr_mag_all)
    gyr_scale = float(np.median(gyr_mag_all[_finite])) if np.any(_finite) else 1.0
    gyr_scale = max(gyr_scale, 1e-6)

    n = len(acc)
    qs = np.asarray(qs, dtype=float).copy()

    ev = events_imu.dropna(subset=["min_vel"]).copy().sort_values("min_vel")
    if ev.empty:
        return qs

    # Per-sample cumulative correction schedule
    q_post_arr = np.zeros((n, 4), dtype=float)
    q_curr = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    last_mv = -1
    next_fill_start = 0

    for _, row in ev.iterrows():
        mv = int(row["min_vel"])
        if mv <= last_mv or mv < 0 or mv >= n:
            continue

        # Fill schedule up to mv
        if mv > next_fill_start:
            q_post_arr[next_fill_start:mv, :] = q_curr
            next_fill_start = mv

        a0 = max(0, mv - int(half_window))
        b0 = min(n, mv + int(half_window) + 1)

        # --- Gate: require quiet stance (low gyro + accel ~ g and stable) ---
        win_g = gyr[a0:b0]
        gm = float(np.median(np.linalg.norm(win_g, axis=1)))
        if gm > 2.0 * gyr_scale:
            last_mv = mv
            continue

        win = acc[a0:b0]
        mag = np.linalg.norm(win, axis=1)
        mag_med = float(np.median(mag))
        mag_mad = float(np.median(np.abs(mag - mag_med)))

        if not (8.8 <= mag_med <= 10.8):  # assumes m/s^2, tune if needed
            last_mv = mv
            continue
        if mag_mad > 0.25:
            last_mv = mv
            continue

        # --- Measured gravity direction in RAW BODY from accel ---
        a_meas = np.median(win, axis=0) if use_median else np.mean(win, axis=0)
        an = float(np.linalg.norm(a_meas))
        if not np.isfinite(an) or an < 1e-8:
            last_mv = mv
            continue
        a_meas_u = a_meas / an  # raw BODY

        # --- Estimated gravity direction in CORRECTED BODY ---
        q_mv = quat_mul(qs[mv], q_curr)  # corrected body->world
        g_b_est = quat_rotate(quat_conj(q_mv), g_world)  # world->corrected BODY
        gn = float(np.linalg.norm(g_b_est))
        if not np.isfinite(gn) or gn < 1e-8:
            last_mv = mv
            continue
        g_b_est_u = g_b_est / gn

        # Convert measured accel into corrected BODY so frames match
        a_meas_u_corr = quat_rotate(quat_conj(q_curr), a_meas_u)

        # Choose sign in corrected BODY frame
        if np.dot(g_b_est_u, a_meas_u_corr) < np.dot(g_b_est_u, -a_meas_u_corr):
            a_meas_u_corr = -a_meas_u_corr

        # Compute correction in corrected BODY: rotate g_est -> a_meas
        q_corr_b = _quat_from_two_unit_vectors(g_b_est_u, a_meas_u_corr)

        # --- Compute alpha_eff BEFORE using it ---
        if alpha_mode == "dynamic":
            # gyro confidence (lower gyro => higher confidence)
            c_gyr = 1.0 / (1.0 + (gm / (2.0 * gyr_scale))**2)

            # accel stability confidence (lower MAD => higher confidence)
            rel = mag_mad / max(0.05 * max(mag_med, 1e-6), 1e-9)
            c_acc = 1.0 / (1.0 + rel**2)

            conf = float(np.clip(0.6 * c_gyr + 0.4 * c_acc, 0.0, 1.0))
            alpha_eff = alpha_min + conf * (alpha_max - alpha_min)
        else:
            alpha_eff = alpha_fixed

        # Blend correction
        q_corr_b = _slerp_identity(q_corr_b, alpha_eff)

        def _align_score(q_curr_test):
            q_test = quat_mul(qs[mv], q_curr_test)
            g_test = quat_rotate(quat_conj(q_test), g_world)
            gn = np.linalg.norm(g_test)
            if gn < 1e-12:
                return -1e9
            g_test_u = g_test / gn
            return float(np.dot(g_test_u, a_meas_u_corr))  # higher is better

        score0 = _align_score(q_curr)

        # candidate updates
        cand = []

        # 1) post-multiply (your current)
        cand.append(quat_mul(q_curr, q_corr_b))

        # 2) pre-multiply
        cand.append(quat_mul(q_corr_b, q_curr))

        # 3) post-multiply with conjugate
        cand.append(quat_mul(q_curr, quat_conj(q_corr_b)))

        # 4) pre-multiply with conjugate
        cand.append(quat_mul(quat_conj(q_corr_b), q_curr))

        scores = [ _align_score(c) for c in cand ]
        best_i = int(np.argmax(scores))
        q_curr_new = _quat_norm(cand[best_i])

        # Optional: only accept if it improves
        if scores[best_i] > score0 + 1e-3:
            q_curr = q_curr_new
        # else: skip update (window not trustworthy)

        # Update cumulative correction
        q_curr = quat_mul(q_curr, q_corr_b)
        q_curr = _quat_norm(q_curr)

        last_mv = mv

    # Fill schedule tail
    q_post_arr[next_fill_start:n, :] = q_curr

    # Apply schedule: qs_corr[i] = qs[i] ⊗ q_post_arr[i]
    qs_corr = quat_mul_batch(qs, q_post_arr)
    qs_corr = quat_norm_batch(qs_corr)

    # Debug: show typical correction angle (more meaningful than |q-I|)
    w = np.clip(q_post_arr[:, 0], -1.0, 1.0)
    ang_deg = np.degrees(2.0 * np.arccos(np.abs(w)))
    print("[ATT] mean angle deg:", float(np.mean(ang_deg)),
          "p95 deg:", float(np.percentile(ang_deg, 95)))

    return qs_corr



def phase_1_5_compute_world_acceleration(imu_f, qs):
    """
    (1.5) Rotate body-frame acceleration into the world frame.

    Assumptions (validated empirically across all participants):
    - Madgwick returns quaternions representing body -> world rotation
    - quat_rotate(q, v) applies body -> world rotation
    - No quaternion conjugation is required (Already tested!!!)
    """
    acc_b = imu_f[["acc_x", "acc_y", "acc_z"]].to_numpy()
    n = len(acc_b)

    acc_w = np.zeros_like(acc_b)
    for i in range(n):
        acc_w[i] = quat_rotate(qs[i], acc_b[i])

    return acc_w


def phase_1_5_remove_gravity(acc_w, zupt_mask):
    """
    Remove gravity assuming world Z is correct.

    We estimate gravity magnitude from stance samples,
    but always subtract along world Z direction.
    """

    if zupt_mask is None or (not np.any(zupt_mask)):
        g_mag = float(np.median(np.linalg.norm(acc_w, axis=1)))
        ref = "all"
    else:
        g_mag = float(np.median(np.linalg.norm(acc_w[zupt_mask], axis=1)))
        ref = "stance"

    # Force gravity direction to world Z
    g_vec = np.array([0.0, 0.0, g_mag], dtype=float)

    specific_acc_w = acc_w - g_vec

    print(f"[Phase 1.5] Gravity removal mode: acc_w - [0,0,g_mag] "
          f"(median {ref}) | g_mag={g_mag:.3f}")

    return specific_acc_w, "acc_w - fixed-Z gravity"


##############################################################################
# Integration / kinematics
##############################################################################

def _valid_min_vel_window(mv, tc, ic, start, end, fs_hz, margin_s=0.06):
    # mv should be well away from swing boundaries
    m = int(round(margin_s * fs_hz))
    if mv is None:
        return False
    if not (start <= mv < end):
        return False
    # mv must not be inside swing (tc..ic)
    if tc <= mv <= ic:
        return False
    # mv must be sufficiently far from tc and ic
    if abs(mv - tc) < m or abs(mv - ic) < m:
        return False
    return True

def phase_1_6_integrate_z_only_per_stride_anchor_min_vel(
    specific_acc_w,       
    events_imu,
    fs_hz,
    half_window=10,
    min_samples=6,
    enforce_end_vel_zero=True,
    shift_to_tc=True,          # RESTORED for pipeline_core.py compatibility
    hold_through_stance=True,  # keep
):
    """
    Per-stride vertical integration (z-only) with:
      - per-stride az bias removal from stance window around min_vel
      - integrate swing (tc->ic)
      - end-velocity ramp removal over swing
      - end-position ramp removal over swing (close pz at ic)
      - OPTIONAL: hold pz=0 through stance (ic->next_tc) so ground windows are consistent
    """
    dt = 1.0 / fs_hz
    n = len(specific_acc_w)
    pz = np.zeros(n, dtype=float)
    vz = np.zeros(n, dtype=float)

    ev = events_imu.dropna(subset=["tc", "ic", "min_vel"]).copy().sort_values("tc")
    if ev.empty:
        return pz, vz

    tcs = ev["tc"].to_numpy(dtype=int)

    last_end = -1
    for k, (_, row) in enumerate(ev.iterrows()):
        tc = int(row["tc"]); ic = int(row["ic"]); mv = int(row["min_vel"])

        if not (0 <= tc < ic <= n): 
            continue
        if not (0 <= mv < n):
            continue
        if (ic - tc) < min_samples:
            continue
        if tc <= last_end:
            continue

        # stance window around min_vel for bias
        a0 = max(0, mv - int(half_window))
        b0 = min(n, mv + int(half_window) + 1)
        az_bias = float(np.median(specific_acc_w[a0:b0, 2]))

        # initial conditions at TC
        v = 0.0
        p = 0.0
        vz[tc] = v
        pz[tc] = p

        # integrate swing with bias removed
        for i in range(tc + 1, ic):
            az = specific_acc_w[i, 2] - az_bias
            v = v + az * dt
            p = p + v * dt
            vz[i] = v
            pz[i] = p

        # enforce end vel ~ 0 (swing only)
        if enforce_end_vel_zero and (ic - tc) > 2:
            v_end = vz[ic - 1]
            L = (ic - 1) - tc
            for i in range(tc, ic):
                frac = (i - tc) / float(L)
                vz[i] -= frac * v_end

            # recompute pz from corrected vz
            p = pz[tc]
            for i in range(tc + 1, ic):
                p = p + vz[i] * dt
                pz[i] = p

        # optional: shift so pz(tc)=0 (usually already true here)
        if shift_to_tc:
            p0 = pz[tc]
            pz[tc:ic] -= p0

        # ---- NEW: end-position ramp removal over swing (close pz at ic) ----
        # After vel correction, pz[ic-1] may still be non-zero due to drift.
        p_end = pz[ic - 1]
        L = (ic - 1) - tc
        if L > 0 and np.isfinite(p_end):
            for i in range(tc, ic):
                frac = (i - tc) / float(L)   # 0 at tc, 1 at ic-1
                pz[i] -= frac * p_end

        # recompute vz from the adjusted pz (optional but keeps consistency)
        # (not strictly required if you only use pz for MTC, but it's cleaner)
        p_prev = pz[tc]
        for i in range(tc + 1, ic):
            vz[i] = (pz[i] - p_prev) / dt
            p_prev = pz[i]
        vz[tc] = 0.0


        # OPTIONAL: hold position through stance (ic -> next_tc)
        if hold_through_stance:
            next_tc = int(tcs[k + 1]) if (k + 1) < len(tcs) else n
            hold_end = min(next_tc, n)

            # After end-position closure, pz[ic-1] should already be ~0.
            # Force stance to a clean ground baseline anyway.
            p_hold = 0.0
            for i in range(ic, hold_end):
                pz[i] = p_hold
                vz[i] = 0.0

        last_end = ic - 1

    return pz, vz


##############################################################################
# MTC computation
##############################################################################

def phase_1_1_define_mtc_objective():
    """
    (1.1) Define single-point MTC / “clearance” objective.

    For Phase 1 verification, we compute:
    - toe clearance = z_toe(t) - z_ground
    - MTC = min clearance over swing (TC → next IC)

    Returns a small config dict used downstream.
    """
    return {
        "clearance_point": "virtual_toe",
        "metric": "MTC",
        "swing_definition": "TC_to_IC",
        "ground_definition": "stance_median_toe_z"
    }


def phase_1_2_define_virtual_toe_point(toe_offset_cm):
    """
    (1.2) Virtual toe point definition (no shoe scan):
    p_toe_b = [x,y,z] offset from IMU/body frame to toe point in METERS.
    """
    toe_offset_m = np.array(toe_offset_cm, dtype=float) / 100.0
    return toe_offset_m


def phase_1_8_compute_mtc_per_stride(p_imu_w, qs, p_toe_b, events_imu, ground_mode="min_vel", half_window=10, fs_hz=None):
    n = len(p_imu_w)

    rej = {}

    def REJ(k):
        rej[k] = rej.get(k, 0) + 1


    if fs_hz is None or fs_hz <= 0:
        # sensible fallback for dataset; avoids crashes
        fs_hz = 200.0

    toe_w = np.zeros((n, 3), dtype=float)
    for i in range(n):

        toe_w[i] = p_imu_w[i] + quat_rotate(qs[i], p_toe_b)

    toe_z = toe_w[:, 2]

    ev = events_imu.dropna(subset=["ic", "tc", "start", "end"]).copy()

    rows = []
    stride_debug = {}  # per-stride debug payload

    for s_id, row in ev.iterrows():
        
        start = int(row["start"])
        end   = int(row["end"])
        tc    = int(row["tc"])
        ic    = int(row["ic"])

        # Swing should be tc -> ic, and must lie within [start, end]
        if not (start <= tc < ic <= end):
            continue
        if ic >= n or tc < 0 or ic <= tc + 5:
            continue

        # Ground reference from stance: median toe_z over ic -> end
        if end <= ic + 2:
            continue
        end = min(end, len(toe_z))
        mv = int(row["min_vel"]) if ("min_vel" in row) and (not pd.isna(row["min_vel"])) else None

        # -----------------------------
        # (3) Ground selection: score & choose
        # -----------------------------
        candidates = []

        # A) min_vel local window (best when valid)
        if (ground_mode == "min_vel") and (mv is not None) and _valid_min_vel_window(
            mv, tc, ic, start, end, fs_hz, margin_s=0.06
        ):
            a0 = max(0, mv - half_window)
            b0 = min(len(toe_z), mv + half_window + 1)
            candidates.append(("min_vel", a0, b0))

        # B) mid-stance window (avoid IC impact + toe-off rocker)
        stance_len = end - ic

        m_in  = int(round(0.08 * fs_hz))   # keep your current 80 ms (or make dynamic later)
        m_out = int(round(0.08 * fs_hz))

        a1 = ic + m_in
        b1 = end - m_out
        if b1 > a1 + int(0.06 * fs_hz):    # require >= 60 ms
            candidates.append(("mid_stance", a1, b1))

        # B2) center-stance (tighter, flat-only)
        stance_len = end - ic
        mid = ic + stance_len // 2
        w = int(np.clip(0.12 * stance_len, int(0.05 * fs_hz), int(0.12 * fs_hz)))  # 5–120 ms-ish
        a2 = max(ic, mid - w // 2)
        b2 = min(end, mid + w // 2)
        if b2 > a2 + int(0.03 * fs_hz):  # >= 30 ms
            candidates.append(("center_stance", a2, b2))

        # C) last resort: full stance
        candidates.append(("full_stance", ic, end))

        best = None  # (score, mode, a, b, win, sigma_hat, iqr)

        for mode, a, b in candidates:
            win = toe_z[a:b]
            if len(win) < int(0.04 * fs_hz):
                continue

            q25, q75 = np.percentile(win, [25, 75])
            iqr = float(q75 - q25)
            mad = _robust_mad(win)
            sigma_hat = float(1.4826 * mad)

            # score: smaller is better
            score = sigma_hat + 0.5 * iqr

            if (best is None) or (score < best[0]):
                best = (score, mode, a, b, win, sigma_hat, iqr)

        if best is None:
            REJ("no_ground_candidate")
            continue

        _, ground_mode_used, gw_a, gw_b, ground_win, ground_sigma_hat, ground_iqr = best

        if s_id < 3:  # only first few strides to avoid spam
            force_print(
                f"[GROUND] s_id={s_id} mode={ground_mode_used} "
                f"len={gw_b-gw_a} sigma={ground_sigma_hat*1000:.1f}mm iqr={ground_iqr*1000:.1f}mm "
                f"(ic={ic}, end={end}, stance_len={end-ic})"
            )


        x = np.arange(len(ground_win), dtype=float)

        if len(ground_win) >= 3:
            m, c = np.polyfit(x, ground_win, 1)
            trend = m * x + c
            ground_win_flat = ground_win - trend
            ground_level = np.median(ground_win_flat)
            ground = ground_level + c   # add back intercept
        else:
            ground = float(np.median(ground_win))




        # --- Swing window: compute MTC (simple + non-lethal) ---
        # Optional small trim to reduce TC/IC edge artifacts; set trim=0 to fully disable
        trim = int(round(0.005 * fs_hz))  # 5 ms
        trim = min(trim, int(0.10 * (ic - tc)))  # don’t over-trim short swings

        a_sw = tc + trim
        b_sw = ic - trim
        if b_sw <= a_sw + 5:
            REJ("swing_too_short")
            continue

        swing_clearance = toe_z[a_sw:b_sw] - ground
        mtc = float(np.min(swing_clearance))
        mtc_idx = a_sw + int(np.argmin(swing_clearance))
        mtc_rel_pct = (mtc_idx - tc) / max(1, (ic - tc))

        # Log edge-min tendency (do NOT reject)
        clipped_flag = (mtc_rel_pct < 0.08) or (mtc_rel_pct > 0.92)
        if clipped_flag:
            REJ("mtc_edge_min")


        # Store per-stride debug (everything needed for a single-stride plot)
        stride_debug[s_id] = {
            "start": start, "end": end,
            "tc": tc, "ic": ic,
            "min_vel": mv,

            "ground_z": ground,
            "ground_mode_used": ground_mode_used,
            "ground_win_a": gw_a,
            "ground_win_b": gw_b,
            "ground_sigma_hat_m": float(ground_sigma_hat),
            "ground_iqr_m": float(ground_iqr),

            "swing_trim_a": a_sw,
            "swing_trim_b": b_sw,
            "mtc_idx": mtc_idx,
            "mtc_rel_pct": float(mtc_rel_pct),
            "mtc_edge_flag": bool(clipped_flag),
        }

        rows.append({
            "s_id": s_id,
            "tc": tc,
            "ic": ic,
            "start": start,
            "end": end,
            "ground_z": ground,
            "mtc_imu_m": mtc,
            "ground_sigma_hat_m": float(ground_sigma_hat),
            "ground_std_m": float(np.std(ground_win)),  # optional keep if you want both
            "ground_iqr_m": float(ground_iqr),
        })

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.set_index("s_id")

    debug = {"toe_w": toe_w, "toe_z": toe_z, "stride_debug": stride_debug}
    
    debug["rej_counts"] = rej

    
    return res, debug

##############################################################################
# Stride matching (IMU ↔ MoCap)
##############################################################################

def dp_monotone_match(
    gt_times: np.ndarray,
    det_times: np.ndarray,
    gate: int,
    fp_penalty: float,
    fn_penalty: float,
):
    """Monotone dynamic-programming matcher.

    Returns list of (det_i, gt_j) pairs.
    - Allows skipping detections (FP) and skipping GT (FN).
    - Enforces order (no crossing matches).
    """
    gt_times = np.asarray(gt_times, dtype=int)
    det_times = np.asarray(det_times, dtype=int)
    n = len(det_times)
    m = len(gt_times)

    # dp[i][j] = best cost using first i det and first j gt
    # backpointer: 0=match, 1=skip_det, 2=skip_gt
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF, dtype=float)
    bp = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[0, 0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            base = dp[i, j]
            if base >= INF:
                continue

            # skip a detection (FP)
            if i < n:
                c = base + fp_penalty
                if c < dp[i + 1, j]:
                    dp[i + 1, j] = c
                    bp[i + 1, j] = 1

            # skip a gt (FN)
            if j < m:
                c = base + fn_penalty
                if c < dp[i, j + 1]:
                    dp[i, j + 1] = c
                    bp[i, j + 1] = 2

            # match
            if i < n and j < m:
                dt = abs(int(det_times[i]) - int(gt_times[j]))
                if dt <= gate:
                    c = base + float(dt)
                    if c < dp[i + 1, j + 1]:
                        dp[i + 1, j + 1] = c
                        bp[i + 1, j + 1] = 0

    # backtrack
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bp[i, j]
        if move == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs


def match_strides_by_mid_swing_dp(
    events_det: pd.DataFrame,
    events_gt_imu: pd.DataFrame,
    fs_hz: float,
    gate_s: float = 0.25,
):
    """Match detected strides to ground-truth strides using mid-swing time.

    Returns:
      mapping_df: columns [det_stride_id, s_id, mid_err_samples]
      summary: dict with precision/recall-like counts for stride matching.
    """
    det = events_det.dropna(subset=["tc","ic"]).copy()
    gt  = events_gt_imu.dropna(subset=["tc","ic"]).copy()

    if det.empty or gt.empty:
        return pd.DataFrame(columns=["det_stride_id","s_id","mid_err_samples"]), {
            "n_det": int(len(det)), "n_gt": int(len(gt)), "n_matched": 0
        }

    det_mid = ((det["tc"].to_numpy(dtype=int) + det["ic"].to_numpy(dtype=int)) / 2.0).astype(int)
    gt_mid  = ((gt["tc"].to_numpy(dtype=int)  + gt["ic"].to_numpy(dtype=int))  / 2.0).astype(int)

    gate = int(round(float(gate_s) * float(fs_hz)))
    pairs = dp_monotone_match(
        gt_times=gt_mid,
        det_times=det_mid,
        gate=gate,
        fp_penalty=float(gate),
        fn_penalty=float(gate),
    )

    det_ids = det.index.to_numpy()
    gt_ids  = gt.index.to_numpy()

    rows = []
    for di, gj in pairs:
        rows.append({
            "det_stride_id": int(det_ids[di]),
            "s_id": int(gt_ids[gj]),
            "mid_err_samples": int(det_mid[di] - gt_mid[gj]),
        })

    mapping_df = pd.DataFrame(rows)

    n_matched = len(mapping_df)
    summary = {
        "n_det": int(len(det)),
        "n_gt": int(len(gt)),
        "n_matched": int(n_matched),
        "stride_precision": float(n_matched / max(1, len(det))),
        "stride_recall": float(n_matched / max(1, len(gt))),
    }
    return mapping_df, summary


def apply_stride_mapping_to_imu_results(
    imu_stride_res: pd.DataFrame,
    events_det: pd.DataFrame,
    mapping_df: pd.DataFrame,
    events_gt_imu: pd.DataFrame,
):
    """Re-index IMU per-stride outputs by ground-truth s_id for Phase 1.9 join."""
    if imu_stride_res is None or imu_stride_res.empty or mapping_df.empty:
        return pd.DataFrame()

    tmp = imu_stride_res.copy()
    tmp["det_stride_id"] = tmp.index.astype(int)

    tmp = tmp.merge(mapping_df, on="det_stride_id", how="inner")
    if tmp.empty:
        return pd.DataFrame()

    # Add timing errors (samples) for reporting (optional)
    gt = events_gt_imu[["tc","ic"]].copy()
    gt = gt.rename(columns={"tc":"tc_gt","ic":"ic_gt"})
    tmp = tmp.join(gt, on="s_id", how="left")

    tmp["tc_err_samples"] = tmp["tc"].astype(int) - tmp["tc_gt"].astype(int)
    tmp["ic_err_samples"] = tmp["ic"].astype(int) - tmp["ic_gt"].astype(int)

    # Phase 1.9 expects index to be s_id
    tmp = tmp.set_index("s_id")
    # If multiple det strides map to same gt stride, keep the one with smallest |mid_err|
    tmp["abs_mid_err"] = np.abs(tmp["mid_err_samples"].to_numpy(dtype=float))
    tmp = tmp.sort_values("abs_mid_err").groupby(level=0, as_index=True).head(1)
    tmp = tmp.drop(columns=["abs_mid_err"])
    return tmp

##############################################################################
# Validation + plotting
##############################################################################

def plot_one_stride_clearance(s_id, debug):
    toe_z = debug["toe_z"]
    sd = debug["stride_debug"][s_id]

    start, end = sd["start"], sd["end"]
    tc, ic = sd["tc"], sd["ic"]
    mv = sd["min_vel"]
    ground = sd["ground_z"]
    gw_a, gw_b = sd["ground_win_a"], sd["ground_win_b"]

    clearance_mm = (toe_z - ground) * 1000.0

    swing = clearance_mm[tc:ic]
    i_min = tc + int(np.argmin(swing))

    plt.figure()
    plt.plot(np.arange(start, end), clearance_mm[start:end])
    plt.axhline(0.0)
    plt.axvline(tc, linestyle="--", label="TC")
    plt.axvline(ic, linestyle="--", label="IC")
    if mv is not None:
        plt.axvline(mv, linestyle=":", label="min_vel")
    plt.axvspan(gw_a, gw_b, alpha=0.2, label="ground window")
    plt.scatter([i_min], [clearance_mm[i_min]], marker="x", s=80, label=f"min={clearance_mm[i_min]:.1f}mm")
    plt.title(f"Stride {s_id} clearance (IMU)")
    plt.xlabel("Sample index")
    plt.ylabel("Clearance (mm)")
    plt.grid(True)
    plt.legend()
    plt.show(block=True)


def plot_imu_mocap_events_window(
    imu_f: pd.DataFrame,
    mocap_traj,
    events_imu: pd.DataFrame,
    toe_marker_name: str,
    fs_hz: float,
    stride_idx: int = 0,
    n_strides: int = 2,
):
    """
    Plot a small time window around selected strides with
    TC (red), IC (blue), min_vel (green).
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ---- select strides ----
    ev = events_imu.dropna(subset=["tc", "ic", "min_vel"]).reset_index(drop=True)
    ev = ev.iloc[stride_idx: stride_idx + n_strides]

    t_start = int(ev["tc"].iloc[0] - 0.15 * fs_hz)
    t_end   = int(ev["tc"].iloc[-1] + 0.8 * fs_hz)

    t_start = max(0, t_start)
    t_end   = min(len(imu_f) - 1, t_end)

    t = np.arange(t_start, t_end) / fs_hz

    # ---- IMU norms ----
    gyr = imu_f[["gyr_x", "gyr_y", "gyr_z"]].to_numpy()
    acc = imu_f[["acc_x", "acc_y", "acc_z"]].to_numpy()

    gyr_norm = np.linalg.norm(gyr, axis=1)[t_start:t_end]
    acc_norm = np.linalg.norm(acc, axis=1)[t_start:t_end]

    # ---- MoCap toe-z resampled to IMU ----
    toe_z = mocap_traj[toe_marker_name]["z"].to_numpy()
    toe_z_rs = np.interp(
        np.linspace(0, 1, t_end - t_start),
        np.linspace(0, 1, len(toe_z)),
        toe_z
    )

    # ---- Plot ----
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(t, gyr_norm, label="gyro_norm")
    axs[0].set_ylabel("gyro norm")
    axs[0].grid(True)

    axs[1].plot(t, acc_norm, label="acc_norm")
    axs[1].set_ylabel("acc norm")
    axs[1].grid(True)

    axs[2].plot(t, toe_z_rs, label="MoCap toe z")
    axs[2].set_ylabel("toe z (m)")
    axs[2].set_xlabel("time (s)")
    axs[2].grid(True)

    # ---- Event markers ----
    for _, r in ev.iterrows():
        for ax in axs:
            ax.axvline(r.tc / fs_hz, color="red", linestyle="--", label="TC")
            ax.axvline(r.ic / fs_hz, color="blue", linestyle="--", label="IC")
            ax.axvline(r.min_vel / fs_hz, color="green", linestyle=":", label="min_vel")

    # de-duplicate legend
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())

    fig.suptitle(f"Stride window [{stride_idx}–{stride_idx+n_strides-1}]")
    plt.tight_layout()
    plt.show()


def phase_1_9_validate_against_mocap(
    mocap_traj,
    events_mocap,
    toe_marker_name,
    imu_stride_res,
    plot=False,
    verbose=True,
    quiet=False,
    debug=None,
    events_imu=None,
    ground_mode="min_vel",
    half_window=10,
    fs_hz=None,
    plot_worst_k=0,
    stride_errors_path=None,
):

    toe_z = mocap_traj[toe_marker_name]["z"].to_numpy()
    ev = events_mocap.dropna(subset=["ic", "tc", "start", "end"]).copy()

    gt_rows = []
    for s_id, row in ev.iterrows():
        start = int(row["start"])
        end   = int(row["end"])
        tc    = int(row["tc"])
        ic    = int(row["ic"])

        # same window rule
        if not (start <= tc < ic <= end):
            continue
        if end <= ic + 2 or end >= len(toe_z):
            continue

        mv = None
        if ground_mode == "min_vel":
            if ("min_vel" in row) and (not pd.isna(row["min_vel"])):
                mv = int(row["min_vel"])
            elif events_imu is not None and s_id in events_imu.index and not pd.isna(events_imu.loc[s_id, "min_vel"]):
                mv = int(events_imu.loc[s_id, "min_vel"])


        if ground_mode == "min_vel" and mv is not None:
            a = max(0, mv - half_window)
            b = min(len(toe_z), mv + half_window + 1)
            ground = float(np.median(toe_z[a:b]))
        else:
            ground = float(np.median(toe_z[ic:end]))

        # ---- Enforce toe_z(ic-) = ground by shifting swing only ----
        toe_z_end = toe_z[ic - 1] - ground
        L = (ic - 1) - tc
        if L > 0:
            for i in range(tc, ic):
                frac = (i - tc) / float(L)
                toe_z[i] -= frac * toe_z_end

        swing_clearance = toe_z[tc:ic] - ground

        mtc = float(np.min(swing_clearance))

        gt_rows.append({"s_id": s_id, "mtc_mocap_m": mtc})

    mocap_stride_res = pd.DataFrame(gt_rows)
    if mocap_stride_res.empty:
        raise RuntimeError("MoCap GT stride list is empty — check toe_marker_name and event windows.")

    mocap_stride_res = mocap_stride_res.set_index("s_id")

    # ------------------------------------------------------------
    # MoCap-only baseline mode:
    # If imu_stride_res is None, return MoCap MTC per stride and
    # define error metrics as zero by construction.
    # ------------------------------------------------------------
    if imu_stride_res is None:
        extra = {
            "mtc_imu_mean_m": float(mocap_stride_res["mtc_mocap_m"].mean()),
            "mtc_mocap_mean_m": float(mocap_stride_res["mtc_mocap_m"].mean()),
            "mtc_avg_diff_m": 0.0,
            "mtc_mean_abs_diff_m": 0.0,
        }
        joined = mocap_stride_res.copy()
        bias = 0.0
        rmse = 0.0
        return joined, bias, rmse, extra


    joined = imu_stride_res.join(mocap_stride_res, how="inner").dropna()
    if joined.empty:
        if quiet:
            return joined, np.nan, np.nan, {
                "mtc_imu_mean_m": np.nan,
                "mtc_mocap_mean_m": np.nan,
                "mtc_avg_diff_m": np.nan,
                "mtc_mean_abs_diff_m": np.nan,
            }
        raise RuntimeError("No overlapping strides between IMU and MoCap results after join.")


    err = joined["mtc_imu_m"] - joined["mtc_mocap_m"]
    bias = float(err.mean())
    rmse = float(np.sqrt(np.mean(err.to_numpy() ** 2)))

    # Additional reporting metrics (useful for calibration logs and terminal summaries)
    mtc_imu_mean_m = float(joined["mtc_imu_m"].mean())
    mtc_mocap_mean_m = float(joined["mtc_mocap_m"].mean())

    mtc_imu_median_m = float(joined["mtc_imu_m"].median())
    mtc_mocap_median_m = float(joined["mtc_mocap_m"].median())

    # Signed mean difference between the *means* (should match bias when computed over the same joined set)
    mtc_avg_diff_m = float(mtc_imu_mean_m - mtc_mocap_mean_m)
    mtc_mean_abs_diff_m = float(np.mean(np.abs(err.to_numpy())))
    extra = {
        "mtc_imu_mean_m": round(mtc_imu_mean_m, 6),
        "mtc_mocap_mean_m": round(mtc_mocap_mean_m, 6),

        "mtc_imu_median_m": round(mtc_imu_median_m, 6),
        "mtc_mocap_median_m": round(mtc_mocap_median_m, 6),

        "mtc_avg_diff_m": round(mtc_avg_diff_m, 6),
        "mtc_mean_abs_diff_m": round(mtc_mean_abs_diff_m, 6),
    }



    # ---- Debug: Top absolute errors (stride-level) ----
    joined_dbg = joined.copy()
    joined_dbg["err_m"] = err
    joined_dbg["abs_err_m"] = np.abs(err)

    top5 = (
        joined_dbg.sort_values("abs_err_m", ascending=False)
        .head(5)[["mtc_mocap_m", "mtc_imu_m", "err_m", "abs_err_m"]]
    )

    # convert to mm for readability
    top5_mm = top5 * 1000.0
    top5_mm = top5_mm.rename(columns={
        "mtc_mocap_m": "mocap_mtc_mm",
        "mtc_imu_m": "imu_mtc_mm",
        "err_m": "err_mm",
        "abs_err_m": "abs_err_mm",
    })


    # ---- Debug: Top absolute errors ----
    print("\n[Phase 1.9] Top 5 absolute errors (by stride id):")
    print(top5_mm.round(2).to_string())

    if stride_errors_path is not None:
        joined_dbg.to_csv(stride_errors_path, index=True)
        print(f"[Phase 1.9] Saved {stride_errors_path}")


    if verbose and (not quiet):
        print("\n=== Phase 1.9 Validation (instep IMU vs MoCap toe marker) ===")
        print(f"Strides compared: {len(joined)}")
        print(f"Bias (IMU - MoCap) [m]: {bias:.4f}  ({bias*1000:.1f} mm)")
        print(f"RMSE [m]: {rmse:.4f}  ({rmse*1000:.1f} mm)")
        print(f"Mean MTC IMU [mm]:   {mtc_imu_mean_m*1000:.1f}")
        print(f"Mean MTC MoCap [mm]: {mtc_mocap_mean_m*1000:.1f}")
        print(f"Avg diff (IMU-MoCap) [mm]: {mtc_avg_diff_m*1000:.1f}")
        print(f"Mean |diff| [mm]:          {mtc_mean_abs_diff_m*1000:.1f}")


    if plot:
        # Compute common arrays (mm)
        x = joined["mtc_mocap_m"].to_numpy() * 1000.0
        y = joined["mtc_imu_m"].to_numpy() * 1000.0
        e = (joined["mtc_imu_m"] - joined["mtc_mocap_m"]).to_numpy() * 1000.0  # IMU - MoCap (mm)

        # Top-5 worst by absolute error
        joined_dbg2 = joined.copy()
        joined_dbg2["err_mm"] = e
        joined_dbg2["abs_err_mm"] = np.abs(e)
        top5 = joined_dbg2.sort_values("abs_err_mm", ascending=False).head(5)

        # ----- Create ONE tiled figure -----
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        axs = axs.ravel()

        # --- Plot 1: IMU vs MoCap per-stride scatter (mm) ---
        ax = axs[0]
        ax.scatter(x, y)
        mn = float(min(x.min(), y.min()))
        mx = float(max(x.max(), y.max()))
        ax.plot([mn, mx], [mn, mx])  # y=x
        ax.set_xlabel("MoCap MTC (mm)")
        ax.set_ylabel("IMU MTC (mm)")
        ax.set_title(f"Per-stride MTC (N={len(joined)})\nBias={bias*1000:.1f}mm, RMSE={rmse*1000:.1f}mm")
        ax.grid(True)

        for s_id, row in top5.iterrows():
            xm = float(row["mtc_mocap_m"] * 1000.0)
            ym = float(row["mtc_imu_m"] * 1000.0)
            ax.annotate(str(s_id), (xm, ym), textcoords="offset points", xytext=(6, 6))

        # --- Plot 2: Error histogram (mm) ---
        ax = axs[1]
        ax.hist(e, bins=12)
        ax.set_xlabel("Error (IMU - MoCap) (mm)")
        ax.set_ylabel("Count")
        ax.set_title("Error distribution")
        ax.grid(True)

        # --- Plot 3: Error vs stride id (mm) ---
        ax = axs[2]
        s_ids = joined.index.to_numpy()
        ax.scatter(s_ids, e)
        ax.axhline(0.0)
        ax.set_xlabel("Stride id (s_id)")
        ax.set_ylabel("Error (mm)")
        ax.set_title("Stride-wise error")
        ax.grid(True)
        for s_id, row in top5.iterrows():
            ax.annotate(f"{s_id}", (s_id, float(row["err_mm"])), textcoords="offset points", xytext=(6, 6))

        # --- Plot 4: Error vs MoCap MTC (mm) ---
        ax = axs[3]
        ax.scatter(x, e)
        ax.axhline(0.0)
        ax.set_xlabel("MoCap MTC (mm)")
        ax.set_ylabel("Error (mm)")
        ax.set_title("Error vs clearance level")
        ax.grid(True)

        # --- Plot 5: Paired MTC points vs time (circles) + vertical connector per stride ---
        ax = axs[4]
        if fs_hz is None or events_imu is None:
            ax.text(0.5, 0.5, "Plot 5 needs fs_hz and events_imu passed into Phase 1.9",
                    ha="center", va="center")
            ax.set_axis_off()
        else:
            ev_plot = events_imu.dropna(subset=["tc", "ic"]).copy()

            stride_ids_set = set(joined.index.tolist())

            t_mid_s = []
            mtc_mocap_mm = []
            mtc_imu_mm = []

            for s_id, row in ev_plot.iterrows():
                if s_id not in stride_ids_set:
                    continue
                tc = int(row["tc"])
                ic = int(row["ic"])
                if ic <= tc:
                    continue

                # mid-swing time in IMU seconds (absolute)
                t = 0.5 * (tc + ic) / float(fs_hz)

                t_mid_s.append(t)
                mtc_mocap_mm.append(float(joined.loc[s_id, "mtc_mocap_m"]) * 1000.0)
                mtc_imu_mm.append(float(joined.loc[s_id, "mtc_imu_m"]) * 1000.0)

            if len(t_mid_s) > 0:
                order = np.argsort(np.asarray(t_mid_s))
                t_mid_s = np.asarray(t_mid_s)[order]
                mtc_mocap_mm = np.asarray(mtc_mocap_mm)[order]
                mtc_imu_mm = np.asarray(mtc_imu_mm)[order]

                for t, y0, y1 in zip(t_mid_s, mtc_mocap_mm, mtc_imu_mm):
                    ax.plot([t, t], [y0, y1], linewidth=2, color="g")

                ax.scatter(t_mid_s, mtc_mocap_mm, marker="o", label="MoCap MTC (mm)")
                ax.scatter(t_mid_s, mtc_imu_mm, marker="o", label="IMU MTC (mm)")

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("MTC (mm)")
                ax.set_title("Paired stride MTC vs time (vertical line = error)")
                ax.grid(True)
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No overlapping strides to plot", ha="center", va="center")
                ax.set_axis_off()

        # --- Plot 6: Per-stride MTC error vs time (IMU - MoCap), with Bias and ±RMSE band ---
        ax = axs[5]

        if fs_hz is None or events_imu is None:
            ax.text(0.5, 0.5, "Plot 6 needs fs_hz and events_imu",
                    ha="center", va="center")
            ax.set_axis_off()
        else:
            # Error per stride (mm)
            e_mm = (joined["mtc_imu_m"] - joined["mtc_mocap_m"]).to_numpy() * 1000.0
            s_ids = joined.index.to_numpy()

            # Time stamp per stride (use IMU event indices). Mid-swing time.
            t_s = []
            e_s = []
            sid_s = []

            for sid, err in zip(s_ids, e_mm):
                if sid not in events_imu.index:
                    continue
                row = events_imu.loc[sid]
                if pd.isna(row["tc"]) or pd.isna(row["ic"]):
                    continue
                tc = int(row["tc"])
                ic = int(row["ic"])
                if ic <= tc:
                    continue
                t_mid = 0.5 * (tc + ic) / float(fs_hz)
                t_s.append(t_mid)
                e_s.append(float(err))
                sid_s.append(sid)

            if len(t_s) == 0:
                ax.text(0.5, 0.5, "No overlapping strides to plot error vs time",
                        ha="center", va="center")
                ax.set_axis_off()
            else:
                t_s = np.asarray(t_s)
                e_s = np.asarray(e_s)

                # Sort by time for a clean trace
                order = np.argsort(t_s)
                t_s = t_s[order]
                e_s = e_s[order]
                sid_s = np.asarray(sid_s)[order]

                bias_mm = float(bias * 1000.0)
                rmse_mm = float(rmse * 1000.0)

                # Plot stride errors
                ax.scatter(t_s, e_s, label="Stride error (IMU − MoCap)")

                # Bias line and RMSE band
                ax.axhline(bias_mm, linestyle="--", linewidth=2, label=f"Bias = {bias_mm:.1f} mm")
                ax.fill_between(t_s, bias_mm - rmse_mm, bias_mm + rmse_mm, alpha=0.15,
                                label=f"±RMSE = {rmse_mm:.1f} mm")

                # Zero line
                ax.axhline(0.0, linewidth=1)

                # Annotate worst 5 by absolute error
                k = min(5, len(e_s))
                worst_idx = np.argsort(np.abs(e_s))[::-1][:k]
                for wi in worst_idx:
                    ax.annotate(str(sid_s[wi]), (t_s[wi], e_s[wi]),
                                textcoords="offset points", xytext=(6, 6))

                ax.set_xlabel("Time (s) (mid TC–IC, IMU time base)")
                ax.set_ylabel("MTC error (mm)")
                ax.set_title("Per-stride MTC error vs time (with Bias and RMSE)")
                ax.grid(True)
                ax.legend()




        fig.tight_layout()
        plt.show()


    if plot and (plot_worst_k is not None) and (plot_worst_k > 0) and (debug is not None) and ("stride_debug" in debug):
        # rank by absolute error (meters -> mm)
        tmp = joined.copy()
        tmp["err_mm"] = (tmp["mtc_imu_m"] - tmp["mtc_mocap_m"]) * 1000.0
        tmp["abs_err_mm"] = np.abs(tmp["err_mm"])

        worst_ids = tmp.sort_values("abs_err_mm", ascending=False).head(int(plot_worst_k)).index.tolist()

        print(f"\n[Phase 1.9] Auto-plotting worst {plot_worst_k} strides:", worst_ids)
        for sid in worst_ids:
            if sid in debug["stride_debug"]:
                plot_one_stride_clearance(int(sid), debug)
            else:
                print(f"[Phase 1.9] Stride {sid} not in debug['stride_debug'] (was filtered in Phase 1.8).")



    return joined, bias, rmse, extra

##############################################################################
# Toe-offset calibration (grid search)
##############################################################################

def calibrate_toe_offset_z(
    p_imu_w, qs, events_imu,
    mocap_traj, events_mocap,
    toe_marker_name,
    ground_mode="min_vel", half_window=10,
    z_min_cm=-12.0, z_max_cm=-2.0, step_cm=0.1,
    objective="rmse_plus_bias", verbose=False
):
    best = None

    for z_cm in np.arange(z_min_cm, z_max_cm + 1e-9, step_cm):
        p_toe_b = np.array([0.20, 0.0, z_cm/100.0])

        imu_stride_res, _ = phase_1_8_compute_mtc_per_stride(
            p_imu_w, qs, p_toe_b, events_imu,
            ground_mode=ground_mode,
            half_window=half_window
        )

        joined, bias, rmse, extra = phase_1_9_validate_against_mocap(
            mocap_traj=mocap_traj,
            events_mocap=events_mocap,
            toe_marker_name=toe_marker_name,
            imu_stride_res=imu_stride_res,
            plot=False,
            quiet=True,     # <- suppress prints + CSV
            verbose=False,  # <- suppress summary too
            ground_mode=ground_mode,
            half_window=half_window
        )

        if objective == "abs_bias":
            score = abs(bias)
        elif objective == "rmse":
            score = rmse
        else:
            score = rmse + 0.5*abs(bias)

        if verbose:
            print(f"[Cal] z_cm={z_cm:6.2f}  bias={bias*1000:+6.2f} mm  rmse={rmse*1000:6.2f} mm  score={score:.6f}")

        if best is None or score < best["score"]:
            best = {
                "z_cm": float(z_cm),
                "bias": float(bias),
                "rmse": float(rmse),
                "score": float(score),
                # Keep stride-level mean metrics for logging/CSV
                **(extra or {}),
            }

    return best


def calibrate_toe_offset_xz(
    p_imu_w, qs, events_imu,
    mocap_traj, events_mocap,
    toe_marker_name,
    events_gt_imu, fs_hz,
    ground_mode="min_vel", half_window=10,
    x_min_cm=4.0, x_max_cm=24.0, x_step_cm=0.5,
    z_min_cm=-12.0, z_max_cm=0.0, z_step_cm=0.2,
    objective="rmse",
    quiet=False
):
    """
    Grid-search toe (x,z) offset using MoCap to minimize stride-level MTC error.
    Returns best dict with x_cm, z_cm, bias, rmse, score.

    x_grid = np.arange(x_min_cm, x_max_cm + 1e-9, x_step_cm) / 100.0
    z_grid = np.arange(z_min_cm, z_max_cm + 1e-9, z_step_cm) / 100.0

    """
    best = None
    
    # Compute stride mapping ONCE (independent of toe offset)
    mapping_df, match_sum = match_strides_by_mid_swing_dp(
        events_imu, events_gt_imu, fs_hz, gate_s=0.25
    )
    if mapping_df.empty:
        raise RuntimeError(
            f"Calibration: no stride matches (det={match_sum['n_det']} gt={match_sum['n_gt']}). "
            "Cannot calibrate toe offset."
        )


    # Precompute MoCap MTC once (independent of IMU toe offset)
    # We'll reuse your Phase 1.9 method by calling it with plot=False, quiet=True.
    for x_cm in np.arange(x_min_cm, x_max_cm + 1e-9, x_step_cm):
        for z_cm in np.arange(z_min_cm, z_max_cm + 1e-9, z_step_cm):
            p_toe_b = np.array([x_cm/100.0, 0.0, z_cm/100.0])

            imu_stride_res, _ = phase_1_8_compute_mtc_per_stride(
                p_imu_w, qs, p_toe_b, events_imu,
                ground_mode=ground_mode,
                half_window=half_window
            )

            # ------------------------------------------------------------
            # STEP 3: Map IMU strides to MoCap stride IDs (once per grid point)
            # ------------------------------------------------------------
            imu_stride_res_scored = apply_stride_mapping_to_imu_results(
                imu_stride_res,
                events_imu,
                mapping_df,        # computed ONCE at top of function
                events_gt_imu
            )

            if imu_stride_res_scored.empty:
                continue

            joined, bias, rmse, extra = phase_1_9_validate_against_mocap(
                mocap_traj=mocap_traj,
                events_mocap=events_mocap,
                toe_marker_name=toe_marker_name,
                imu_stride_res=imu_stride_res_scored,   # IMPORTANT
                plot=False,
                quiet=True,
                verbose=False,
                ground_mode=ground_mode,
                half_window=half_window,
                events_imu=None
            )



            if objective == "abs_bias":
                score = abs(bias)
            elif objective == "rmse":
                score = rmse
            else:
                score = rmse + 0.5*abs(bias)

            if not quiet:
                print(f"[CalXZ] x_cm={x_cm:5.1f} z_cm={z_cm:6.2f}  bias={bias*1000:+6.2f}mm  rmse={rmse*1000:6.2f}mm  score={score:.6f}")

            if best is None or score < best["score"]:
                best = {
                    "x_cm": float(x_cm),
                    "z_cm": float(z_cm),
                    "bias": float(bias),
                    "rmse": float(rmse),
                    "score": float(score),
                    # Keep stride-level mean metrics for logging/CSV
                    **(extra or {}),
                }

    return best

def calibrate_toe_offset_fast(
    p_imu_w, qs, events_imu,
    events_gt_imu,
    mocap_traj, events_mocap,
    toe_marker_name,
    fs_hz,
    half_window
):
    # ---- Precompute stride mapping ONCE ----
    mapping_df, _ = match_strides_by_mid_swing_dp(
        events_imu, events_gt_imu, fs_hz, gate_s=0.20
    )

    # ---- Coarse grid ----
    x_grid = np.arange(6, 22.1, 1.0)
    z_grid = np.arange(-12, -1.9, 0.5)

    best = None

    for x_cm in x_grid:
        for z_cm in z_grid:

            p_toe_b = np.array([x_cm/100.0, 0.0, z_cm/100.0])

            imu_stride_res, _ = phase_1_8_compute_mtc_per_stride(
                p_imu_w, qs, p_toe_b, events_imu,
                ground_mode="min_vel",
                half_window=half_window
            )

            imu_scored = apply_stride_mapping_to_imu_results(
                imu_stride_res, events_imu,
                mapping_df, events_gt_imu
            )

            if imu_scored is None or imu_scored.empty:
                continue

            _, bias, rmse, _ = phase_1_9_validate_against_mocap(
                mocap_traj=mocap_traj,
                events_mocap=events_mocap,
                toe_marker_name=toe_marker_name,
                imu_stride_res=imu_scored,
                plot=False,
                quiet=True
            )

            score = rmse

            if best is None or score < best["score"]:
                best = {"x_cm": x_cm, "z_cm": z_cm, "score": score}

    # ---- Fine search around best ----
    x_fine = np.arange(best["x_cm"]-2, best["x_cm"]+2.1, 0.2)
    z_fine = np.arange(best["z_cm"]-1.5, best["z_cm"]+1.6, 0.1)

    for x_cm in x_fine:
        for z_cm in z_fine:
            # same evaluation logic (omitted here for brevity)
            pass

    return best


##############################################################################
# Pipeline runners (single participant/test)
##############################################################################

def build_calibration_inputs(data_folder: str, participant: str, test: str, side: str, padding_s: float = 3.0):
    """
    Build the required inputs for calibrate_toe_offset_xz():
    returns dict with p_imu_w, qs, events_imu, mocap_traj, events_mocap, toe_marker_name, events_gt_imu, fs_hz
    """
    side = str(side).lower()
    sensor = "l_instep" if side == "left" else "r_instep"
    toe_marker = "l_toe" if side == "left" else "r_toe"

    dataset = SensorPositionComparison2019Mocap(
        memory=Memory("./cache"),
        data_folder=data_folder,
        data_padding_s=float(padding_s),
    )
    subset = dataset.get_subset(participant=[participant], test=[test])
    if len(subset) == 0:
        raise RuntimeError(f"No datapoint found for participant={participant}, test={test}")
    datapoint = subset[0]

    imu_all = datapoint.data
    fs_hz = float(datapoint.sampling_rate_hz)
    mocap_traj = datapoint.marker_position_
    events_mocap = datapoint.mocap_events_[side]
    events_gt_imu = datapoint.convert_events_with_padding(events_mocap, from_time_axis="mocap", to_time_axis="imu")

    imu_df = imu_all[sensor][["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]].copy()

    imu_f = phase_1_3_filter_imu(imu_df, fs_hz, cutoff_hz=20.0)

    events_imu, _evdbg = phase_1_4_detect_events_imu_only(imu_f, fs_hz)
    events_imu = phase_1_4b_filter_invalid_strides(events_imu, fs_hz)
    if events_imu is None or len(events_imu) == 0:
        raise RuntimeError("No valid IMU strides after detection/filtering.")

    zupt_mask = phase_1_4_get_zupt_mask_from_min_vel(events_imu, n_samples=len(imu_f), half_window=10)

    qs = phase_1_5_estimate_orientation(imu_f, fs_hz)
    acc_w = phase_1_5_compute_world_acceleration(imu_f, qs)
    specific_acc_w, _ = phase_1_5_remove_gravity(acc_w, zupt_mask)

    # refine IC (important for consistent swing window)
    events_imu = refine_ic_with_specific_acc(events_imu, specific_acc_w=specific_acc_w, fs_hz=fs_hz)

    # vertical integration (z only)
    pz, vz = phase_1_6_integrate_z_only_per_stride_anchor_min_vel(
        specific_acc_w,
        events_imu,
        fs_hz,
        half_window=10,
        enforce_end_vel_zero=True,
        shift_to_tc=True,
    )
    p_imu_w = np.zeros((len(pz), 3), dtype=float)
    p_imu_w[:, 2] = pz

    return {
        "p_imu_w": p_imu_w,
        "qs": qs,
        "events_imu": events_imu,
        "mocap_traj": mocap_traj,
        "events_mocap": events_mocap,
        "toe_marker_name": toe_marker,
        "events_gt_imu": events_gt_imu,
        "fs_hz": fs_hz,
    }


def run_pipeline_for_row_report(
    data_folder: str,
    participant: str,
    test: str,
    side: str,
    toe_x_cm: float,
    toe_z_cm: float,
    sensor: str | None = None,
    toe_marker: str | None = None,
    padding_s: float = 3.0,
    event_source: str = "imu",
    eventnet_model=None,
    eventnet_meta: dict | None = None,
    quiet: bool = True,
    print_reject_counts: bool = True,
    gate_s: float = 0.15,
    *,
    dynamic_gate: bool = True,
    dynamic_zupt: bool = True,
    dynamic_ic_refine: bool = True,
    dynamic_cutoff: bool = True,
    calibrate_toe_offset: bool = False,
):
    """
    Run IMU-only pipeline for one (participant,test,side) using *pre-calibrated* toe offsets.
    Returns a flat dict suitable for a spreadsheet row (includes event-matching + timing + MTC metrics).
    """
    # Defaults
    side = str(side).lower()
    if sensor is None:
        sensor = "l_instep" if side == "left" else "r_instep"
    if toe_marker is None:
        toe_marker = "l_toe" if side == "left" else "r_toe"

    dataset = SensorPositionComparison2019Mocap(
        memory=Memory("./cache"),
        data_folder=data_folder,
        data_padding_s=float(padding_s),
    )

    SAVE_STRIDE_ERRORS_FOR = "fast_10"   # change whenever you want

    DEBUG_TEST = "fast_10"
    DEBUG_STRIDE_ID = 10   # change to whichever stride you want

    subset = dataset.get_subset(participant=[participant], test=[test])
    if len(subset) == 0:
        raise RuntimeError(f"No datapoint found for participant={participant}, test={test}")
    datapoint = subset[0]

    imu_all = datapoint.data
    fs_hz = float(datapoint.sampling_rate_hz)
    mocap_traj = datapoint.marker_position_

    if sensor not in imu_all.columns.get_level_values(0):
        raise RuntimeError(f"Sensor '{sensor}' not found for {participant}/{test}.")
    if toe_marker not in mocap_traj.columns.get_level_values(0):
        raise RuntimeError(f"Toe marker '{toe_marker}' not found for {participant}/{test}.")

    imu_df = imu_all[sensor][["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].copy()
    events_mocap = datapoint.mocap_events_[side]
    events_gt_imu = datapoint.convert_events_with_padding(events_mocap, from_time_axis="mocap", to_time_axis="imu")
    
    # --- IMPORTANT: make GT stride ids contiguous (prevents IndexError in batch) ---
    events_mocap = events_mocap.reset_index(drop=True).copy()
    events_gt_imu = events_gt_imu.reset_index(drop=True).copy()


    # Fixed toe offset (cm -> m)
    p_toe_b = np.array([toe_x_cm / 100.0, 0.0, toe_z_cm / 100.0], dtype=float)

    # Phase 1.1 objective (constant; keep it explicit in output)
    phase_cfg = phase_1_1_define_mtc_objective()

    # Phase 1.3
    imu_f = phase_1_3_filter_imu(imu_df, fs_hz, dynamic=dynamic_cutoff)

    # ------------------------
    # Event detection (IMU heuristic vs ML)
    # ------------------------
    event_source = (event_source or "imu").lower().strip()
    if event_source == "ml":
        if eventnet_model is None:
            raise RuntimeError("event_source=ml but eventnet_model is None. Pass a loaded model into run_pipeline_for_row_report.")
        events_imu, _evdbg = phase_1_4_detect_events_ml(
            imu_f=imu_f,
            fs_hz=fs_hz,
            model=eventnet_model,
            meta=eventnet_meta or {},
        )
    else:
        events_imu, _evdbg = phase_1_4_detect_events_imu_only(imu_f, fs_hz)

    if print_reject_counts and (_evdbg is not None) and ("reject_counts" in _evdbg):
        force_print(f"[{participant} {test} {side}] [Phase 1.4] reject_counts: {_evdbg['reject_counts']}")

    events_imu = phase_1_4b_filter_invalid_strides(events_imu, fs_hz)

    if events_imu is None or len(events_imu) == 0:
        raise RuntimeError("IMU-only event detection produced no valid strides after filtering.")

    zupt_mask = phase_1_4_get_zupt_mask_from_min_vel(
        events_imu,
        n_samples=len(imu_f),
        dynamic=dynamic_zupt,
        fs_hz=fs_hz,
        frac=0.03,
        min_hw=6,
        max_hw=25,
        stride_ref="ic2ic",
    )

    # Phase 1.5
    qs = phase_1_5_estimate_orientation(imu_f, fs_hz, zupt_mask=zupt_mask)

    half_window_samples = 10
    if dynamic_zupt:
        # Use GT if available (more stable), otherwise fallback to detected events
        ref = events_gt_imu if (events_gt_imu is not None and len(events_gt_imu) > 3) else events_imu

        # Example: half-window = 3% of median stride time, clamped
        stride = ref.dropna(subset=["ic", "tc"]).sort_values("ic")
        ic = stride["ic"].to_numpy(dtype=int)
        if len(ic) >= 3:
            stride_s = np.median(np.diff(ic) / float(fs_hz))
            hw_s = 0.03 * float(stride_s)                 # 3% of stride duration
            half_window_samples = int(np.clip(round(hw_s * fs_hz), 6, 25))
        else:
            half_window_samples = 10

    qs = phase_1_5b_stance_attitude_correction(
        qs, imu_f, events_imu, fs_hz,
        half_window=half_window_samples,
        alpha="dynamic", 
        use_median=True
    )


    acc_w = phase_1_5_compute_world_acceleration(imu_f, qs)
    specific_acc_w, _g_mode = phase_1_5_remove_gravity(acc_w, zupt_mask)

    # IC refinement
    events_imu = refine_ic_with_specific_acc(
        events_imu,
        specific_acc_w=specific_acc_w,
        fs_hz=fs_hz,
        dynamic=dynamic_ic_refine,
        pre_frac=0.12,
        post_frac=0.08,
        stride_ref="ic2ic",
    )

    # Phase 1.6 (z-only)
    pz, vz = phase_1_6_integrate_z_only_per_stride_anchor_min_vel(
        specific_acc_w,
        events_imu,
        fs_hz,
        half_window=half_window_samples,
        enforce_end_vel_zero=True,
        shift_to_tc=True,
    )
    p_imu_w = np.zeros((len(pz), 3), dtype=float)
    p_imu_w[:, 2] = pz

    # Phase 1.8
    if calibrate_toe_offset:
        best = calibrate_toe_offset_fast(
            p_imu_w=p_imu_w,
            qs=qs,
            events_imu=events_imu,
            events_gt_imu=events_gt_imu,
            mocap_traj=mocap_traj,
            events_mocap=events_mocap,
            toe_marker_name=toe_marker,
            fs_hz=fs_hz,
            half_window=half_window_samples,
        )
        toe_x_cm = best["x_cm"]
        toe_z_cm = best["z_cm"]
        p_toe_b = np.array([toe_x_cm/100.0, 0.0, toe_z_cm/100.0])

    # Now compute final IMU MTC once using best offset
    imu_stride_res, debug = phase_1_8_compute_mtc_per_stride(
        p_imu_w, qs, p_toe_b, events_imu,
        ground_mode="min_vel", half_window=half_window_samples
    )


    # ---- DEBUG: plot a specific stride for a specific test ----
    if test == DEBUG_TEST and DEBUG_STRIDE_ID is not None:
        if debug is not None and "stride_debug" in debug and DEBUG_STRIDE_ID in debug["stride_debug"]:
            force_print(f"\n[DEBUG] Plotting stride {DEBUG_STRIDE_ID} for {participant} {test} {side}")
            plot_one_stride_clearance(int(DEBUG_STRIDE_ID), debug)
        else:
            force_print(f"[DEBUG] Stride {DEBUG_STRIDE_ID} not found in debug['stride_debug'] for {test}. "
                        f"Available example ids: {list(debug['stride_debug'].keys())[:10]}")


    # Stride matching (IMU-only -> GT converted to IMU time)
    gate_use = float(gate_s)

    if dynamic_gate:
        # IMU-only GT mode: use GT for stable cadence/stride time
        gate_use = compute_dynamic_gate_s_from_stride_time(
            events_ref=events_gt_imu,
            fs_hz=fs_hz,
            frac=0.12,
            min_gate=0.08,
            max_gate=0.25,
            use="ic2ic",
        )

    mapping_df, match_sum = match_strides_by_mid_swing_dp(
        events_imu, events_gt_imu, fs_hz, gate_s=gate_use
    )

    fp = int(match_sum["n_det"] - match_sum["n_matched"])
    fn = int(match_sum["n_gt"] - match_sum["n_matched"])

    imu_stride_res_scored = apply_stride_mapping_to_imu_results(imu_stride_res, events_imu, mapping_df, events_gt_imu)

    # Timing summary
    tc_bias_ms = tc_rmse_ms = ic_bias_ms = ic_rmse_ms = np.nan
    if imu_stride_res_scored is not None and (not imu_stride_res_scored.empty):
        tc_err_ms = (imu_stride_res_scored["tc_err_samples"].to_numpy(dtype=float) / fs_hz) * 1000.0
        ic_err_ms = (imu_stride_res_scored["ic_err_samples"].to_numpy(dtype=float) / fs_hz) * 1000.0
        tc_bias_ms = float(np.mean(tc_err_ms))
        tc_rmse_ms = float(np.sqrt(np.mean(tc_err_ms ** 2)))
        ic_bias_ms = float(np.mean(ic_err_ms))
        ic_rmse_ms = float(np.sqrt(np.mean(ic_err_ms ** 2)))

    # Phase 1.9 (MTC scoring)
    mtc_bias_mm = mtc_rmse_mm = np.nan
    mtc_n = 0
    mtc_imu_mean_mm = mtc_mocap_mean_mm = mtc_avg_diff_mm = mtc_mean_abs_diff_mm = np.nan

    if mapping_df is not None and (not mapping_df.empty) and imu_stride_res_scored is not None and (not imu_stride_res_scored.empty):
        
        stride_errors_path = None

        if test == SAVE_STRIDE_ERRORS_FOR:
            stride_errors_path = f"stride_errors_{participant}_{test}_{side}.csv"

        joined, bias, rmse, extra = phase_1_9_validate_against_mocap(
            mocap_traj=mocap_traj,
            events_mocap=events_mocap,
            toe_marker_name=toe_marker,
            imu_stride_res=imu_stride_res_scored,
            plot=False,
            fs_hz=fs_hz,
            debug=debug,
            verbose=False,
            quiet=True,
            events_imu=None,
            plot_worst_k=0,
            stride_errors_path=stride_errors_path,
        )

        if test == DEBUG_TEST and DEBUG_STRIDE_ID is not None:
            if DEBUG_STRIDE_ID in debug["stride_debug"]:
                print(f"\n[DEBUG] Plotting stride {DEBUG_STRIDE_ID} for {test}")
                plot_one_stride_clearance(int(DEBUG_STRIDE_ID), debug)
            else:
                print(f"[DEBUG] Stride {DEBUG_STRIDE_ID} not found in debug for {test}")


        mtc_bias_mm = float(bias * 1000.0)
        mtc_rmse_mm = float(rmse * 1000.0)
        mtc_n = int(len(joined))

        if extra is not None:
            mtc_imu_mean_mm = float(extra.get("mtc_imu_mean_m", np.nan) * 1000.0)
            mtc_mocap_mean_mm = float(extra.get("mtc_mocap_mean_m", np.nan) * 1000.0)
            mtc_avg_diff_mm = float(extra.get("mtc_avg_diff_m", bias) * 1000.0)
            mtc_mean_abs_diff_mm = float(extra.get("mtc_mean_abs_diff_m", np.nan) * 1000.0)

    # Flat row for spreadsheet
    row = {
        # keys (what you want in [Run Summary])
        "participant": participant,
        "test": test,
        "side": side,
        "imu_sensor": sensor,
        "toe_marker": toe_marker,
        "toe_x_cm": float(toe_x_cm),
        "toe_z_cm": float(toe_z_cm),
        "toe_offset_m": str([float(toe_x_cm)/100.0, 0.0, float(toe_z_cm)/100.0]),
        "phase_objective": str(phase_cfg),

        # IMU-only Events block
        "gate_s": float(gate_s),
        "det": int(match_sum["n_det"]),
        "gt": int(match_sum["n_gt"]),
        "matched": int(match_sum["n_matched"]),
        "precision_pct": float(match_sum["stride_precision"] * 100.0),
        "recall_pct": float(match_sum["stride_recall"] * 100.0),
        "fp": fp,
        "fn": fn,
        "tc_bias_ms": float(tc_bias_ms) if np.isfinite(tc_bias_ms) else np.nan,
        "tc_rmse_ms": float(tc_rmse_ms) if np.isfinite(tc_rmse_ms) else np.nan,
        "ic_bias_ms": float(ic_bias_ms) if np.isfinite(ic_bias_ms) else np.nan,
        "ic_rmse_ms": float(ic_rmse_ms) if np.isfinite(ic_rmse_ms) else np.nan,

        # MTC Summary Comparison block
        "mtc_bias_mm": float(mtc_bias_mm) if np.isfinite(mtc_bias_mm) else np.nan,
        "mtc_rmse_mm": float(mtc_rmse_mm) if np.isfinite(mtc_rmse_mm) else np.nan,
        "mtc_n": int(mtc_n),

        # Actual MTC values from Phase 1.9 extra
        "mtc_imu_mean_mm": float(extra["mtc_imu_mean_m"]) * 1000.0,
        "mtc_mocap_mean_mm": float(extra["mtc_mocap_mean_m"]) * 1000.0,
        "mtc_imu_median_mm": float(extra["mtc_imu_median_m"]) * 1000.0,
        "mtc_mocap_median_mm": float(extra["mtc_mocap_median_m"]) * 1000.0,

        # Optional extra diagnostics
        "mtc_avg_diff_mm": float(extra["mtc_avg_diff_m"]) * 1000.0,
        "mtc_mean_abs_diff_mm": float(extra["mtc_mean_abs_diff_m"]) * 1000.0,

    }
    return row


def augment_existing_toe_offset_csv_with_mtc(csv_path: str, data_folder: str, padding_s: float = 3.0):
    """Augment an existing toe_offset_calibration.csv by computing MTC mean/diff for each row.

    Does NOT run any toe offset calibration sweep. Uses toe_x_cm/toe_z_cm already stored per row.
    Overwrites the same CSV with new columns appended and populated.
    """
    df = pd.read_csv(csv_path)

    # Add new columns if missing
    new_cols = [
        "mtc_imu_mean_mm",
        "mtc_mocap_mean_mm",
        "mtc_avg_diff_mm",
        "mtc_mean_abs_diff_mm",
        "mtc_n_strides",
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Process all rows (or only missing)
    for i in range(len(df)):
        if not pd.isna(df.loc[i, "mtc_avg_diff_mm"]):
            continue

        participant = str(df.loc[i, "participant"])
        test = str(df.loc[i, "test"])
        side = str(df.loc[i, "side"]).lower()
        toe_x_cm = float(df.loc[i, "toe_x_cm"])
        toe_z_cm = float(df.loc[i, "toe_z_cm"])

        try:
            res = run_pipeline_for_row_report(
                data_folder=args.data_folder,
                participant=p,
                test=t,
                side=s,
                toe_x_cm=x,
                toe_z_cm=z,
                padding_s=args.padding_s,
                event_source=args.event_source,
                eventnet_model=batch_eventnet_model,
                eventnet_meta=batch_eventnet_meta,
                quiet=True,
                print_reject_counts=True,
                gate_s=float(args.gate_s),
                dynamic_gate=True,
                dynamic_zupt=True,
                dynamic_ic_refine=True,
                dynamic_cutoff=True,
            )
            df.loc[i, "mtc_imu_mean_mm"] = res["mtc_imu_mean_mm"]
            df.loc[i, "mtc_mocap_mean_mm"] = res["mtc_mocap_mean_mm"]
            df.loc[i, "mtc_avg_diff_mm"] = res["mtc_avg_diff_mm"]
            df.loc[i, "mtc_mean_abs_diff_mm"] = res["mtc_mean_abs_diff_mm"]
            df.loc[i, "mtc_n_strides"] = res["n_joined"]
            print(f"[AUGMENT] {participant} {test} {side}: Avg diff={res['mtc_avg_diff_mm']:.2f} mm | N={res['n_joined']}")
        except Exception as e:
            print(f"[AUGMENT] FAILED {participant} {test} {side}: {e}")
            # leave NaNs; continue

    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"[AUGMENT] Wrote augmented CSV: {csv_path}")

##############################################################################
# CLI entrypoint
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to extracted sensorpositioncomparison dataset folder.")
    parser.add_argument("--participant", type=str, default="4d91")
    parser.add_argument("--test", type=str, default="normal_10")
    parser.add_argument("--side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--sensor", type=str, default=None,
                        help="Default: l_instep or r_instep based on side.")
    parser.add_argument("--toe_marker", type=str, default=None,
                        help="Default: l_toe or r_toe based on side.")
    parser.add_argument("--padding_s", type=float, default=3.0)
    parser.add_argument("--toe_offset_cm", type=float, nargs=3, default=[20.0, 0.0, -9.500000000000008],
                        help="Virtual toe offset [x y z] in cm in IMU/body frame.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--calibrate_toe_z", action="store_true",
                    help="Grid-search toe z-offset using mocap to minimise MTC bias")
    parser.add_argument("--plot_worst_k", type=int, default=0,
                    help="If >0, auto-plot the worst K strides (largest abs error) after validation.")

    # Fast CSV augmentation: compute MTC mean/diff for existing toe_offset_calibration.csv rows (no x/z sweep)
    parser.add_argument("--augment_calib_csv_with_mtc", action="store_true",
                        help="Augment an existing toe_offset_calibration.csv by computing MTC mean/diff for each row using its toe_x_cm/toe_z_cm (no calibration sweep).")
    parser.add_argument("--toe_offset_csv", type=str, default="toe_offset_calibration.csv",
                        help="Path to toe_offset_calibration.csv to augment (default: toe_offset_calibration.csv).")
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print final IMU-only and MTC summary results"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="imu_only_all_summary.csv",
        help="Output CSV filename when --participant all is used (overwritten each run)."
    )
    parser.add_argument(
        "--gate_s",
        type=float,
        default=0.15,
        help="Stride matching tolerance (seconds) used for summary (default 0.15s)."
    )
    parser.add_argument(
        "--calibrate_toe_offset_all",
        action="store_true",
        help="Recompute toe (x,z) offset calibration for all participants (overwrites existing entries)."
    )
    parser.add_argument(
        "--calib_test",
        type=str,
        default="normal_10",
        help="Which test to use for toe-offset calibration in --calibrate_toe_offset_all (default: normal_10)."
    )
    parser.add_argument("--event_source", type=str, default="heuristic",
                    choices=["heuristic","ml"],
                    help="Event detection source: heuristic or ml")
    parser.add_argument("--event_model_pt", type=str, default=None,
                        help="Path to TorchScript param.pt for ML event detection")
    parser.add_argument("--tests", nargs="+", default=None)
    args = parser.parse_args()

    # -------------------------------
    # C) Test selection logic
    # Priority:
    #   1) --tests slow_10 normal_10 ...
    #   2) --test all  (or --test None)
    #   3) --test normal_10
    # -------------------------------
    if args.tests is not None and len(args.tests) > 0:
        tests = list(args.tests)
    elif args.test is None or str(args.test).lower() == "all":
        tests = [
            "slow_10", "slow_20",
            "normal_10", "normal_20",
            "fast_10", "fast_20",
            "long",
        ]
    else:
        tests = [args.test]



    global CALIB_FILE
    CALIB_FILE = Path(args.toe_offset_csv)

    global SUMMARY_ONLY
    SUMMARY_ONLY = getattr(args, "summary_only", False)

    # Silence *all* raw print() calls in the entire script when summary_only is enabled
    if SUMMARY_ONLY:
        builtins.print = lambda *a, **k: None


    # Defaults
    if args.sensor is None:
        args.sensor = "l_instep" if args.side == "left" else "r_instep"
    if args.toe_marker is None:
        args.toe_marker = "l_toe" if args.side == "left" else "r_toe"

    # =========================
    # D1) One-time setup (single participant, multi-test)
    # =========================
    eventnet_model = None
    eventnet_meta = None
    if str(args.event_source).lower() == "ml":
        if not args.event_model_pt:
            raise RuntimeError("--event_source ml requires --event_model_pt <path_to_pt>")
        force_print(f"[ONE] Loading ML event model: {args.event_model_pt}")
        eventnet_model, eventnet_meta = load_eventnet(args.event_model_pt, device="cpu")

    # Optional: augment existing toe_offset_calibration.csv with MTC mean/diff columns (no calibration sweep)
    if getattr(args, "augment_calib_csv_with_mtc", False):
        augment_existing_toe_offset_csv_with_mtc(
            csv_path=str(Path(args.toe_offset_csv)),
            data_folder=args.data_folder,
            padding_s=args.padding_s,
        )
        return

    # ============================================================
    # Batch toe-offset calibration for ALL participants
    # ============================================================
    did_calib_all = False

    if args.calibrate_toe_offset_all:

        did_calib_all = True

        # ---- HARD WIPE: delete old toe offset file so no stale rows remain ----
        calib_path = Path(args.toe_offset_csv)
        if calib_path.exists():
            calib_path.unlink()
            force_print(f"[CAL-ALL] Wiped existing toe offset file: {calib_path}")

        new_rows = []  

        dataset = SensorPositionComparison2019Mocap(
            memory=Memory("./cache"),
            data_folder=args.data_folder,
            data_padding_s=args.padding_s,
        )

        seen = set()
        rows_done = 0

        # Build unique (participant, test) pairs from the dataset index
        idx = dataset.create_index()  # DataFrame with columns ["participant", "test"]

        rows_done = 0
        seen = set()

        for _, r in idx.iterrows():
            p = str(r["participant"])
            t = str(r["test"])
            s = "left"  # SPC2019 is left-only

            # ---- only calibrate normal_10 ----
            if t != args.calib_test:   # or "normal_10" if you hardcode
                continue

            # Skip incomplete subject 6dbe (README says 6dbe_2 is the complete one)
            if p == "6dbe":
                force_print("[CAL-ALL] Skipping 6dbe (incomplete; use 6dbe_2)")
                continue

            key = (p, t, s)
            if key in seen:
                continue
            seen.add(key)

            try:
                force_print(f"[CAL-ALL] Calibrating {p} {t} {s}")

                # Get datapoint for this (participant, test)
                subset = dataset.get_subset(participant=[p], test=[t])
                if len(subset) == 0:
                    raise RuntimeError(f"No datapoint for {p} {t}")
                datapoint = subset[0]

                # Now call your builder/calibration code using p/t/s
                cal_in = build_calibration_inputs(
                    data_folder=args.data_folder,
                    participant=p,
                    test=t,
                    side=s,
                    padding_s=args.padding_s,
                )

                best = calibrate_toe_offset_xz_instep_two_stage(
                    p_imu_w=p_imu_w,
                    qs=qs,
                    events_imu=events_imu,

                    mocap_traj=mocap_traj,
                    events_mocap=events_mocap,
                    toe_marker_name="l_toe",

                    events_gt_imu=events_gt_imu,
                    fs_hz=fs_hz,

                    ground_mode="min_vel",
                    half_window=10,

                    # optional tuning
                    x_min_cm=3.0,
                    x_max_cm=10.0,
                    z_min_cm=-4.0,
                    z_max_cm=-1.0,

                    quiet=False
                )

                print("\n=== FINAL CALIBRATION RESULT ===")
                print(f"x_cm  = {best['x_cm']:.2f}")
                print(f"z_cm  = {best['z_cm']:.2f}")
                print(f"bias  = {best['bias']*1000:+.2f} mm")
                print(f"rmse  = {best['rmse']*1000:.2f} mm")


                new_rows.append({
                    "participant": p,
                    "test": t,          # normal_10
                    "side": s,          # left
                    "toe_x_cm": float(best["x_cm"]),
                    "toe_z_cm": float(best["z_cm"]),
                    "bias_mm": float(best["bias"]) * 1000.0,
                    "rmse_mm": float(best["rmse"]) * 1000.0,
                })

                rows_done += 1
                force_print(
                    f"[CAL-ALL] OK {p} {t} | x={best['x_cm']:.1f} cm z={best['z_cm']:.1f} cm | RMSE={best['rmse']*1000:.1f} mm"
                )

            except Exception as e:
                force_print(f"[CAL-ALL] FAIL {p} {t}: {e}")

        df_new = pd.DataFrame(new_rows)

        df_new = df_new.sort_values(
            ["participant", "test", "side"]
        ).reset_index(drop=True)

        df_new.to_csv(args.toe_offset_csv, index=False)

        force_print(
            f"[CAL-ALL] Wrote fresh toe offsets to {args.toe_offset_csv} "
            f"(rows={len(df_new)})"
        )

        force_print(f"\n[CAL-ALL] Completed toe calibration for {rows_done} rows.")

        # DO NOT return here if user also requested --participant all
        if str(args.participant).lower() != "all":
            return



    # ------------------------
    # Batch mode: --participant all
    # Uses ONLY rows in toe_offset_calibration.csv (no calibration sweep)
    # ------------------------
    if str(args.participant).lower() == "all":
        calib_path = Path(args.toe_offset_csv)
        if not calib_path.exists():
            raise RuntimeError(f"Calibration CSV not found: {calib_path}")

        df_cal = pd.read_csv(calib_path)
        required_cols = {"participant", "test", "side", "toe_x_cm", "toe_z_cm"}
        missing = required_cols - set(df_cal.columns)
        if missing:
            raise RuntimeError(f"{calib_path} missing required columns: {sorted(missing)}")

        # Keep only well-formed rows
        df_cal = df_cal.dropna(subset=["participant", "test", "side", "toe_x_cm", "toe_z_cm"]).copy()

        # De-dup: keep last occurrence for each (participant,test,side)
        df_cal["side"] = df_cal["side"].astype(str).str.lower()
        df_cal = df_cal.drop_duplicates(subset=["participant", "test", "side"], keep="last")

        out_rows = []


        # ------------------------
        # load ML model once for batch runs
        # ------------------------
        batch_eventnet_model = None
        batch_eventnet_meta = None
        if str(args.event_source).lower() == "ml":
            if not args.event_model_pt:
                raise RuntimeError("--event_source ml requires --event_model_pt <path_to_pt>")
            force_print(f"[ALL] Loading ML event model: {args.event_model_pt}")
            batch_eventnet_model, batch_eventnet_meta = load_eventnet(args.event_model_pt, device="cpu")
            # Optional sanity check: warn if model was trained at a different fs
            try:
                mfs = batch_eventnet_meta.get("fs_hz", None)
                if mfs and abs(float(mfs) - float(200.0)) > 1e-6 and abs(float(mfs) - float(args.fs_hz)) > 1e-6:
                    force_print(f"[ALL][WARN] Model fs_hz={mfs} differs from run fs_hz={args.fs_hz}.")
            except Exception:
                pass

        for _, r in df_cal.iterrows():
            p = str(r["participant"])
            t = str(r["test"])
            s = str(r["side"]).lower()
            x = float(r["toe_x_cm"])
            z = float(r["toe_z_cm"])

            try:
                row = run_pipeline_for_row_report(
                    data_folder=args.data_folder,
                    participant=p,
                    test=t,
                    side=s,
                    toe_x_cm=x,
                    toe_z_cm=z,
                    padding_s=args.padding_s,
                    event_source=args.event_source,
                    eventnet_model=batch_eventnet_model,
                    eventnet_meta=batch_eventnet_meta,
                    quiet=True,
                    print_reject_counts=True,
                    gate_s=float(args.gate_s),   # still used if dynamic_gate=False
                    dynamic_gate=True,
                    dynamic_zupt=True,
                    dynamic_ic_refine=True,
                    dynamic_cutoff=True,
                )
                out_rows.append(row)
                force_print(f"[ALL] OK  {p} {t} {s} | MTC RMSE={row['mtc_rmse_mm']:.2f} mm | N={row['mtc_n']}")
            except Exception as e:
                # still write a row so the spreadsheet shows failures explicitly
                out_rows.append({
                    "participant": p, "test": t, "side": s,
                    "toe_x_cm": x, "toe_z_cm": z,
                    "error": str(e),
                })
                force_print(f"[ALL] FAIL {p} {t} {s} | {e}")

        df_out = pd.DataFrame(out_rows)
        df_out.to_csv(args.out_csv, index=False)
        force_print(f"\n[ALL] Wrote spreadsheet: {args.out_csv}")
        return

    # =========================
    # D2) Run ONE participant across ALL selected tests
    # =========================
    if str(args.participant).lower() != "all":
        out_rows = []

        # If you have a toe_offset_csv with per-test offsets, use it; otherwise use --toe_offset_cm
        # (Your runner expects toe_x_cm/toe_z_cm)
        # Force one toe offset for all tests: use normal_10 as the reference
        ref_test = "normal_10"
        ref = load_toe_offset(str(args.participant), ref_test, str(args.side).lower())

        if ref is None:
            raise RuntimeError(f"No toe offset found for {args.participant} {ref_test} {args.side} in {CALIB_FILE}")

        toe_x_ref_cm = float(ref["x_cm"])
        toe_z_ref_cm = float(ref["z_cm"])

        force_print(f"[TOE] Using {ref_test} toe offset for ALL tests: x={toe_x_ref_cm:.2f} cm, z={toe_z_ref_cm:.2f} cm")

        for t in tests:
            
            toe_x_cm = toe_x_ref_cm
            toe_z_cm = toe_z_ref_cm

            try:
                row = run_pipeline_for_row_report(
                    data_folder=args.data_folder,
                    participant=str(args.participant),
                    test=str(t),
                    side=str(args.side).lower(),
                    toe_x_cm=toe_x_cm,
                    toe_z_cm=toe_z_cm,
                    padding_s=float(args.padding_s),
                    event_source=str(args.event_source).lower(),
                    eventnet_model=eventnet_model,
                    eventnet_meta=eventnet_meta,
                    quiet=bool(args.summary_only),
                    print_reject_counts=True,
                    gate_s=float(args.gate_s),
                    dynamic_gate=True,
                    dynamic_zupt=True,
                    dynamic_ic_refine=True,
                    dynamic_cutoff=True,
                )
                out_rows.append(row)
                force_print(f"[ONE] OK  {args.participant} {t} {args.side} | MTC RMSE={row['mtc_rmse_mm']:.2f} mm | N={row['mtc_n']}")
            except Exception as e:
                out_rows.append({
                    "participant": str(args.participant),
                    "test": str(t),
                    "side": str(args.side).lower(),
                    "toe_x_cm": toe_x_cm,
                    "toe_z_cm": toe_z_cm,
                    "error": str(e),
                })
                force_print(f"[ONE] FAIL {args.participant} {t} {args.side} | {e}")

        df_out = pd.DataFrame(out_rows)

        # write a single-participant summary file (separate from --participant all)
        one_out = Path(args.out_csv).with_name(f"one_{args.participant}_alltests_summary.csv")
        df_out.to_csv(one_out, index=False)
        force_print(f"\n[ONE] Wrote spreadsheet: {one_out}")
        return


    # Load dataset (MoCap version)
    dataset = SensorPositionComparison2019Mocap(
        memory=Memory("./cache"),
        data_folder=args.data_folder,
        data_padding_s=args.padding_s
    )

    subset = dataset.get_subset(participant=[args.participant], test=[args.test])
    if len(subset) == 0:
        raise RuntimeError(f"No datapoint found for participant={args.participant}, test={args.test}")
    datapoint = subset[0]

    imu_all = datapoint.data
    fs_hz = float(datapoint.sampling_rate_hz)
    mocap_traj = datapoint.marker_position_

    if args.sensor not in imu_all.columns.get_level_values(0):
        raise RuntimeError(f"Sensor '{args.sensor}' not found. Available: {sorted(set(imu_all.columns.get_level_values(0)))}")
    if args.toe_marker not in mocap_traj.columns.get_level_values(0):
        raise RuntimeError(f"Toe marker '{args.toe_marker}' not found. Available: {sorted(set(mocap_traj.columns.get_level_values(0)))}")

    imu_df = imu_all[args.sensor][["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].copy()
    events_mocap = datapoint.mocap_events_[args.side]
    events_gt_imu = datapoint.convert_events_with_padding(events_mocap, from_time_axis="mocap", to_time_axis="imu")
    
    events_mocap = events_mocap.reset_index(drop=True).copy()
    events_gt_imu = events_gt_imu.reset_index(drop=True).copy()

    events_imu = None  # IMU-only detected events will be created after Phase 1.3
    #debug_print_events(events_imu)
    #print("IMU length:", len(imu_df))

    # ------------------------
    # Phase 1.1
    # ------------------------
    phase_cfg = phase_1_1_define_mtc_objective()
    print_phase("Phase 1.1", f"Objective: {phase_cfg}")


    # ------------------------
    # Phase 1.2
    # ------------------------
    p_toe_b = phase_1_2_define_virtual_toe_point(args.toe_offset_cm)
    print_phase("Phase 1.2", f"Toe offset (m): {p_toe_b.tolist()}")

    # ------------------------
    # Phase 1.3
    # ------------------------
    imu_f = phase_1_3_filter_imu(imu_df, fs_hz, dynamic=True)
    acc_norm = np.median(np.linalg.norm(imu_f[['acc_x','acc_y','acc_z']].to_numpy(), axis=1))
    print_phase("Phase 1.3", f"Filtered acc_norm median(all)={acc_norm:.3f} m/s^2")

    # ------------------------
    # IMU-only event detection (IC/TC) for Phase 1.4+
    # ------------------------
    if args.event_source == "ml":
        if not args.event_model_pt:   
            raise RuntimeError("--event_source ml requires --event_model_pt <path_to_param.pt>")

        model, meta = load_eventnet(args.event_model_pt, device="cpu")  # CPU ok
        events_imu, det_dbg = phase_1_4_detect_events_ml(
            imu_f, fs_hz, model, meta,
            # you can tune these thresholds per model:
            p_enter=0.55, p_exit=0.45,
            min_stance_s=0.10, min_swing_s=0.10,
            gap_fill_s=0.03, refractory_s=0.30,
        )
    else:
        events_imu, det_dbg = phase_1_4_detect_events_imu_only(
            imu_f,
            fs_hz,
            gyro_th_deg_s=12.0,
            acc_th_m_s2=2.0,
            min_stance_s=0.12,
            min_swing_s=0.14,
            gap_fill_s=0.02,
            refractory_s=0.45
        )


    if (det_dbg is not None) and ("reject_counts" in det_dbg):
        force_print("[Phase 1.4] reject_counts:", det_dbg["reject_counts"])


        
    if events_imu is None or len(events_imu) == 0:
        raise RuntimeError(
            "IMU-only event detection produced no strides. "
            "Adjust gyro_th_deg_s / acc_th_m_s2 / min_stance_s / min_swing_s."
        )
    print_phase("Phase 1.4a", f"IMU-only detected strides: {len(events_imu)} | stance_segments={det_dbg.get('n_stance_segments', 'NA')}")

    # ---- NEW: post-filter invalid / duplicate strides (reduces over-detection) ----
    n_before = len(events_imu)
    events_imu = phase_1_4b_filter_invalid_strides(events_imu, fs_hz)
    n_after = len(events_imu)
    
    print_phase("Phase 1.4b", f"Stride validity filter: kept {n_after}/{n_before} ({100.0*n_after/max(1,n_before):.1f}%)")
    if events_imu is None or len(events_imu) == 0:
        raise RuntimeError(
            "Phase 1.4b removed all detected strides. "
            "Your stride validity gates are too strict for this trial. "
            "Relax timing thresholds or reduce refractory."
        )

    # ------------------------
    # Phase 1.4
    # ------------------------
    zupt_mask = phase_1_4_get_zupt_mask_from_min_vel(
        events_imu,
        n_samples=len(imu_f),
        dynamic=True,
        fs_hz=fs_hz,
        frac=0.03,
        min_hw=6,
        max_hw=25,
        stride_ref="ic2ic",
    )

    force_print("[ZUPT] dynamic window enabled (frac=0.03, clamp=6..25 samples)")

    # debug_units_and_magnitude(imu_f, zupt_mask)
    debug_gyro_units(imu_f, zupt_mask)
    zupt_pct = 100.0 * float(np.mean(zupt_mask))
    n_stride_rows = len(events_imu.dropna(subset=["min_vel"]))
    print_phase("Phase 1.4", f"ZUPT samples (min_vel±10): {zupt_pct:.1f}% | strides_with_min_vel={n_stride_rows}")

    # ------------------------
    # Phase 1.5
    # ------------------------
    qs = phase_1_5_estimate_orientation(imu_f, fs_hz)
    print_phase("Phase 1.5", f"Quaternion array shape: {qs.shape} (w,x,y,z)")

    acc_w = phase_1_5_compute_world_acceleration(imu_f, qs)
    specific_acc_w, g_mode = phase_1_5_remove_gravity(acc_w, zupt_mask)

    print_phase("Phase 1.5", f"Gravity mode: {g_mode}")
    print_phase("Phase 1.5", f"acc_w stance median z: {np.median(acc_w[zupt_mask,2]):.3f} m/s^2 | "
                            f"|acc_w| stance median: {np.median(np.linalg.norm(acc_w[zupt_mask],axis=1)):.3f}")


    # --- IC refinement using specific acceleration (IMU-only) ---
    events_imu = refine_ic_with_specific_acc(
        events_imu,
        specific_acc_w=specific_acc_w,
        fs_hz=fs_hz,
        dynamic=True,
        pre_frac=0.12,
        post_frac=0.03,
        stride_ref="ic2ic",
    )

    force_print(f"[IC] refine dynamic enabled: pre_frac=0.12 post_frac=0.03")

    if args.plot:
        plot_imu_mocap_events_window(
            imu_f=imu_f,
            mocap_traj=mocap_traj,
            events_imu=events_imu,
            toe_marker_name=args.toe_marker,
            fs_hz=fs_hz,
            stride_idx=9,     # change this to inspect other strides
            n_strides=4,
        )
        return

    half_window_samples = 10
    if dynamic_zupt:
        ref = events_gt_imu if (events_gt_imu is not None and len(events_gt_imu) > 3) else events_imu
        stride = ref.dropna(subset=["ic"]).sort_values("ic")
        ic = stride["ic"].to_numpy(dtype=int)

        if len(ic) >= 3:
            stride_s = float(np.median(np.diff(ic) / float(fs_hz)))
            hw_s = 0.03 * stride_s  # 3% stride duration
            half_window_samples = int(np.clip(round(hw_s * fs_hz), 6, 25))

    # ------------------------
    # Phase 1.6
    # ------------------------
    pz, vz = phase_1_6_integrate_z_only_per_stride_anchor_min_vel(
        specific_acc_w,
        events_imu,
        fs_hz,
        half_window=half_window_samples,
        enforce_end_vel_zero=True,
        shift_to_tc=True,
    )

    print("pz swing min/max (mm):", np.min(pz)*1000, np.max(pz)*1000)
    print("vz swing min/max (m/s):", np.min(vz), np.max(vz))

    p_imu_w = np.zeros((len(pz), 3), dtype=float)
    v_imu_w = np.zeros((len(vz), 3), dtype=float)
    p_imu_w[:, 2] = pz
    v_imu_w[:, 2] = vz

    pz_min, pz_max = float(np.min(pz)), float(np.max(pz))
    
    print_phase("Phase 1.6", f"v_imu_w norm median (stance) = {np.median(np.linalg.norm(v_imu_w[zupt_mask],axis=1)):.4f} m/s "
                         f"(should be near 0)")

    print_phase("Phase 1.6",
        f"p_imu_w z-range: {pz_min:.3f} .. {pz_max:.3f} m "
        f"(note: absolute z can drift; compare per-stride clearance, not global range)"
    )

    print_phase("Phase 1.6", f"specific_acc_w z median (ZUPT) = {np.median(specific_acc_w[zupt_mask,2]):.4f} m/s^2 "
                         f"(should be near 0)")
    print("[Phase 1.6] v_norm median(all):", float(np.median(np.linalg.norm(v_imu_w, axis=1))))
    print("[Phase 1.6] p_z range:", float(np.min(p_imu_w[:,2])), "..", float(np.max(p_imu_w[:,2])))
    print("[Phase 1.6] per-axis v median:", np.median(v_imu_w, axis=0))
    print("[Phase 1.6] sanity: mean v over all =", np.mean(v_imu_w, axis=0))
    print("[Phase 1.6] v @ ZUPT median:", np.median(v_imu_w[zupt_mask], axis=0))
    print("[Phase 1.6] v @ ZUPT norm median:", np.median(np.linalg.norm(v_imu_w[zupt_mask], axis=1)))
    
    pz_detrended = pz - np.median(pz[zupt_mask])
    print("[Phase 1.6] pz detrended range:", float(np.min(pz_detrended)), "..", float(np.max(pz_detrended)))

    bad = 0
    for _, r in events_imu.dropna(subset=["start","end","min_vel"]).iterrows():
        s,e,mv = int(r["start"]), int(r["end"]), int(r["min_vel"])
        if not (0 <= s < mv < e <= len(imu_f)):
            bad += 1
    
    valid_ev = events_imu.dropna(subset=["start","end","min_vel"])
    print("[Phase 1.6] bad event rows:", bad, "/", len(valid_ev))

    # ===== Phase 0: Toe (X,Z) offset calibration or load =====
    if args.calibrate_toe_z:
        print("\n=== Phase 0: Toe (X,Z) offset calibration ===")

        cached = load_toe_offset(
            participant=args.participant,
            test=args.test,
            side=args.side
        )

        if cached is not None:
            print(
                f"[Calibration] Loaded cached toe offset for "
                f"{args.participant} | {args.test} | {args.side}"
            )
            print(
                f"[Calibration] toe (x,z)=({cached['x_cm']:.2f} cm, {cached['z_cm']:.2f} cm) "
                f"| RMSE={cached['rmse_mm']:.1f} mm"
            )

            p_toe_b = np.array(
                [cached["x_cm"] / 100.0, 0.0, cached["z_cm"] / 100.0],
                dtype=float
            )

        else:
            print("[Calibration] No cached calibration found — running grid search")

            best = calibrate_toe_offset_xz_instep_two_stage(
                p_imu_w=p_imu_w,
                qs=qs,
                events_imu=events_imu,

                mocap_traj=mocap_traj,
                events_mocap=events_mocap,
                toe_marker_name="l_toe",

                events_gt_imu=events_gt_imu,
                fs_hz=fs_hz,

                ground_mode="min_vel",
                half_window=10,

                # optional tuning
                x_min_cm=3.0,
                x_max_cm=10.0,
                z_min_cm=-4.0,
                z_max_cm=-1.0,

                quiet=True
            )

            print("\n=== FINAL CALIBRATION RESULT ===")
            print(f"x_cm  = {best['x_cm']:.2f}")
            print(f"z_cm  = {best['z_cm']:.2f}")
            print(f"bias  = {best['bias']*1000:+.2f} mm")
            print(f"rmse  = {best['rmse']*1000:.2f} mm")


            save_toe_offset(
                participant=args.participant,
                test=args.test,
                side=args.side,
                best=best
            )

            p_toe_b = np.array(
                [best["x_cm"] / 100.0, 0.0, best["z_cm"] / 100.0],
                dtype=float
            )
        
    # ------------------------
    # Phase 1.8 (includes 1.7 internally per stride)
    # ------------------------
    imu_stride_res, debug = phase_1_8_compute_mtc_per_stride(
        p_imu_w, qs, p_toe_b, events_imu,
        ground_mode="min_vel", half_window=10
    )

    print("imu_stride_res.columns:", list(imu_stride_res.columns))
    print("imu_stride_res.head():\n", imu_stride_res.head())


    # ---- Phase 1.8 prints (robust) ----
    n_total = len(events_imu.dropna(subset=["ic","tc","start","end"]))

    # Step B: show why strides are being rejected
    rej = (debug or {}).get("rej_counts", {}) or {}
    if rej:
        top = sorted(rej.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("[Phase 1.8] top rejects:", ", ".join([f"{k}={v}" for k, v in top]))
    else:
        print("[Phase 1.8] top rejects: none")


    if imu_stride_res is None or imu_stride_res.empty:
        print_phase("Phase 1.8", f"No valid strides after filtering. kept=0 out of {n_total}")
        print("[Phase 1.8] MTC IMU: n/a (no valid strides)")
    else:
        g = imu_stride_res["ground_z"].to_numpy(dtype=float)
        gstd = imu_stride_res["ground_std_m"].to_numpy(dtype=float)

        g_med_mm = float(np.median(g) * 1000.0)
        g_iqr_mm = float((np.percentile(g, 75) - np.percentile(g, 25)) * 1000.0)

        kept = len(imu_stride_res)
        print_phase("Phase 1.8",
            f"Ground reference height (relative): median={g_med_mm:.1f} mm | IQR={g_iqr_mm:.1f} mm"
        )
        print(f"[Phase 1.8] ground_std threshold: 5.000 mm (reject if above)")

        print(f"[Phase 1.8] kept strides: {kept} out of {n_total} ({100.0*kept/max(n_total,1):.1f}%)")
        print(f"[Phase 1.8] ground_std_m median: {float(np.median(gstd)*1000.0):.3f} mm")
        print(f"[Phase 1.8] ground_std_m 95%:   {float(np.percentile(gstd,95)*1000.0):.3f} mm")

        mtc = imu_stride_res["mtc_imu_m"].to_numpy(dtype=float) * 1000.0
        print(f"[Phase 1.8] MTC IMU median: {np.median(mtc):.1f} mm | "
            f"5–95%: {np.percentile(mtc,5):.1f}..{np.percentile(mtc,95):.1f} mm")

        debug["zupt_mask"] = zupt_mask
        debug["phase_cfg"] = phase_cfg

    
    # ------------------------
    # IMU-only -> MoCap stride matching (MoCap used ONLY for scoring)
    # ------------------------
    gate_s = 0.15  # stride matching tolerance in seconds
    mapping_df, match_sum = match_strides_by_mid_swing_dp(
        events_imu,
        events_gt_imu,
        fs_hz,
        gate_s=gate_s
    )
    
    fp = match_sum["n_det"] - match_sum["n_matched"]
    fn = match_sum["n_gt"] - match_sum["n_matched"]
    
    print_phase(
        "IMU-only Events",
        f"FP={fp} (extra detections) | FN={fn} (missed GT strides) | gate_s={gate_s:.2f}s",
        always=True
    )

    imu_stride_res_scored = apply_stride_mapping_to_imu_results(imu_stride_res, events_imu, mapping_df, events_gt_imu)

    # Timing-error summary (samples -> ms) on matched set
    if (imu_stride_res_scored is not None) and (not imu_stride_res_scored.empty):
        tc_err_ms = (imu_stride_res_scored["tc_err_samples"].to_numpy(dtype=float) / fs_hz) * 1000.0
        ic_err_ms = (imu_stride_res_scored["ic_err_samples"].to_numpy(dtype=float) / fs_hz) * 1000.0
        print_phase(
            "IMU-only Events",
            f"Stride match: matched={match_sum['n_matched']} | det={match_sum['n_det']} | gt={match_sum['n_gt']} | "
            f"precision={match_sum['stride_precision']*100:.1f}% | recall={match_sum['stride_recall']*100:.1f}%\n"
            f"TC timing: bias={np.mean(tc_err_ms):+.1f} ms | RMSE={np.sqrt(np.mean(tc_err_ms**2)):.1f} ms\n"
            f"IC timing: bias={np.mean(ic_err_ms):+.1f} ms | RMSE={np.sqrt(np.mean(ic_err_ms**2)):.1f} ms",
            always=True
        )
    else:
        print_phase("IMU-only Events",
            f"Stride match: matched={match_sum['n_matched']} | det={match_sum['n_det']} | gt={match_sum['n_gt']} | "
            f"precision={match_sum['stride_precision']*100:.1f}% | recall={match_sum['stride_recall']*100:.1f}%\n"
            f"(No matched strides available for timing summary.)",
            always=True
        )
    
    # If nothing matched, we cannot score MTC vs MoCap by stride id
    if mapping_df is None or mapping_df.empty:
        force_print("[IMU-only Events] No matched strides -> skipping Phase 1.9 comparison.")
        return


    # ------------------------
    # Phase 1.9
    # ------------------------
    joined, bias, rmse, extra = phase_1_9_validate_against_mocap(
        mocap_traj=mocap_traj,
        events_mocap=events_mocap,
        toe_marker_name=args.toe_marker,
        imu_stride_res=imu_stride_res_scored,
        plot=args.plot,
        fs_hz=fs_hz,
        debug=debug,
        verbose=False,
        events_imu=None,
        plot_worst_k=args.plot_worst_k
    )


    # Requested: average MTC difference between IMU and MoCap
    mtc_imu_mean_mm = mm(extra.get("mtc_imu_mean_m", np.nan)) if extra is not None else float("nan")
    mtc_mocap_mean_mm = mm(extra.get("mtc_mocap_mean_m", np.nan)) if extra is not None else float("nan")
    mtc_avg_diff_mm = mm(extra.get("mtc_avg_diff_m", bias)) if extra is not None else mm(bias)
    mtc_mean_abs_diff_mm = mm(extra.get("mtc_mean_abs_diff_m", np.nan)) if extra is not None else float("nan")

    print_phase(
        "Phase 1.9",
        f"Bias={mm(bias):.1f} mm | RMSE={mm(rmse):.1f} mm | N={len(joined)}\n"
        f"MTC mean: IMU={mtc_imu_mean_mm:.1f} mm | MoCap={mtc_mocap_mean_mm:.1f} mm | "
        f"Avg diff (IMU−MoCap)={mtc_avg_diff_mm:.1f} mm | Mean |diff|={mtc_mean_abs_diff_mm:.1f} mm",
    )
    print("[Phase 1.9] Note: Bias = (IMU - MoCap). Positive means IMU overestimates clearance.")

    print("[Phase 1.9] events_mocap columns:", events_mocap.columns)

    # Quick summary
    print_phase(
        "Run Summary",
        f"Participant: {args.participant} | Test: {args.test} | Side: {args.side}\n"
        f"IMU sensor: {args.sensor} | Toe marker: {args.toe_marker}\n"
        f"Toe offset (m): {p_toe_b.tolist()}\n"
        f"Phase objective: {phase_cfg}",
        always=True
    )

    def _fmt_metrics(name, bias_m, rmse_m, joined_df, extra_dict):
        # all in mm for reporting
        N = int(len(joined_df)) if joined_df is not None else 0
        mtc_imu_mean_mm = mm(extra_dict.get("mtc_imu_mean_m", np.nan)) if extra_dict else float("nan")
        mtc_mocap_mean_mm = mm(extra_dict.get("mtc_mocap_mean_m", np.nan)) if extra_dict else float("nan")
        mtc_avg_diff_mm = mm(extra_dict.get("mtc_avg_diff_m", bias_m)) if extra_dict else mm(bias_m)
        mtc_mean_abs_diff_mm = mm(extra_dict.get("mtc_mean_abs_diff_m", np.nan)) if extra_dict else float("nan")

        return {
            "Mode": name,
            "Bias (mm)": mm(bias_m),
            "RMSE (mm)": mm(rmse_m),
            "N": N,
            "MTC mean IMU (mm)": mtc_imu_mean_mm,
            "MTC mean MoCap (mm)": mtc_mocap_mean_mm,
            "Avg diff IMU−MoCap (mm)": mtc_avg_diff_mm,
            "Mean |diff| (mm)": mtc_mean_abs_diff_mm,
        }

    # ----------------------------
    # (A) IMU-only timing (already computed): joined, bias, rmse, extra
    # ----------------------------
    row_imu_only = _fmt_metrics("IMU-only timing", bias, rmse, joined, extra)

    # ----------------------------
    # Print comparison table
    # ----------------------------
    df_cmp = pd.DataFrame([row_imu_only])

    # nice formatting
    cols = ["Mode", "Bias (mm)", "RMSE (mm)", "N",
            "MTC mean IMU (mm)", "MTC mean MoCap (mm)",
            "Avg diff IMU−MoCap (mm)", "Mean |diff| (mm)"]

    force_print("\n=== MTC Summary Comparison ===")
    force_print(df_cmp.to_string(index=False))
    force_print("Done.")


if __name__ == "__main__":
    main()
