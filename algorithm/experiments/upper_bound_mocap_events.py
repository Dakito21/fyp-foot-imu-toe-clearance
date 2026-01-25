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
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from joblib import Memory

from gaitmap_datasets.sensor_position_comparison_2019 import SensorPositionComparison2019Mocap

from ahrs.filters import Madgwick
import csv
from pathlib import Path

CALIB_FILE = Path("toe_offset_calibration.csv")

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
    csv_append_row_union_cols(str(CALIB_FILE), row)





def csv_append_row_union_cols(csv_path: str, row: dict):
    """
    Appends a row to csv_path. If the CSV exists, it will be rewritten with union columns
    so new columns (e.g., mtc_avg_diff_mm) are added without re-running calibration.
    """
    import os
    import pandas as pd

    new_row = pd.DataFrame([row])

    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)

        # add missing columns in either direction
        for c in new_row.columns:
            if c not in old.columns:
                old[c] = pd.NA
        for c in old.columns:
            if c not in new_row.columns:
                new_row[c] = pd.NA

        # align order: keep old columns first, new ones at end
        new_cols = [c for c in new_row.columns if c not in old.columns]
        out_cols = list(old.columns) + new_cols

        # concat
        out = pd.concat([old[out_cols], new_row[out_cols]], ignore_index=True)
    else:
        out = new_row

    out.to_csv(csv_path, index=False)


# ============================================================
# Utilities: Quaternions (minimal set)
# ============================================================

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

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def quat_norm(q):
    n = np.linalg.norm(q)
    return q if n < 1e-12 else q / n

def quat_from_omega(omega_rad_s, dt):
    """Small rotation quaternion from angular rate omega over dt."""
    angle = np.linalg.norm(omega_rad_s) * dt
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = omega_rad_s / (np.linalg.norm(omega_rad_s) + 1e-12)
    half = 0.5 * angle
    return quat_norm(np.array([np.cos(half), *(np.sin(half) * axis)], dtype=float))

def quat_rotate(q, v):
    """Rotate 3D vector v by quaternion q."""
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

# After you have qs, p_imu_w, events_imu, and can call phase_1_8 + phase_1_9
def calibrate_toe_offset_z(
    p_imu_w, qs, events_imu,
    mocap_traj, events_mocap,
    toe_marker_name, quat_mode,
    ground_mode="min_vel", half_window=10,
    z_min_cm=-12.0, z_max_cm=-2.0, step_cm=0.1,
    objective="rmse_plus_bias", verbose=False
):
    best = None

    for z_cm in np.arange(z_min_cm, z_max_cm + 1e-9, step_cm):
        p_toe_b = np.array([0.20, 0.0, z_cm/100.0])

        imu_stride_res, _ = phase_1_8_compute_mtc_per_stride(
            p_imu_w, qs, p_toe_b, events_imu,
            quat_mode=quat_mode,
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
    toe_marker_name, quat_mode,
    ground_mode="min_vel", half_window=10,
    x_min_cm=8.0, x_max_cm=24.0, x_step_cm=0.5,
    z_min_cm=-12.0, z_max_cm=-2.0, z_step_cm=0.1,
    objective="rmse_plus_bias", verbose=False
):
    """
    Grid-search toe (x,z) offset using MoCap to minimize stride-level MTC error.
    Returns best dict with x_cm, z_cm, bias, rmse, score.
    """
    best = None

    # Precompute MoCap MTC once (independent of IMU toe offset)
    # We'll reuse your Phase 1.9 method by calling it with plot=False, quiet=True.
    for x_cm in np.arange(x_min_cm, x_max_cm + 1e-9, x_step_cm):
        for z_cm in np.arange(z_min_cm, z_max_cm + 1e-9, z_step_cm):
            p_toe_b = np.array([x_cm/100.0, 0.0, z_cm/100.0])

            imu_stride_res, _ = phase_1_8_compute_mtc_per_stride(
                p_imu_w, qs, p_toe_b, events_imu,
                quat_mode=quat_mode,
                ground_mode=ground_mode,
                half_window=half_window
            )

            joined, bias, rmse, extra = phase_1_9_validate_against_mocap(
                mocap_traj=mocap_traj,
                events_mocap=events_mocap,
                toe_marker_name=toe_marker_name,
                imu_stride_res=imu_stride_res,
                plot=False,
                quiet=True,      # no prints/csv
                verbose=False,
                ground_mode=ground_mode,
                half_window=half_window,
                events_imu=events_imu  # allows min_vel carryover if needed
            )

            if objective == "abs_bias":
                score = abs(bias)
            elif objective == "rmse":
                score = rmse
            else:
                score = rmse + 0.5*abs(bias)

            if verbose:
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


def mm(x):  # meters -> millimeters
    return float(x) * 1000.0

def print_phase(tag, msg):
    print(f"\n[{tag}] {msg}")

def debug_print_events(events_imu, n=5):
    print("\n=== Debug: events_imu ===")
    print("Type:", type(events_imu))
    print("Columns:", getattr(events_imu, "columns", None))
    print("Head:")
    try:
        print(events_imu.head(n))
    except Exception as e:
        print("Could not print head:", e)

    # Check required columns
    required = ["ic", "tc"]
    if hasattr(events_imu, "columns"):
        missing = [c for c in required if c not in events_imu.columns]
        if missing:
            print("MISSING required columns:", missing)

def debug_units_and_magnitude(imu_f, zupt_mask):
    acc = imu_f[["acc_x","acc_y","acc_z"]].to_numpy()
    gyr = imu_f[["gyr_x","gyr_y","gyr_z"]].to_numpy()

    acc_norm = np.linalg.norm(acc, axis=1)
    gyr_norm = np.linalg.norm(gyr, axis=1)

    st = zupt_mask
    print("\n=== Debug: IMU magnitudes ===")
    print(f"acc_norm median (all):   {np.median(acc_norm):.3f}")
    print(f"acc_norm median (stance):{np.median(acc_norm[st]):.3f}")
    print(f"gyr_norm median (stance):{np.median(gyr_norm[st]):.3f}")

    print("Interpretation guide:")
    print("- If stance acc_norm ~ 9.8 -> acc is likely m/s^2 (good)")
    print("- If stance acc_norm ~ 1.0 -> acc is likely in g (you must *9.81)")

def debug_gyro_units(imu_f, zupt_mask):
    gyr = imu_f[["gyr_x","gyr_y","gyr_z"]].to_numpy()
    gnorm_all = np.linalg.norm(gyr, axis=1)
    gnorm_st  = np.linalg.norm(gyr[zupt_mask], axis=1)

    print("\n=== Debug: Gyro units check ===")
    print(f"gyro_norm median (all):   {np.median(gnorm_all):.3f}")
    print(f"gyro_norm median (stance):{np.median(gnorm_st):.3f}")
    print(f"gyro_norm 95% (stance):   {np.percentile(gnorm_st,95):.3f}")
    print("Guide:")
    print("- If typical values are ~0.1–3 -> likely rad/s")
    print("- If typical values are ~10–300 -> likely deg/s")

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
    plt.show()

# ============================================================
# Utilities: Filtering
# ============================================================

def lowpass_df(df, fs_hz, cutoff_hz=20.0, order=4):
    b, a = butter(order, cutoff_hz / (0.5 * fs_hz), btype="low", analog=False)
    out = df.copy()
    for col in df.columns:
        out[col] = filtfilt(b, a, df[col].to_numpy())
    return out

# ============================================================
# Phase 1.1 → 1.9 as individual functions
# ============================================================

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

def phase_1_3_filter_imu_signals(imu_df, fs_hz, cutoff_hz=20.0):
    """
    (1.3) IMU acquisition & filtering:
    - Low-pass accel and gyro to prevent high-frequency noise exploding under integration.
    """
    imu_f = imu_df.copy()
    imu_f[["acc_x", "acc_y", "acc_z"]] = lowpass_df(
        imu_f[["acc_x", "acc_y", "acc_z"]], fs_hz, cutoff_hz=cutoff_hz
    )
    imu_f[["gyr_x", "gyr_y", "gyr_z"]] = lowpass_df(
        imu_f[["gyr_x", "gyr_y", "gyr_z"]], fs_hz, cutoff_hz=cutoff_hz
    )
    return imu_f

def phase_1_4_get_zupt_mask_from_mocap(events_imu, n_samples):
    """
    For this dataset, each stride row is (tc -> ic -> min_vel) inside [start, end].
    So stance can be approximated as ic -> end.
    """
    zupt_mask = np.zeros(n_samples, dtype=bool)

    for _, row in events_imu.iterrows():
        ic = int(row["ic"])
        end = int(row["end"])
        if 0 <= ic < n_samples and 0 <= end < n_samples and end > ic:
            zupt_mask[ic:end] = True

    return zupt_mask

def phase_1_4_get_zupt_mask_from_min_vel(events_imu, n_samples, half_window=5):
    """
    ZUPT mask centered around min_vel (minimum velocity point) for each stride.
    This is far more reliable than ic->end for true zero-velocity updates.
    """
    zupt = np.zeros(n_samples, dtype=bool)
    for _, row in events_imu.dropna(subset=["min_vel"]).iterrows():
        mv = int(row["min_vel"])
        a = max(0, mv - half_window)
        b = min(n_samples, mv + half_window + 1)
        zupt[a:b] = True
    return zupt

def phase_1_5_estimate_orientation(imu_f, fs_hz, zupt_mask=None):
    """
    (1.5) Orientation using Madgwick (gyro+acc).

    Upgrades:
    - Auto-detect gyro units (deg/s vs rad/s) using stance (preferred) or whole-signal norm.
    """
    acc = imu_f[["acc_x","acc_y","acc_z"]].to_numpy()
    gyr = imu_f[["gyr_x","gyr_y","gyr_z"]].to_numpy()

    # --- Auto-detect gyro units ---
    gnorm_all = np.linalg.norm(gyr, axis=1)
    if zupt_mask is not None and np.any(zupt_mask):
        gnorm_ref = np.linalg.norm(gyr[zupt_mask], axis=1)
        ref_name = "stance"
    else:
        gnorm_ref = gnorm_all
        ref_name = "all"

    med = float(np.median(gnorm_ref))
    p95 = float(np.percentile(gnorm_ref, 95))

    # Heuristic: if typical magnitudes are in the tens, it's almost certainly deg/s.
    # If typical magnitudes are ~0.1–3, it's likely rad/s.
    # if med > 6.0:
    #     gyro_units = "deg/s"
    #     gyr_rad = np.deg2rad(gyr)
    # else: 
    #     gyro_units = "rad/s"
    #     gyr_rad = gyr

    gyro_units = "deg/s"
    gyr_rad = np.deg2rad(gyr)

    print(f"[Phase 1.5] Gyro auto-detect using {ref_name}: median={med:.3f}, p95={p95:.3f} -> {gyro_units}")

    f = Madgwick(frequency=fs_hz)

    Q = np.zeros((len(acc), 4))
    Q[0] = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

    for i in range(1, len(acc)):
        Q[i] = f.updateIMU(Q[i-1], gyr=gyr_rad[i], acc=acc[i])

    return Q

def phase_1_5_compute_world_acceleration_with_quat_fix(imu_f, qs, zupt_mask):
    """
    (1.5) Rotate body acceleration into a world frame, BUT fix quaternion direction mismatch.
    We test:
      A) acc_w = rotate(q, acc_b)
      B) acc_w = rotate(conj(q), acc_b)
    and choose the one that makes stance look like gravity (~9.81 m/s^2) on the vertical axis.
    """
    acc_b = imu_f[["acc_x", "acc_y", "acc_z"]].to_numpy()
    n = len(acc_b)

    def rotate_with(qs_use):
        acc_w = np.zeros_like(acc_b)
        for i in range(n):
            acc_w[i] = quat_rotate(qs_use[i], acc_b[i])
        return acc_w

    # Option A: use q directly
    acc_w_A = rotate_with(qs)

    # Option B: use conjugate(q)
    qs_conj = np.array([quat_conj(q) for q in qs])
    acc_w_B = rotate_with(qs_conj)

    # Evaluate which is "more gravity-like" during stance
    # We don't assume z sign, so compare |median z| to 9.81
    def score(acc_w):
        z_med = float(np.median(acc_w[zupt_mask, 2]))
        z_abs_err = abs(abs(z_med) - 9.81)
        # also check total magnitude is sane
        mag_med = float(np.median(np.linalg.norm(acc_w[zupt_mask], axis=1)))
        mag_err = abs(mag_med - 9.81)
        return z_abs_err + 0.3 * mag_err, z_med, mag_med

    score_A, zA, mA = score(acc_w_A)
    score_B, zB, mB = score(acc_w_B)

    print("\n[Phase 1.5] Quaternion direction check:")
    print(f"  Option A (use q):     stance median acc_w_z={zA:.3f}, |acc|={mA:.3f}, score={score_A:.3f}")
    print(f"  Option B (use q*):    stance median acc_w_z={zB:.3f}, |acc|={mB:.3f}, score={score_B:.3f}")

    if score_B < score_A:
        print("  -> Using conjugate(q) for body->world rotation (fix applied).")
        acc_w = acc_w_B
        quat_mode = "conj(q)"
    else:
        print("  -> Using q for body->world rotation.")
        acc_w = acc_w_A
        quat_mode = "q"

    return acc_w, quat_mode

def phase_1_5_remove_gravity(acc_w, zupt_mask):
    """
    Remove gravity using a stance-estimated gravity vector.

    Instead of assuming perfect world axes and subtracting [0,0,9.81], we estimate
    the gravity vector as the median world-acceleration during stance/ZUPT samples:
        g_hat = median(acc_w[stance])
    and compute specific acceleration as:
        specific_acc_w = acc_w - g_hat

    This substantially reduces gravity leakage when attitude has small tilt error.
    """
    if zupt_mask is None or (not np.any(zupt_mask)):
        # Fallback: use global median if stance mask is unavailable
        g_hat = np.median(acc_w, axis=0)
        ref = "all"
    else:
        g_hat = np.median(acc_w[zupt_mask], axis=0)
        ref = "stance"

    g_hat = np.asarray(g_hat, dtype=float).reshape(3,)
    specific_acc_w = acc_w - g_hat
    g_mode = f"acc_w - g_hat(median {ref})"

    print(f"[Phase 1.5] Gravity removal mode: {g_mode} | g_hat={g_hat} | |g_hat|={np.linalg.norm(g_hat):.3f}")
    return specific_acc_w, g_mode

# full 3D
def phase_1_6_integrate_with_zupt_per_stride(specific_acc_w, events_imu, fs_hz, half_window=10):
    dt = 1.0 / fs_hz
    n = len(specific_acc_w)

    v = np.zeros((n, 3), dtype=float)
    p = np.zeros((n, 3), dtype=float)

    ev = events_imu.dropna(subset=["min_vel"]).copy()
    mvs = sorted([int(r["min_vel"]) for _, r in ev.iterrows() if 0 <= int(r["min_vel"]) < n])

    # build ZUPT windows
    zupt_windows = []
    for mv in mvs:
        a = max(0, mv - half_window)
        b = min(n, mv + half_window + 1)
        zupt_windows.append((a, b, mv))

    # Integrate + drift-correct between successive ZUPT windows
    # We treat each interval [end_of_prev_zupt, start_of_next_zupt] as "free integration"
    # and enforce v = 0 at both ends using a linear ramp.
    last_end = 0
    # set v to 0 in first ZUPT window explicitly
    if zupt_windows:
        a0, b0, mv0 = zupt_windows[0]
        last_end = b0

    for k in range(len(zupt_windows)-1):
        a_next, b_next, mv_next = zupt_windows[k+1]
        seg_start = last_end
        seg_end   = a_next

        if seg_end <= seg_start + 5:
            last_end = b_next
            continue

        # 1) integrate accel->vel over the segment
        for i in range(seg_start+1, seg_end):
            v[i] = v[i-1] + specific_acc_w[i] * dt

        # 2) enforce end velocity to 0 by subtracting linear drift ramp
        v_end = v[seg_end-1].copy()
        L = (seg_end-1) - seg_start
        for i in range(seg_start, seg_end):
            frac = (i - seg_start) / max(L, 1)
            v[i] -= frac * v_end

        # 3) enforce ZUPT window itself to zero
        v[a_next:b_next] = 0.0
        last_end = b_next

    # integrate velocity -> position
    for i in range(1, n):
        p[i] = p[i-1] + v[i] * dt

    # optional: vertical baseline per stride using ic->end (keep for now)
    ev2 = events_imu.dropna(subset=["ic","end"]).copy()
    z = p[:, 2].copy()
    for _, row in ev2.iterrows():
        ic  = int(row["ic"])
        end = int(row["end"])
        if 0 <= ic < end <= n and end > ic + 10:
            z[ic:end] -= np.median(z[ic:end])
    p[:, 2] = z

    return p, v

#vertical only
def phase_1_6_integrate_z_only_with_zupt(specific_acc_w, events_imu, fs_hz, half_window=10):
    dt = 1.0 / fs_hz
    n = len(specific_acc_w)

    vz = np.zeros(n, dtype=float)
    pz = np.zeros(n, dtype=float)

    ev = events_imu.dropna(subset=["min_vel"]).copy()
    mvs = sorted([int(r["min_vel"]) for _, r in ev.iterrows() if 0 <= int(r["min_vel"]) < n])

    zupt_windows = []
    for mv in mvs:
        a = max(0, mv - half_window)
        b = min(n, mv + half_window + 1)
        zupt_windows.append((a, b))

    # zero first window
    last_end = zupt_windows[0][1] if zupt_windows else 0

    for k in range(len(zupt_windows)-1):
        a_next, b_next = zupt_windows[k+1]
        seg_start = last_end
        seg_end   = a_next
        if seg_end <= seg_start + 5:
            last_end = b_next
            continue

        # integrate az -> vz
        for i in range(seg_start+1, seg_end):
            vz[i] = vz[i-1] + specific_acc_w[i,2] * dt

        # ramp to make vz end at 0
        v_end = vz[seg_end-1]
        L = (seg_end-1) - seg_start
        for i in range(seg_start, seg_end):
            frac = (i - seg_start) / max(L, 1)
            vz[i] -= frac * v_end

        # enforce ZUPT window
        vz[a_next:b_next] = 0.0
        last_end = b_next

    # integrate vz -> pz
    for i in range(1, n):
        pz[i] = pz[i-1] + vz[i] * dt

    return pz, vz

def phase_1_6_integrate_z_only_per_stride_tc_to_ic(
    specific_acc_w,
    events_imu,
    fs_hz,
    min_swing_samples=6,
    enforce_end_vel_zero=True,
    enforce_end_pos_zero=True,
):
    """
    Per-stride swing-only integration (TC -> IC) with closure constraints.

    Resets per stride:
      vz(tc)=0, pz(tc)=0
    Integrates only in swing [tc, ic).
    Optionally enforces:
      vz(ic-) = 0  (velocity closure)
      pz(ic-) = 0  (position closure)  <-- this is what you need now

    Returns:
      pz (n,), vz (n,)
    """
    dt = 1.0 / fs_hz
    n = len(specific_acc_w)
    pz = np.zeros(n, dtype=float)
    vz = np.zeros(n, dtype=float)

    ev = events_imu.dropna(subset=["tc", "ic"]).copy().sort_values("tc")
    last_end = -1

    for s_id, row in ev.iterrows():
        tc = int(row["tc"])
        ic = int(row["ic"])
        if not (0 <= tc < ic <= n):
            continue
        if (ic - tc) < min_swing_samples:
            continue
        if tc <= last_end:
            continue

        # Reset at TC
        vz[tc] = 0.0
        pz[tc] = 0.0

        # 1) integrate acceleration -> velocity in swing
        for i in range(tc + 1, ic):
            vz[i] = vz[i - 1] + specific_acc_w[i, 2] * dt

        # 2) enforce vz(ic-) = 0 by removing a linear drift ramp
        if enforce_end_vel_zero:
            v_end = vz[ic - 1]
            L = (ic - 1) - tc
            if L > 0:
                for i in range(tc, ic):
                    frac = (i - tc) / float(L)
                    vz[i] -= frac * v_end

        # 3) integrate velocity -> position in swing
        for i in range(tc + 1, ic):
            pz[i] = pz[i - 1] + vz[i] * dt

        # 4) enforce pz(ic-) = 0 by removing a linear position drift ramp
        #    (keeps pz(tc)=0 and makes the swing land back at 0)
        if enforce_end_pos_zero:
            p_end = pz[ic - 1]
            L = (ic - 1) - tc
            if L > 0:
                for i in range(tc, ic):
                    frac = (i - tc) / float(L)
                    pz[i] -= frac * p_end

        last_end = ic - 1

    return pz, vz

def phase_1_6_integrate_z_only_per_stride_anchor_min_vel(
    specific_acc_w,
    events_imu,
    fs_hz,
    half_window=10,
    min_samples=6,
    enforce_end_vel_zero=True,
    shift_to_tc=True,
):
    """
    Per-stride swing-only vertical integration, anchored at min_vel (ZUPT-ish).

    For each stride:
      - Use min_vel as a ZUPT anchor: vz(mv)=0, pz(mv)=0
      - Integrate mv->tc to estimate initial conditions at TC
      - Integrate tc->ic to get swing displacement
      - Optional: shift pz so that pz(tc)=0 (swing translation relative to toe-off)

    Returns:
      pz (n,), vz (n,) with values filled mainly in [tc, ic).
    """
    dt = 1.0 / fs_hz
    n = len(specific_acc_w)
    pz = np.zeros(n, dtype=float)
    vz = np.zeros(n, dtype=float)

    ev = events_imu.dropna(subset=["tc", "ic", "min_vel"]).copy().sort_values("tc")
    last_end = -1

    for s_id, row in ev.iterrows():
        tc = int(row["tc"])
        ic = int(row["ic"])
        mv = int(row["min_vel"])

        if not (0 <= tc < ic <= n):
            continue
        if not (0 <= mv < n):
            continue
        if (ic - tc) < min_samples:
            continue
        if tc <= last_end:
            continue

        # Define a ZUPT anchor window around min_vel (optional but stabilizes)
        a0 = max(0, mv - half_window)
        b0 = min(n, mv + half_window + 1)

        # Anchor at mv: v=0, p=0
        v_m = 0.0
        p_m = 0.0

        # --- Integrate forward from mv to tc to get initial conditions at TC ---
        if tc > mv:
            v = v_m
            p = p_m
            # Keep v=0 inside the ZUPT window
            for i in range(mv + 1, tc + 1):
                if a0 <= i < b0:
                    v = 0.0
                else:
                    v = v + specific_acc_w[i, 2] * dt
                p = p + v * dt
            v_tc = v
            p_tc = p
        else:
            # If tc <= mv (rare with your events), integrate backward mv->tc
            v = v_m
            p = p_m
            for i in range(mv, tc, -1):
                if a0 <= i < b0:
                    v = 0.0
                else:
                    v = v - specific_acc_w[i, 2] * dt
                p = p - v * dt
            v_tc = v
            p_tc = p

        # --- Integrate swing tc -> ic using those initial conditions ---
        v = v_tc
        p = p_tc
        vz[tc] = v
        pz[tc] = p

        for i in range(tc + 1, ic):
            v = v + specific_acc_w[i, 2] * dt
            p = p + v * dt
            vz[i] = v
            pz[i] = p

        # Optional: enforce vz(ic-) ~ 0 by ramp removal across the SWING only
        if enforce_end_vel_zero and (ic - tc) > 2:
            v_end = vz[ic - 1]
            L = (ic - 1) - tc
            for i in range(tc, ic):
                frac = (i - tc) / float(L)
                vz[i] -= frac * v_end
            # recompute pz from tc with corrected vz (keeps same starting pz[tc])
            p = pz[tc]
            for i in range(tc + 1, ic):
                p = p + vz[i] * dt
                pz[i] = p

        # Optional: shift so swing translation is relative to TC (pz(tc)=0)
        if shift_to_tc:
            p0 = pz[tc]
            pz[tc:ic] -= p0

        last_end = ic - 1

    return pz, vz


def phase_1_7_ground_reference_from_stance(toe_z, ic, tc):
    """
    (1.7) Ground height reference:
    - ground z = median toe height during stance (IC → TC)
    """
    if tc <= ic + 5:
        return np.nan
    return float(np.median(toe_z[ic:tc]))

def phase_1_8_compute_mtc_per_stride(p_imu_w, qs, p_toe_b, events_imu, quat_mode="q", ground_mode="min_vel", half_window=10):
    n = len(p_imu_w)

    toe_w = np.zeros((n, 3), dtype=float)
    for i in range(n):
        if quat_mode == "q":
            q_use = qs[i]
        elif quat_mode == "conj(q)":
            q_use = quat_conj(qs[i])
        else:
            raise ValueError("Unknown quat_mode")

        toe_w[i] = p_imu_w[i] + quat_rotate(q_use, p_toe_b)

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

    

        if ground_mode == "min_vel" and (mv is not None):
            gw_a = max(0, mv - half_window)
            gw_b = min(len(toe_z), mv + half_window + 1)
            ground_win = toe_z[gw_a:gw_b]
        else:
            gw_a = ic
            gw_b = end
            ground_win = toe_z[gw_a:gw_b]

        ground = float(np.median(ground_win))

        # quality gate (5 mm)
        if np.std(ground_win) > 0.005:
            continue

        swing_clearance = toe_z[tc:ic] - ground
        mtc = float(np.min(swing_clearance))

        # Store per-stride debug (everything needed for a single-stride plot)
        stride_debug[s_id] = {
            "start": start, "end": end,
            "tc": tc, "ic": ic,
            "min_vel": mv,
            "ground_z": ground,
            "ground_win_a": gw_a,
            "ground_win_b": gw_b,
        }


        rows.append({
            "s_id": s_id,
            "tc": tc,
            "ic": ic,
            "start": start,
            "end": end,
            "ground_z": ground,
            "mtc_imu_m": mtc,
            "ground_std_m": float(np.std(ground_win))
        })

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.set_index("s_id")

    debug = {"toe_w": toe_w, "toe_z": toe_z, "stride_debug": stride_debug}
    return res, debug



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

        swing_clearance = toe_z[tc:ic] - ground
        mtc = float(np.min(swing_clearance))

        gt_rows.append({"s_id": s_id, "mtc_mocap_m": mtc})

    mocap_stride_res = pd.DataFrame(gt_rows)
    if mocap_stride_res.empty:
        raise RuntimeError("MoCap GT stride list is empty — check toe_marker_name and event windows.")

    mocap_stride_res = mocap_stride_res.set_index("s_id")

    joined = imu_stride_res.join(mocap_stride_res, how="inner").dropna()
    if joined.empty:
        raise RuntimeError("No overlapping strides between IMU and MoCap results after join.")

    err = joined["mtc_imu_m"] - joined["mtc_mocap_m"]
    bias = float(err.mean())
    rmse = float(np.sqrt(np.mean(err.to_numpy() ** 2)))

    # Additional reporting metrics (useful for calibration logs and terminal summaries)
    mtc_imu_mean_m = float(joined["mtc_imu_m"].mean())
    mtc_mocap_mean_m = float(joined["mtc_mocap_m"].mean())
    # Signed mean difference between the *means* (should match bias when computed over the same joined set)
    mtc_avg_diff_m = float(mtc_imu_mean_m - mtc_mocap_mean_m)
    mtc_mean_abs_diff_m = float(np.mean(np.abs(err.to_numpy())))
    extra = {
        "mtc_imu_mean_m": round(mtc_imu_mean_m, 6),
        "mtc_mocap_mean_m": round(mtc_mocap_mean_m, 6),
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

    if not quiet:
        # ---- Debug: Top absolute errors ----
        print("\n[Phase 1.9] Top 5 absolute errors (by stride id):")
        print(top5_mm.round(2).to_string())

        joined_dbg.to_csv("stride_errors.csv")
        print("[Phase 1.9] Saved stride_errors.csv")

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


# ============================================================
# main(): calls Phase 1.1 → 1.9 chronologically
# ============================================================


def run_pipeline_for_row(
    data_folder: str,
    participant: str,
    test: str,
    side: str,
    toe_x_cm: float,
    toe_z_cm: float,
    sensor: str | None = None,
    toe_marker: str | None = None,
    padding_s: float = 3.0,
    quiet: bool = True,
):
    """Run Phase 1.1–1.9 once for a given participant/test/side using a fixed toe offset.
    Returns a dict with MTC mean/diff metrics (mm) computed on the joined stride set.

    This is intended for augmenting an existing toe_offset_calibration.csv without running the
    expensive x/z grid search.
    """
    # Defaults
    if sensor is None:
        sensor = "l_instep" if side == "left" else "r_instep"
    if toe_marker is None:
        toe_marker = "l_toe" if side == "left" else "r_toe"

    # Load dataset (MoCap version)
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

    if sensor not in imu_all.columns.get_level_values(0):
        raise RuntimeError(f"Sensor '{sensor}' not found for {participant}/{test}.")
    if toe_marker not in mocap_traj.columns.get_level_values(0):
        raise RuntimeError(f"Toe marker '{toe_marker}' not found for {participant}/{test}.")

    imu_df = imu_all[sensor][["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].copy()
    events_mocap = datapoint.mocap_events_[side]
    events_imu = datapoint.convert_events_with_padding(events_mocap, from_time_axis="mocap", to_time_axis="imu")

    # Phase 1.2: fixed toe offset (cm -> m)
    p_toe_b = np.array([toe_x_cm / 100.0, 0.0, toe_z_cm / 100.0], dtype=float)

    # Phase 1.3
    imu_f = phase_1_3_filter_imu_signals(imu_df, fs_hz, cutoff_hz=20.0)

    # Phase 1.4: ZUPT mask (min_vel ± 10)
    zupt_mask = phase_1_4_get_zupt_mask_from_min_vel(events_imu, n_samples=len(imu_f), half_window=10)

    # Phase 1.5: orientation + world acc
    qs = phase_1_5_estimate_orientation(imu_f, fs_hz, zupt_mask=zupt_mask)
    acc_w, quat_mode = phase_1_5_compute_world_acceleration_with_quat_fix(imu_f, qs, zupt_mask)
    specific_acc_w, _g_mode = phase_1_5_remove_gravity(acc_w, zupt_mask)

    # Phase 1.6: integrate z-only per stride
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

    # Phase 1.8: per-stride MTC from IMU
    imu_stride_res, debug = phase_1_8_compute_mtc_per_stride(
        p_imu_w, qs, p_toe_b, events_imu, quat_mode=quat_mode,
        ground_mode="min_vel", half_window=10
    )

    # Phase 1.9: validate against mocap -> returns extra in meters
    joined, bias, rmse, extra = phase_1_9_validate_against_mocap(
        mocap_traj=mocap_traj,
        events_mocap=events_mocap,
        toe_marker_name=toe_marker,
        imu_stride_res=imu_stride_res,
        plot=False,
        fs_hz=fs_hz,
        debug=debug,
        verbose=(not quiet),
        quiet=quiet,
        events_imu=events_imu,
        plot_worst_k=0,
    )

    # Convert to mm for CSV
    res = {
        "bias_mm": float(bias * 1000.0),
        "rmse_mm": float(rmse * 1000.0),
        "n_joined": int(len(joined)),
        "mtc_imu_mean_mm": float(extra.get("mtc_imu_mean_m", np.nan) * 1000.0),
        "mtc_mocap_mean_mm": float(extra.get("mtc_mocap_mean_m", np.nan) * 1000.0),
        "mtc_avg_diff_mm": float(extra.get("mtc_avg_diff_m", bias) * 1000.0),
        "mtc_mean_abs_diff_mm": float(extra.get("mtc_mean_abs_diff_m", np.nan) * 1000.0),
    }
    return res


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
            res = run_pipeline_for_row(
                data_folder=data_folder,
                participant=participant,
                test=test,
                side=side,
                toe_x_cm=toe_x_cm,
                toe_z_cm=toe_z_cm,
                padding_s=padding_s,
                quiet=True,
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

    args = parser.parse_args()

    # Defaults
    if args.sensor is None:
        args.sensor = "l_instep" if args.side == "left" else "r_instep"
    if args.toe_marker is None:
        args.toe_marker = "l_toe" if args.side == "left" else "r_toe"

    # Optional: augment existing toe_offset_calibration.csv with MTC mean/diff columns (no calibration sweep)
    if getattr(args, "augment_calib_csv_with_mtc", False):
        augment_existing_toe_offset_csv_with_mtc(
            csv_path=str(Path(args.toe_offset_csv)),
            data_folder=args.data_folder,
            padding_s=args.padding_s,
        )
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
    events_imu = datapoint.convert_events_with_padding(events_mocap, from_time_axis="mocap", to_time_axis="imu")
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
    imu_f = phase_1_3_filter_imu_signals(imu_df, fs_hz, cutoff_hz=20.0)
    acc_norm = np.median(np.linalg.norm(imu_f[['acc_x','acc_y','acc_z']].to_numpy(), axis=1))
    print_phase("Phase 1.3", f"Filtered acc_norm median(all)={acc_norm:.3f} m/s^2")


    # ------------------------
    # Phase 1.4
    # ------------------------
    zupt_mask = phase_1_4_get_zupt_mask_from_min_vel(events_imu, n_samples=len(imu_f), half_window=10)
    # debug_units_and_magnitude(imu_f, zupt_mask)
    debug_gyro_units(imu_f, zupt_mask)
    zupt_pct = 100.0 * float(np.mean(zupt_mask))
    n_stride_rows = len(events_imu.dropna(subset=["min_vel"]))
    print_phase("Phase 1.4", f"ZUPT samples (min_vel±10): {zupt_pct:.1f}% | strides_with_min_vel={n_stride_rows}")

    # ------------------------
    # Phase 1.5
    # ------------------------
    qs = phase_1_5_estimate_orientation(imu_f, fs_hz, zupt_mask=zupt_mask)
    print_phase("Phase 1.5", f"Quaternion array shape: {qs.shape} (w,x,y,z)")

    acc_w, quat_mode = phase_1_5_compute_world_acceleration_with_quat_fix(imu_f, qs, zupt_mask)
    specific_acc_w, g_mode = phase_1_5_remove_gravity(acc_w, zupt_mask)

    print_phase("Phase 1.5", f"Rotation mode: {quat_mode} | Gravity mode: {g_mode}")
    print_phase("Phase 1.5", f"acc_w stance median z: {np.median(acc_w[zupt_mask,2]):.3f} m/s^2 | "
                            f"|acc_w| stance median: {np.median(np.linalg.norm(acc_w[zupt_mask],axis=1)):.3f}")

    # ------------------------
    # Phase 1.6
    # ------------------------
    pz, vz = phase_1_6_integrate_z_only_per_stride_anchor_min_vel(
        specific_acc_w,
        events_imu,
        fs_hz,
        half_window=10,
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

            best = calibrate_toe_offset_xz(
                p_imu_w=p_imu_w,
                qs=qs,
                events_imu=events_imu,
                mocap_traj=mocap_traj,
                events_mocap=events_mocap,
                toe_marker_name=args.toe_marker,
                quat_mode=quat_mode,
                ground_mode="min_vel",
                half_window=10,
                x_min_cm=4.0, x_max_cm=24.0, x_step_cm=0.5,
                z_min_cm=-12.0, z_max_cm=-0.0, z_step_cm=0.2,
                objective="rmse"
            )

            print(
                f"[Calibration] Best toe (x,z)=({best['x_cm']:.2f} cm, {best['z_cm']:.2f} cm) "
                f"| Bias={best['bias']*1000:.1f} mm | RMSE={best['rmse']*1000:.1f} mm"
            )

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
        p_imu_w, qs, p_toe_b, events_imu, quat_mode=quat_mode,
        ground_mode="min_vel", half_window=10
    )

    print("imu_stride_res.columns:", list(imu_stride_res.columns))
    print("imu_stride_res.head():\n", imu_stride_res.head())


    # ---- Phase 1.8 prints (robust) ----
    n_total = len(events_imu.dropna(subset=["ic","tc","start","end"]))

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
    # Phase 1.9
    # ------------------------
    joined, bias, rmse, extra = phase_1_9_validate_against_mocap(
        mocap_traj=mocap_traj,
        events_mocap=events_mocap,
        toe_marker_name=args.toe_marker,
        imu_stride_res=imu_stride_res,
        plot=args.plot,
        fs_hz=fs_hz,
        debug=debug,
        verbose=False,
        events_imu=events_imu,
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
    print("\n=== Run Summary ===")
    print(f"Participant: {args.participant} | Test: {args.test} | Side: {args.side}")
    print(f"IMU sensor: {args.sensor} | Toe marker: {args.toe_marker}")
    print(f"Toe offset (m): {p_toe_b.tolist()}")
    print(f"Phase objective: {phase_cfg}")
    print("Done.")


if __name__ == "__main__":
    main()
