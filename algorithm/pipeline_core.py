
"""pipeline_core.py

Clean orchestration layer for the ENG4701 IMU→MTC pipeline.

This module provides a tidy, testable API while reusing the existing
phase_* implementations from the legacy script:
    mtc_pipeline.py

Philosophy
- Keep the heavy math in the legacy module for now.
- Provide clean orchestration + separation of concerns (calibration vs evaluation).
- Minimize runtime by precomputing up to Phase 1.6 once, then looping only Phase 1.8+1.9.

Typical usage
- Calibration: see calibrate_toe_offset.py
- Evaluation:  see evaluate.py

NOTE
This file assumes the legacy module is importable (same folder / PYTHONPATH).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict, List, Union


import numpy as np
import pandas as pd
import os
import json
import contextlib, os, sys

@contextlib.contextmanager
def suppress_stdout():
    save = sys.stdout
    try:
        with open(os.devnull, "w") as f:
            sys.stdout = f
            yield
    finally:
        sys.stdout = save

# Legacy implementation (your existing, messy script)
import mtc_pipeline as legacy


# -----------------------------
# Data containers
# -----------------------------

@dataclass(frozen=True)
class TrialKey:
    participant: str
    test: str
    side: str  # "left" | "right"


@dataclass
class TrialData:
    key: TrialKey
    fs_hz: float
    sensor: str
    toe_marker: str
    imu_df: pd.DataFrame
    mocap_traj: pd.DataFrame
    events_mocap: pd.DataFrame
    events_gt_imu: pd.DataFrame
    meta: Dict[str, Any]



@dataclass
class Precomputed:
    """Outputs of precompute() that do not depend on toe offset."""

    trial: TrialData
    imu_f: pd.DataFrame
    events_imu: pd.DataFrame
    zupt_mask: np.ndarray
    qs: np.ndarray
    p_imu_w: np.ndarray
    half_window_samples: int
    mapping_df: pd.DataFrame


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class PipelineToggles:
    dynamic_cutoff: bool = True
    dynamic_zupt: bool = True
    dynamic_ic_refine: bool = True
    dynamic_gate: bool = True


@dataclass(frozen=True)
class CalibrationSearch:

    # Hard clamps (match the box)
    x_min_cm: float = 8.0
    x_max_cm: float = 22.0
    z_min_cm: float = -4.0
    z_max_cm: float = 4.0


    # Coarse grid (x only style range)
    x_coarse_cm: Tuple[float, float, float] = (x_min_cm, x_max_cm, 0.5)   # start, stop, step
    z_coarse_cm: Tuple[float, float, float] = (z_min_cm, z_max_cm, 0.5)     # force z=0 in coarse

    # Fine window around best coarse
    x_fine_halfwidth_cm: float = 1.5
    z_fine_halfwidth_cm: float = 0.8
    x_fine_step_cm: float = 0.1
    z_fine_step_cm: float = 0.1  # unused if z_halfwidth=0

# -----------------------------
# Loading
# -----------------------------

def load_trial(
    *,
    data_folder: str,
    participant: str,
    test: str,
    side: str,
    padding_s: float = 3.0,
    sensor: Optional[str] = None,
    toe_marker: Optional[str] = None,
) -> TrialData:
    """Load a single (participant, test, side) trial from SensorPositionComparison2019Mocap."""

    side_l = str(side).lower()
    if sensor is None:
        sensor = "l_instep" if side_l == "left" else "r_instep"
    if toe_marker is None:
        toe_marker = "l_toe" if side_l == "left" else "r_toe"

    dataset = legacy.SensorPositionComparison2019Mocap(
        memory=legacy.Memory("./cache"),
        data_folder=data_folder,
        data_padding_s=float(padding_s),
    )

    # Load participant meta_data.json (e.g., ...\data\4d91\meta_data.json)
    meta_path = os.path.join(data_folder, "data", str(participant), "meta_data.json")
    meta: Dict[str, Any] = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}


    subset = dataset.get_subset(participant=[participant], test=[test])
    if len(subset) == 0:
        raise RuntimeError(f"No datapoint found for participant={participant}, test={test}")

    datapoint = subset[0]
    fs_hz = float(datapoint.sampling_rate_hz)

    imu_all = datapoint.data
    mocap_traj = datapoint.marker_position_

    if sensor not in imu_all.columns.get_level_values(0):
        raise RuntimeError(f"Sensor '{sensor}' not found for {participant}/{test}.")
    if toe_marker not in mocap_traj.columns.get_level_values(0):
        raise RuntimeError(f"Toe marker '{toe_marker}' not found for {participant}/{test}.")

    imu_df = imu_all[sensor][["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].copy()

    events_mocap = datapoint.mocap_events_[side_l]
    events_gt_imu = datapoint.convert_events_with_padding(
        events_mocap, from_time_axis="mocap", to_time_axis="imu"
    )

    # Ensure contiguous stride ids (prevents batch index weirdness)
    events_mocap = events_mocap.reset_index(drop=True).copy()
    events_gt_imu = events_gt_imu.reset_index(drop=True).copy()

    return TrialData(
        key=TrialKey(participant=str(participant), test=str(test), side=side_l),
        fs_hz=fs_hz,
        sensor=sensor,
        toe_marker=toe_marker,
        imu_df=imu_df,
        mocap_traj=mocap_traj,
        events_mocap=events_mocap,
        events_gt_imu=events_gt_imu,
        meta=meta
    )


# -----------------------------
# Precompute (independent of toe offset)
# -----------------------------

def precompute(
    trial: TrialData,
    *,
    event_source: str = "imu",
    eventnet_model=None,
    eventnet_meta: Optional[dict] = None,
    toggles: PipelineToggles = PipelineToggles(),
    gate_s: float = 0.15,
) -> Precomputed:
    """Run the pipeline up to Phase 1.6, and also compute stride mapping once."""

    fs_hz = float(trial.fs_hz)

    # Phase 1.3 filter
    imu_f = legacy.phase_1_3_filter_imu(trial.imu_df, fs_hz, dynamic=toggles.dynamic_cutoff)

    # Phase 1.4 events
    src = (event_source or "imu").lower().strip()
    if src == "ml":
        if eventnet_model is None:
            raise RuntimeError("event_source=ml but eventnet_model is None")
        events_imu, evdbg = legacy.phase_1_4_detect_events_ml(
            imu_f=imu_f,
            fs_hz=fs_hz,
            model=eventnet_model,
            meta=eventnet_meta or {},
        )
    else:
        events_imu, evdbg = legacy.phase_1_4_detect_events_imu_only(imu_f, fs_hz)

    events_imu = legacy.phase_1_4b_filter_invalid_strides(events_imu, fs_hz)
    if events_imu is None or len(events_imu) == 0:
        raise RuntimeError("IMU-only event detection produced no valid strides after filtering")

    # ZUPT mask
    zupt_mask = legacy.phase_1_4_get_zupt_mask_from_min_vel(
        events_imu,
        n_samples=len(imu_f),
        dynamic=toggles.dynamic_zupt,
        fs_hz=fs_hz,
        frac=0.03,
        min_hw=6,
        max_hw=25,
        stride_ref="ic2ic",
    )

    # half-window heuristic (same as your current behavior)
    half_window_samples = 10
    if toggles.dynamic_zupt:
        ref = trial.events_gt_imu if (trial.events_gt_imu is not None and len(trial.events_gt_imu) > 3) else events_imu
        stride = ref.dropna(subset=["ic", "tc"]).sort_values("ic")
        ic = stride["ic"].to_numpy(dtype=int)
        if len(ic) >= 3:
            stride_s = float(np.median(np.diff(ic) / fs_hz))
            hw_s = 0.03 * stride_s
            half_window_samples = int(np.clip(round(hw_s * fs_hz), 6, 25))

    # Phase 1.5 orientation
    qs = legacy.phase_1_5_estimate_orientation(imu_f, fs_hz, zupt_mask=zupt_mask)

    # Optional stance attitude correction
    qs = legacy.phase_1_5b_stance_attitude_correction(
        qs, imu_f, events_imu, fs_hz,
        half_window=half_window_samples,
        alpha="dynamic",
        use_median=True,
    )

    # World accel + gravity removal
    acc_w = legacy.phase_1_5_compute_world_acceleration(imu_f, qs)
    specific_acc_w, _ = legacy.phase_1_5_remove_gravity(acc_w, zupt_mask)

    # IC refinement
    if toggles.dynamic_ic_refine:
        events_imu = legacy.refine_ic_with_specific_acc(
            events_imu,
            specific_acc_w=specific_acc_w,
            fs_hz=fs_hz,
            dynamic=True,
            pre_frac=0.12,
            post_frac=0.03,
            stride_ref="ic2ic",
        )

    # Phase 1.6 z-only integration
    pz, vz = legacy.phase_1_6_integrate_z_only_per_stride_anchor_min_vel(
        specific_acc_w,
        events_imu,
        fs_hz,
        half_window=half_window_samples,
        enforce_end_vel_zero=True,
        shift_to_tc=True,
    )
    p_imu_w = np.zeros((len(pz), 3), dtype=float)
    p_imu_w[:, 2] = pz

    print("[DBG] p_imu_w xy max abs:", np.max(np.abs(p_imu_w[:, :2])))

    # Stride mapping ONCE (independent of toe offset)
    mapping_df, match_sum = legacy.match_strides_by_mid_swing_dp(
        events_imu, trial.events_gt_imu, fs_hz, gate_s=float(gate_s)
    )
    if mapping_df is None or mapping_df.empty:
        raise RuntimeError(
            f"No stride matches for calibration/eval (det={match_sum.get('n_det')} gt={match_sum.get('n_gt')})."
        )

    return Precomputed(
        trial=trial,
        imu_f=imu_f,
        events_imu=events_imu,
        zupt_mask=zupt_mask,
        qs=np.asarray(qs, dtype=float),
        p_imu_w=np.asarray(p_imu_w, dtype=float),
        half_window_samples=int(half_window_samples),
        mapping_df=mapping_df,
    )


# -----------------------------
# Toe offset evaluation + scoring
# -----------------------------

def compute_imu_mtc_for_offset(
    pre: Precomputed,
    *,
    toe_x_cm: float,
    toe_z_cm: float,
    ground_mode: str = "min_vel",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run Phase 1.8 for a given (x,z) in cm and return stride-level results."""

    p_toe_b = np.array([toe_x_cm / 100.0, 0.0, toe_z_cm / 100.0], dtype=float)
    

    imu_stride_res, debug = legacy.phase_1_8_compute_mtc_per_stride(
        pre.p_imu_w, pre.qs, p_toe_b, pre.events_imu,
        ground_mode=ground_mode,
        half_window=pre.half_window_samples,
    )
    return imu_stride_res, (debug or {})

def compute_imu_mtc_for_offset_y(
    pre: Precomputed,
    *,
    toe_y_cm: float,
    toe_z_cm: float = 0.0,
    ground_mode: str = "min_vel",
):
    p_toe_b = np.array([0.0, toe_y_cm / 100.0, toe_z_cm / 100.0], dtype=float)

    imu_stride_res, debug = legacy.phase_1_8_compute_mtc_per_stride(
        pre.p_imu_w, pre.qs, p_toe_b, pre.events_imu,
        ground_mode=ground_mode,
        half_window=pre.half_window_samples,
    )
    return imu_stride_res, (debug or {})


def score_offset_against_mocap(
    pre: Precomputed,
    imu_stride_res: pd.DataFrame,
    *,
    ground_mode: str = "min_vel",
) -> Optional[Dict[str, Any]]:
    """Apply stride mapping + compute bias/RMSE vs MoCap (Phase 1.9)."""

    imu_scored = legacy.apply_stride_mapping_to_imu_results(
        imu_stride_res,
        pre.events_imu,
        pre.mapping_df,
        pre.trial.events_gt_imu,
    )
    if imu_scored is None or imu_scored.empty:
        return None

    with suppress_stdout():
        joined, bias, rmse, extra = legacy.phase_1_9_validate_against_mocap(
            mocap_traj=pre.trial.mocap_traj,
            events_mocap=pre.trial.events_mocap,
            toe_marker_name=pre.trial.toe_marker,
            imu_stride_res=imu_scored,
            plot=False,
            quiet=True,
            verbose=False,
            ground_mode=ground_mode,
            half_window=pre.half_window_samples,
            events_imu=None,
        )

    if bias is None or rmse is None:
        return None
    if not np.isfinite(bias) or not np.isfinite(rmse):
        return None

    out = {
        "bias_m": float(bias),
        "rmse_m": float(rmse),
        "n": int(len(joined)) if joined is not None else 0,
    }
    if extra:
        out.update(extra)
    return out


# -----------------------------
# Fast calibration
# -----------------------------
def calibrate_toe_offset_coarse_to_fine(
    pre: Union[Precomputed, List[Precomputed]],
    *,
    objective: str = "rmse",
    search: CalibrationSearch = CalibrationSearch(),
    ground_mode: str = "min_vel",
) -> Dict[str, Any]:
    """
    (x,z) calibration WITHOUT touching legacy:
    - uses legacy compute_imu_mtc_for_offset + score_offset_against_mocap
    - adds *weak* priors (MAP) and robust scoring to avoid degeneracy
    - allows z≈0 if the data doesn’t support non-zero z
    """

    objective = (objective or "rmse").lower().strip()

    # Accept either a single Precomputed or a list of them
    pres = pre if isinstance(pre, (list, tuple)) else [pre]
    pre0 = pres[0]  # for priors / metadata

    # -------------------------
    # Metadata-derived x prior only (weak)
    # -------------------------
    shoe_size = None
    try:
        shoe_size = pre0.trial.meta.get("shoe_size", None)
    except Exception:
        shoe_size = None

    if shoe_size is None:
        foot_len_cm = 27.0
    else:
        foot_len_cm = 26.5 + 0.5 * ((float(shoe_size) - 40.0) / 2.0)

    # --- Regularization (normalized) ---
    # Prior means (cm)
    x0_cm = float(np.clip(0.35 * foot_len_cm, search.x_min_cm, search.x_max_cm))
    z0_cm = 0.0

    # Prior scales (cm) ~ "1 sigma"
    sig_x_cm = 2.5   # 25 mm: allows variation, prevents x=0 cheating
    sig_z_cm = 1.5   # 15 mm

    # Regularization weights
    lam_x  = 0.0
    lam_z  = 0.10
    lam_z0 = 0.05    # extra soft pull toward z≈0 (tie-breaker)

    def reg_penalty(x_cm: float, z_cm: float) -> float:
        dx = (x_cm - x0_cm) / sig_x_cm
        dz = (z_cm - z0_cm) / sig_z_cm
        dz0 = (z_cm - 0.0) / sig_z_cm
        return float(lam_x*dx*dx + lam_z*dz*dz + lam_z0*dz0*dz0)


    def clamp_ranges(x_cm: float, z_cm: float) -> tuple[float, float]:
        return (
            float(np.clip(x_cm, search.x_min_cm, search.x_max_cm)),
            float(np.clip(z_cm, search.z_min_cm, search.z_max_cm)),
        )

    # Coarse grid from CalibrationSearch
    x_start, x_stop, x_step = search.x_coarse_cm
    z_start, z_stop, z_step = search.z_coarse_cm

    def base_score_from_metrics(
        m: Dict[str, Any],
        joined: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Returns the per-test data term in meters.

        Modes:
            - "rmse"     → robust trimmed RMSE (preferred)
            - "abs_bias" → |bias|
            - "hybrid"   → robust_RMSE + 0.5*|bias|

        Uses joined if available for robust trimming.
        """

        bias_m = float(m.get("bias_m", 0.0))
        rmse_m = float(m.get("rmse_m", 0.0))

        # -------------------------
        # Robust trimmed RMSE helper
        # -------------------------
        def robust_rmse_from_joined(joined_df: pd.DataFrame) -> float:
            e = (joined_df["mtc_imu_m"] - joined_df["mtc_mocap_m"]).to_numpy(dtype=float)
            if e.size == 0:
                return float(rmse_m)

            if e.size < 5:
                return float(np.sqrt(np.mean(e * e)))

            abs_e = np.abs(e)

            # Drop worst 10% (min 1, max 3)
            k = int(np.clip(np.floor(0.10 * e.size), 1, 3))
            keep = np.argsort(abs_e)[: max(1, e.size - k)]
            e2 = e[keep]

            return float(np.sqrt(np.mean(e2 * e2)))

        # -------------------------
        # Objective modes
        # -------------------------
        if objective == "abs_bias":
            return abs(bias_m)

        if objective == "rmse":
            if joined is not None and not joined.empty:
                return robust_rmse_from_joined(joined)
            return rmse_m

        # Hybrid (default fallback)
        if joined is not None and not joined.empty:
            return robust_rmse_from_joined(joined) + 0.5 * abs(bias_m)

        return rmse_m + 0.5 * abs(bias_m)


    def multi_test_data_term(x_cm: float, z_cm: float) -> tuple[float, float, float, int, list]:
        """
        Returns:
        data_term (equal-weighted across tests),
        bias_mean (for logging),
        rmse_mean (for logging),
        n_total,
        per_test_details
        """
        d_terms = []
        biases = []
        rmses = []
        n_total = 0
        details = []

        for pre_t in pres:
            imu_stride_res, _ = compute_imu_mtc_for_offset(
                pre_t, toe_x_cm=x_cm, toe_z_cm=z_cm, ground_mode=ground_mode
            )

            imu_scored = legacy.apply_stride_mapping_to_imu_results(
                imu_stride_res,
                pre_t.events_imu,
                pre_t.mapping_df,
                pre_t.trial.events_gt_imu,
            )
            if imu_scored is None or imu_scored.empty:
                d_terms.append(0.20)  # 200 mm penalty
                details.append((pre_t.trial.key.test, False, None, None, 0.20))
                continue

            with suppress_stdout():
                joined, bias, rmse, extra = legacy.phase_1_9_validate_against_mocap(
                    mocap_traj=pre_t.trial.mocap_traj,
                    events_mocap=pre_t.trial.events_mocap,
                    toe_marker_name=pre_t.trial.toe_marker,
                    imu_stride_res=imu_scored,
                    plot=False,
                    quiet=True,
                    verbose=False,
                    ground_mode=ground_mode,
                    half_window=pre_t.half_window_samples,
                    events_imu=pre_t.events_imu,
                )


            if joined is None or joined.empty or (not np.isfinite(bias)) or (not np.isfinite(rmse)):
                d_terms.append(0.20)
                details.append((pre_t.trial.key.test, False, None, None, 0.20))
                continue

            m = {"bias_m": float(bias), "rmse_m": float(rmse), "n": int(len(joined))}
            if extra:
                m.update(extra)

            # robust per-test score (your existing helper)
            d_t = float(base_score_from_metrics(m, joined=joined))

            d_terms.append(d_t)
            biases.append(float(bias))
            rmses.append(float(rmse))
            n_total += int(len(joined))
            details.append((pre_t.trial.key.test, True, float(bias), float(rmse), d_t))

        data_term = float(np.mean(d_terms)) if d_terms else 0.20
        bias_mean = float(np.mean(biases)) if biases else 0.0
        rmse_mean = float(np.mean(rmses)) if rmses else data_term
        return data_term, bias_mean, rmse_mean, int(n_total), details


    x_grid = np.arange(x_start, x_stop + 1e-9, x_step)
    z_grid = np.arange(z_start, z_stop + 1e-9, z_step)

    total = len(x_grid) * len(z_grid)
    k = 0

    # Guarantee z=0 is included (important when z range is not exactly aligned)
    if not np.any(np.isclose(z_grid, 0.0, atol=1e-9)):
        z_grid = np.sort(np.unique(np.append(z_grid, 0.0)))

    best = None

    for x_cm in x_grid:
        for z_cm in z_grid:
            x_cm, z_cm = clamp_ranges(x_cm, z_cm)

            data_term, bias_mean, rmse_mean, n_total, details = multi_test_data_term(x_cm, z_cm)
            score = float(data_term + reg_penalty(x_cm, z_cm))

            cand = {
                "toe_x_cm": x_cm,
                "toe_z_cm": z_cm,
                "bias_m": bias_mean,
                "rmse_m": rmse_mean,
                "n": int(n_total),
                "data_term_m": float(data_term),
                "score": score,
                "per_test": details,
            }

            k += 1

            if (k == 1) or (k % 20 == 0) or (k == total):
                print(
                    f"[CAL coarse] {k}/{total}  "
                    f"x={x_cm:.2f}  z={z_cm:.2f}  "
                    f"RMSE={rmse_mean*1000:.2f}mm  "
                    f"Bias={bias_mean*1000:.2f}mm",
                    flush=True
                )

            if best is None or cand["score"] < best["score"]:
                best = cand

    # -------------------------
    # Fine grid around best
    # -------------------------
    cx, cz = float(best["toe_x_cm"]), float(best["toe_z_cm"])
    # Fine grid from CalibrationSearch
    x_fine = np.arange(
        cx - search.x_fine_halfwidth_cm,
        cx + search.x_fine_halfwidth_cm + 1e-9,
        search.x_fine_step_cm,
    )
    z_fine = np.arange(
        cz - search.z_fine_halfwidth_cm,
        cz + search.z_fine_halfwidth_cm + 1e-9,
        search.z_fine_step_cm,
    )

    total_f = len(x_fine) * len(z_fine)
    kf = 0

    # Guarantee z=0 is included
    if not np.any(np.isclose(z_fine, 0.0, atol=1e-9)):
        z_fine = np.sort(np.unique(np.append(z_fine, 0.0)))


    for x_cm in x_fine:
        for z_cm in z_fine:
            x_cm, z_cm = clamp_ranges(x_cm, z_cm)

            data_term, bias_mean, rmse_mean, n_total, details = multi_test_data_term(x_cm, z_cm)
            score = float(data_term + reg_penalty(x_cm, z_cm))

            cand = {
                "toe_x_cm": x_cm,
                "toe_z_cm": z_cm,
                "bias_m": bias_mean,
                "rmse_m": rmse_mean,
                "n": int(n_total),
                "data_term_m": float(data_term),
                "score": score,
                "per_test": details,
            }

            kf += 1
            
            if (kf == 1) or (kf % 50 == 0) or (kf == total_f):
                print(
                    f"[CAL fine] {kf}/{total_f}  "
                    f"x={x_cm:.2f}  z={z_cm:.2f}  "
                    f"RMSE={rmse_mean*1000:.2f}mm  "
                    f"Bias={bias_mean*1000:.2f}mm",
                    flush=True
                )


            if best is None or cand["score"] < best["score"]:
                best = cand

    # Attach priors for logging
    best["shoe_size_prior"] = float(shoe_size) if shoe_size is not None else np.nan
    best["x0_cm_prior"] = float(x0_cm)
    best["z0_cm_prior"] = float(z0_cm)
    best["lam_x"] = float(lam_x)
    best["lam_z"] = float(lam_z)
    best["lam_z0"] = float(lam_z0)
    best["sig_x_cm"] = float(sig_x_cm)
    best["sig_z_cm"] = float(sig_z_cm)


    return best
