# calibrate_toe_offset.py
# Calibrate toe offset (x,z) with small runtime:
# - Run pipeline ONCE up to Phase 1.8 (events, zupt, qs, p_imu_w)
# - Then grid-search (coarse -> fine) only re-running Phase 1.8 + Phase 1.9 scoring

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import pipeline_core as core


def precompute_until_phase_18(
    *,
    data_folder: str,
    participant: str,
    test: str,
    side: str,
    sensor: str | None,
    toe_marker: str | None,
    padding_s: float,
    event_source: str,
    eventnet_model,
    eventnet_meta: dict | None,
    dynamic_cutoff: bool,
    dynamic_zupt: bool,
    dynamic_ic_refine: bool,
):
    """
    Compute everything up to Phase 1.8 inputs, ONCE.
    Returns dict with everything needed to evaluate different (x,z) offsets fast.
    """
    side = str(side).lower()
    if sensor is None:
        sensor = "l_instep" if side == "left" else "r_instep"
    if toe_marker is None:
        toe_marker = "l_toe" if side == "left" else "r_toe"

    dataset = core.SensorPositionComparison2019Mocap(
        memory=core.Memory("./cache"),
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
    events_gt_imu = datapoint.convert_events_with_padding(
        events_mocap, from_time_axis="mocap", to_time_axis="imu"
    )

    # contiguous stride ids (prevents batch index issues)
    events_mocap = events_mocap.reset_index(drop=True).copy()
    events_gt_imu = events_gt_imu.reset_index(drop=True).copy()

    # Phase 1.3
    imu_f = core.phase_1_3_filter_imu(imu_df, fs_hz, dynamic=dynamic_cutoff)

    # Phase 1.4 events
    event_source = (event_source or "imu").lower().strip()
    if event_source == "ml":
        if eventnet_model is None:
            raise RuntimeError("event_source=ml but eventnet_model is None.")
        events_imu, _evdbg = core.phase_1_4_detect_events_ml(
            imu_f=imu_f, fs_hz=fs_hz, model=eventnet_model, meta=eventnet_meta or {}
        )
    else:
        events_imu, _evdbg = core.phase_1_4_detect_events_imu_only(imu_f, fs_hz)

    events_imu = core.phase_1_4b_filter_invalid_strides(events_imu, fs_hz)
    if events_imu is None or len(events_imu) == 0:
        raise RuntimeError("IMU event detection produced no valid strides after filtering.")

    # ZUPT mask
    zupt_mask = core.phase_1_4_get_zupt_mask_from_min_vel(
        events_imu,
        n_samples=len(imu_f),
        dynamic=dynamic_zupt,
        fs_hz=fs_hz,
        frac=0.03,
        min_hw=6,
        max_hw=25,
        stride_ref="ic2ic",
    )

    # Phase 1.5 orientation
    qs = core.phase_1_5_estimate_orientation(imu_f, fs_hz, zupt_mask=zupt_mask)

    # half-window heuristic (same as your current logic)
    half_window_samples = 10
    if dynamic_zupt:
        ref = events_gt_imu if (events_gt_imu is not None and len(events_gt_imu) > 3) else events_imu
        stride = ref.dropna(subset=["ic", "tc"]).sort_values("ic")
        ic = stride["ic"].to_numpy(dtype=int)
        if len(ic) >= 3:
            stride_s = np.median(np.diff(ic) / float(fs_hz))
            hw_s = 0.03 * float(stride_s)
            half_window_samples = int(np.clip(round(hw_s * fs_hz), 6, 25))

    qs = core.phase_1_5b_stance_attitude_correction(
        qs, imu_f, events_imu, fs_hz,
        half_window=half_window_samples,
        alpha="dynamic",
        use_median=True
    )

    acc_w = core.phase_1_5_compute_world_acceleration(imu_f, qs)
    specific_acc_w, _ = core.phase_1_5_remove_gravity(acc_w, zupt_mask)

    # optional IC refine
    events_imu = core.refine_ic_with_specific_acc(
        events_imu,
        specific_acc_w=specific_acc_w,
        fs_hz=fs_hz,
        dynamic=dynamic_ic_refine,
        pre_frac=0.12,
        post_frac=0.03,
        stride_ref="ic2ic",
    )

    # Phase 1.6 (z-only integrate)
    pz, vz = core.phase_1_6_integrate_z_only_per_stride_anchor_min_vel(
        specific_acc_w,
        events_imu,
        fs_hz,
        half_window=half_window_samples,
        enforce_end_vel_zero=True,
        shift_to_tc=True,
    )
    p_imu_w = np.zeros((len(pz), 3), dtype=float)
    p_imu_w[:, 2] = pz

    # Precompute mapping once (independent of toe offset)
    # Use a stable gate (you can expose this to CLI if needed)
    mapping_df, match_sum = core.match_strides_by_mid_swing_dp(
        events_imu, events_gt_imu, fs_hz, gate_s=0.15
    )
    if mapping_df.empty:
        raise RuntimeError(
            f"Calibration: no stride matches (det={match_sum['n_det']} gt={match_sum['n_gt']})."
        )

    return dict(
        participant=participant,
        test=test,
        side=side,
        sensor=sensor,
        toe_marker=toe_marker,
        fs_hz=fs_hz,
        mocap_traj=mocap_traj,
        events_mocap=events_mocap,
        events_gt_imu=events_gt_imu,
        events_imu=events_imu,
        qs=qs,
        p_imu_w=p_imu_w,
        mapping_df=mapping_df,
        half_window_samples=half_window_samples,
    )

def score_offset(pre, x_cm: float, z_cm: float, *, ground_mode="min_vel"):
    # Phase 1.8 via pipeline_core wrapper
    imu_stride_res, _debug = core.compute_imu_mtc_for_offset(
        pre,
        toe_x_cm=float(x_cm),
        toe_z_cm=float(z_cm),
        ground_mode=ground_mode,
    )

    # Phase 1.9 via pipeline_core wrapper
    m = core.score_offset_against_mocap(pre, imu_stride_res, ground_mode=ground_mode)
    if m is None:
        return None

    # Keep your existing return shape used by coarse_to_fine_calibration
    bias = float(m["bias_m"])
    rmse = float(m["rmse_m"])
    extra = {k: v for k, v in m.items() if k not in {"bias_m", "rmse_m", "n"}}
    n = int(m.get("n", 0))
    return bias, rmse, extra, n

def score_offset_multi(pres, x_cm: float, z_cm: float, *, ground_mode="min_vel",
                       objective="rmse", beta=0.5,
                       fail_penalty_m=0.20,
                       rmse_cap_m=0.20):
    """
    Multi-test, equal-weighted objective for a single (x,z).

    pres: list of Precomputed (one per test)
    returns: (bias, rmse, extra, n) in the SAME SHAPE your coarse-to-fine expects,
             plus per-test details in `extra`.
    """
    objective = (objective or "rmse").lower().strip()

    per_test = []
    data_terms = []
    biases = []
    rmses = []
    n_total = 0

    for pre in pres:
        imu_stride_res, _ = core.compute_imu_mtc_for_offset(
            pre, toe_x_cm=float(x_cm), toe_z_cm=float(z_cm), ground_mode=ground_mode
        )
        m = core.score_offset_against_mocap(pre, imu_stride_res, ground_mode=ground_mode)

        if m is None:
            # treat as a "failed test": contributes penalty but doesn't explode the search
            d = float(fail_penalty_m)
            per_test.append({"test": pre.trial.test, "ok": False, "d_m": d})
            data_terms.append(d)
            continue

        bias = float(m["bias_m"])
        rmse = float(m["rmse_m"])
        n = int(m.get("n", 0))
        n_total += n

        # Robustify each test’s contribution so one catastrophic trial doesn't dominate
        rmse_c = min(rmse, float(rmse_cap_m))

        if objective == "abs_bias":
            d = abs(bias)
        elif objective == "hybrid":
            d = rmse_c + float(beta) * abs(bias)
        else:  # "rmse"
            d = rmse_c

        per_test.append({
            "test": pre.trial.test,
            "ok": True,
            "bias_m": bias,
            "rmse_m": rmse,
            "rmse_used_m": rmse_c,
            "d_m": d,
            "n": n,
        })

        data_terms.append(d)
        biases.append(bias)
        rmses.append(rmse)

    if len(data_terms) == 0:
        return None

    # Equal weight per test (NOT per stride)
    data_term = float(np.mean(data_terms))

    # Return "aggregate bias/rmse" mainly for logging/compat; objective uses `data_term`
    bias_mean = float(np.mean(biases)) if biases else 0.0
    rmse_mean = float(np.mean(rmses)) if rmses else float(fail_penalty_m)

    extra = {
        "data_term_m": data_term,
        "per_test": per_test,
        "n_tests": len(pres),
        "n_ok": int(sum(1 for p in per_test if p.get("ok"))),
    }
    return bias_mean, rmse_mean, extra, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--side", required=True, choices=["left", "right"])
    ap.add_argument("--event_source", default="imu", choices=["imu", "ml"])
    ap.add_argument("--event_model_pt", default=None)
    ap.add_argument("--padding_s", type=float, default=3.0)
    ap.add_argument("--objective", default="rmse", choices=["rmse", "abs_bias", "hybrid"])
    ap.add_argument("--out_csv", default="toe_offset_calibration_xz.csv")

    # toggles
    ap.add_argument("--no_dynamic_cutoff", action="store_true")
    ap.add_argument("--no_dynamic_zupt", action="store_true")
    ap.add_argument("--no_dynamic_ic_refine", action="store_true")
    
    ap.add_argument("--tests", required=True,
        help="Comma-separated list, e.g. slow_10,normal_10,fast_10,slow_20,normal_20,fast_20,long")

    args = ap.parse_args()

    eventnet_model = None
    eventnet_meta = None
    if args.event_source == "ml":
        if not args.event_model_pt:
            raise SystemExit("--event_source ml requires --event_model_pt")
        eventnet_model, eventnet_meta = core.legacy.load_eventnet(args.event_model_pt, device="cpu")

    # ---- MULTI-TEST: parse tests and build Precomputed list ----
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    if len(tests) == 0:
        raise SystemExit("--tests is empty")

    # ---- Pipeline toggles (define ONCE) ----
    toggles = core.PipelineToggles(
        dynamic_cutoff=not args.no_dynamic_cutoff,
        dynamic_zupt=not args.no_dynamic_zupt,
        dynamic_ic_refine=not args.no_dynamic_ic_refine,
        dynamic_gate=True,
    )

    pres = []
    trials = []  # keep for metadata/output
    for t in tests:
        trial_t = core.load_trial(
            data_folder=args.data_folder,
            participant=args.participant,
            test=t,
            side=args.side,
            padding_s=args.padding_s,
            sensor=None,
            toe_marker=None,
        )
        pre_t = core.precompute(
            trial_t,
            event_source=args.event_source,
            eventnet_model=eventnet_model,
            eventnet_meta=eventnet_meta,
            toggles=toggles,
            gate_s=0.15,
        )
        trials.append(trial_t)
        pres.append(pre_t)

    print("[CAL] Multi-test calibration on:", ", ".join([tr.key.test for tr in trials]))

    # NOTE: This assumes you modify calibrate_toe_offset_coarse_to_fine to accept pres (list)
    best = core.calibrate_toe_offset_coarse_to_fine(
        pres,                       # list now supported
        objective=args.objective,
        ground_mode="min_vel",
    )



    row = dict(
        participant=trials[0].key.participant,
        test=",".join([tr.key.test for tr in trials]),   # <-- list of tests
        side=trials[0].key.side,
        imu_sensor=trials[0].sensor,
        toe_marker=trials[0].toe_marker,
        toe_x_cm=float(best["toe_x_cm"]),
        toe_z_cm=float(best["toe_z_cm"]),
        bias_mm=float(best["bias_m"]) * 1000.0,
        rmse_mm=float(best["rmse_m"]) * 1000.0,
        n=int(best.get("n", 0)),
    )


    df = pd.DataFrame([row])
    df.to_csv(args.out_csv, index=False)
    print(f"[CAL] Saved: {args.out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()