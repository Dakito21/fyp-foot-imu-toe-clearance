"""calibrate_toe_offset.py

Calibrate toe offset (x,z) using pipeline_core:
- For each requested test: load trial, precompute up to Phase 1.6 once
- Run coarse-to-fine (x,z) search by re-running Phase 1.8+1.9 only

This is a simplified version of calibrate_toe_offset.py:
- removes unused helper functions
- keeps only the multi-test calibration path
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

import pipeline_core as core

import contextlib
import os
import sys

@contextlib.contextmanager
def suppress_stdout():
    save_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as fnull:
            sys.stdout = fnull
            yield
    finally:
        sys.stdout = save_stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--side", required=True, choices=["left", "right"])

    ap.add_argument("--event_source", default="imu", choices=["imu", "ml"])
    ap.add_argument("--event_model_pt", default=None)

    ap.add_argument("--padding_s", type=float, default=3.0)
    ap.add_argument("--gate_s", type=float, default=0.15)

    ap.add_argument("--objective", default="rmse", choices=["rmse", "abs_bias", "hybrid"])
    ap.add_argument("--out_csv", default="toe_offset_calibration_xz.csv")

    ap.add_argument(
        "--tests",
        required=True,
        help="Comma-separated list, e.g. slow_10,normal_10,fast_10,slow_20,normal_20,fast_20,long",
    )

    args = ap.parse_args()

    eventnet_model = None
    eventnet_meta = None
    if args.event_source == "ml":
        if not args.event_model_pt:
            raise SystemExit("--event_source ml requires --event_model_pt")
        eventnet_model, eventnet_meta = core.legacy.load_eventnet(args.event_model_pt, device="cpu")

    tests = [t.strip() for t in str(args.tests).split(",") if t.strip()]
    if not tests:
        raise SystemExit("--tests is empty")

    toggles = core.PipelineToggles(
        dynamic_cutoff=True,
        dynamic_zupt=True,
        dynamic_ic_refine=True,
        dynamic_gate=True,
    )

    pres = []
    trials = []
    for t in tests:
        trial = core.load_trial(
            data_folder=args.data_folder,
            participant=args.participant,
            test=t,
            side=args.side,
            padding_s=args.padding_s,
            sensor=None,
            toe_marker=None,
        )
        pre = core.precompute(
            trial,
            event_source=args.event_source,
            eventnet_model=eventnet_model,
            eventnet_meta=eventnet_meta,
            toggles=toggles,
            gate_s=float(args.gate_s),
        )
        trials.append(trial)
        pres.append(pre)

    print("[CAL] Multi-test calibration on:", ", ".join([tr.key.test for tr in trials]))

    sensitivity_sweep(pres)

    best = core.calibrate_toe_offset_coarse_to_fine(
        pres,
        objective=args.objective,
        ground_mode="min_vel",
    )

    row = dict(
        participant=trials[0].key.participant,
        test=",".join([tr.key.test for tr in trials]),
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

def sensitivity_sweep(pres):
    print("\n=== Sensitivity Sweep (Multi-test mean RMSE/Bias) ===")

    def eval_offset(x_cm, z_cm):
        rmses = []
        biases = []
        n_total = 0

        for pre in pres:
            with suppress_stdout():
                imu_stride_res, _ = core.compute_imu_mtc_for_offset(
                    pre,
                    toe_x_cm=float(x_cm),
                    toe_z_cm=float(z_cm),
                    ground_mode="min_vel",
                )
                scored = core.score_offset_against_mocap(
                    pre,
                    imu_stride_res,
                    ground_mode="min_vel",
                )
                
            if scored is None:
                continue

            rmses.append(scored["rmse_m"])
            biases.append(scored["bias_m"])
            n_total += int(scored.get("n", 0))

        if not rmses:
            return float("nan"), float("nan"), 0

        bias_mm = 1000.0 * float(np.mean(biases))
        rmse_mm = 1000.0 * float(np.mean(rmses))
        return bias_mm, rmse_mm, n_total

    def eval_offset_y(y_cm, z_cm=0.0):
        rmses = []
        biases = []
        n_total = 0
        for pre in pres:
            imu_stride_res, _ = core.compute_imu_mtc_for_offset_y(
                pre,
                toe_y_cm=float(y_cm),
                toe_z_cm=float(z_cm),
                ground_mode="min_vel",
            )
            scored = core.score_offset_against_mocap(pre, imu_stride_res, ground_mode="min_vel")
            if scored is None:
                continue
            rmses.append(scored["rmse_m"])
            biases.append(scored["bias_m"])
            n_total += int(scored.get("n", 0))
        if not rmses:
            return float("nan"), float("nan"), 0
        return 1000.0*np.mean(biases), 1000.0*np.mean(rmses), n_total

    def eval_offset_xy(x_cm, y_cm, z_cm=0.0):
        rmses, biases = [], []
        n_total = 0
        for pre in pres:
            p_toe_b = np.array([x_cm/100.0, y_cm/100.0, z_cm/100.0], float)

            imu_stride_res, _ = core.legacy.phase_1_8_compute_mtc_per_stride(
                pre.p_imu_w, pre.qs, p_toe_b, pre.events_imu,
                ground_mode="min_vel",
                half_window=pre.half_window_samples,
            )

            scored = core.score_offset_against_mocap(pre, imu_stride_res, ground_mode="min_vel")
            if scored is None:
                continue
            rmses.append(scored["rmse_m"])
            biases.append(scored["bias_m"])
            n_total += int(scored.get("n", 0))

        if not rmses:
            return float("nan"), float("nan"), 0

        return 1000.0*np.mean(biases), 1000.0*np.mean(rmses), n_total


    # Sweep X (z=0)
    print("\n-- Sweep X (z=0) --")
    for x in [-12, -15, -8, 0, 8, 15, 22]:
        bias_mm, rmse_mm, n_total = eval_offset(x, 0.0)
        print(f"x={x:>5} cm  →  RMSE={rmse_mm:8.2f} mm   Bias={bias_mm:8.2f} mm   n={n_total}")

    x_fix = -8.0

    print(f"\n-- Sweep Z (x={x_fix:g}) --")
    for i, z in enumerate([-6, -4, -2, 0, 2, 4, 6], 1):
        print(f"[SWEEP Z] {i}/7  x={x_fix:.2f} cm  z={z:.2f} cm", flush=True)
        bias_mm, rmse_mm, n_total = eval_offset(x_fix, float(z))
        print(f"          RMSE={rmse_mm:8.2f} mm   Bias={bias_mm:8.2f} mm   n={n_total}")

    print("\n-- Sweep Y (x=0, z=0) --")
    ys = [18,20,22,24,26]
    for i, y in enumerate(ys, 1):
        print(f"[SWEEP Y] {i}/{len(ys)}  y={y:.2f} cm", flush=True)
        bias_mm, rmse_mm, n_total = eval_offset_y(float(y), 0.0)
        print(f"          RMSE={rmse_mm:8.2f} mm   Bias={bias_mm:8.2f} mm   n={n_total}")

    print("\n-- Sweep (X,Y) grid (z=0) --")
    xs = [-6, -4, -2, 0]
    ys = [12, 14, 16, 18, 20]

    best = None
    count = 0
    total = len(xs) * len(ys)

    for x in xs:
        for y in ys:
            count += 1
            print(f"[SWEEP XY] {count}/{total}  x={x:.2f}  y={y:.2f}", flush=True)

            bias_mm, rmse_mm, n_total = eval_offset_xy(x, y, 0.0)  # you add eval_offset_xy below
            print(f"           RMSE={rmse_mm:8.2f} mm   Bias={bias_mm:8.2f} mm   n={n_total}")

            if np.isfinite(rmse_mm) and (best is None or rmse_mm < best["rmse_mm"]):
                best = {"x": x, "y": y, "rmse_mm": rmse_mm, "bias_mm": bias_mm}

    print("\n[SWEEP XY] Best:", best)



if __name__ == "__main__":
    main()
