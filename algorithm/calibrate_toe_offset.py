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
import inspect

import pipeline_core as core

def sensitivity_sweep(
    pres,
    *,
    ground_mode: str = "min_vel",
    x_vals_cm=None,
    z_vals_cm=None,
    y_vals_cm=None,
) -> None:
    """Quick 1D sensitivity sweeps around (0,0,0) for debugging/calibration intuition.

    Prints multi-test mean RMSE and bias for:
      - X sweep with (y=z=0)
      - Z sweep with (x=y=0)
      - Y sweep with (x=z=0) if core supports toe_y_cm

    Uses core.compute_imu_mtc_for_offset + core.score_offset_against_mocap.
    """
    if x_vals_cm is None:
        x_vals_cm = [-22, -15, -8, 0, 8, 15, 22]
    if z_vals_cm is None:
        z_vals_cm = [-4, -2, 0, 2, 4]
    if y_vals_cm is None:
        y_vals_cm = [0, 2, 4, 6, 8, 10]

    sig = inspect.signature(core.compute_imu_mtc_for_offset)
    has_y = "toe_y_cm" in sig.parameters
    sig_score = inspect.signature(core.score_offset_against_mocap)
    score_supports_quiet = "quiet" in sig_score.parameters

    print("\n=== Sensitivity Sweep (multi-test mean RMSE/Bias) ===")

    def eval_offset(x_cm: float, y_cm: float, z_cm: float):
        rmses = []
        biases = []
        n_total = 0

        for pre in pres:
            kwargs = dict(
                pre=pre,
                toe_x_cm=float(x_cm),
                toe_y_cm=float(y_cm),
                toe_z_cm=float(z_cm),
                ground_mode=str(ground_mode),
                show_progress=False,
            )
            if has_y:
                kwargs["toe_y_cm"] = float(y_cm)

            imu_stride_res, _ = core.compute_imu_mtc_for_offset(**kwargs)
            score_kwargs = dict(
                pre=pre,
                imu_stride_res=imu_stride_res,
                ground_mode=str(ground_mode),
            )

            if score_supports_quiet:
                score_kwargs["quiet"] = True

            scored = core.score_offset_against_mocap(**score_kwargs)

            if not scored:
                continue

            rm = scored.get("rmse_m", None)
            bi = scored.get("bias_m", None)
            if rm is None or bi is None:
                continue

            rmses.append(float(rm))
            biases.append(float(bi))
            n_total += int(scored.get("n", 0))

        if not rmses:
            return float("nan"), float("nan"), 0

        bias_mm = 1000.0 * float(np.mean(biases))
        rmse_mm = 1000.0 * float(np.mean(rmses))
        return bias_mm, rmse_mm, int(n_total)

    # Sweep X (y=z=0)
    print("\n-- Sweep X (y=0, z=0) --")
    for i, x in enumerate(x_vals_cm, start=1):
        bias_mm, rmse_mm, n_total = eval_offset(x, 0.0, 0.0)
        print(f"x={x:>6} cm ({i}/{len(x_vals_cm)}) →  RMSE={rmse_mm:8.2f} mm   Bias={bias_mm:8.2f} mm   n={n_total}")

    # Sweep Z (x=y=0)
    print("\n-- Sweep Z (x=0, y=0) --")
    for i, z in enumerate(z_vals_cm, start=1):
        bias_mm, rmse_mm, n_total = eval_offset(0.0, 0.0, z)
        print(f"z={z:>6} cm ({i}/{len(z_vals_cm)}) →  RMSE={rmse_mm:8.2f} mm   Bias={bias_mm:8.2f} mm   n={n_total}")

    # Sweep Y (x=z=0) if supported
    print("\n-- Sweep Y (x=0, z=0) --")
    if not has_y:
        print("y-sweep skipped: pipeline_core.compute_imu_mtc_for_offset has no toe_y_cm parameter in this version.")
        return

    for i, y in enumerate(y_vals_cm, start=1):
        bias_mm, rmse_mm, n_total = eval_offset(0.0, y, 0.0)
        print(f"y={y:>6} cm ({i}/{len(y_vals_cm)}) →  RMSE={rmse_mm:8.2f} mm   Bias={bias_mm:8.2f} mm   n={n_total}")

def sensitivity_sweep_xy(
    pres,
    *,
    ground_mode: str = "min_vel",
    x_vals_cm=None,
    y_vals_cm=None,
    z_cm: float = 0.0,
) -> None:
    """
    2D sensitivity sweep: (x,y) with fixed z=z_cm.
    Prints multi-test mean RMSE/Bias (mm) per grid point.
    """

    import numpy as np
    import inspect

    if x_vals_cm is None:
        x_vals_cm = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    if y_vals_cm is None:
        y_vals_cm = [0, 2, 4, 6, 8, 10]

    # Detect whether compute_imu_mtc_for_offset supports toe_y_cm
    sig = inspect.signature(core.compute_imu_mtc_for_offset)
    has_y = "toe_y_cm" in sig.parameters
    if not has_y:
        print("\n[SWEEP XY] skipped: core.compute_imu_mtc_for_offset has no toe_y_cm")
        return

    # score_offset_against_mocap may not support quiet
    sig_score = inspect.signature(core.score_offset_against_mocap)
    score_supports_quiet = "quiet" in sig_score.parameters

    print("\n=== Sensitivity Sweep 2D (x,y) with z fixed ===")
    print(f"ground_mode={ground_mode} | z={z_cm:.2f} cm")
    print(f"x grid: {len(x_vals_cm)} values | y grid: {len(y_vals_cm)} values | total={len(x_vals_cm)*len(y_vals_cm)}")

    def eval_offset(x_cm: float, y_cm: float, z_cm: float):
        rmses = []
        biases = []
        n_total = 0

        for pre in pres:
            imu_stride_res, _ = core.compute_imu_mtc_for_offset(
                pre,
                toe_x_cm=float(x_cm),
                toe_y_cm=float(y_cm),
                toe_z_cm=float(z_cm),
                ground_mode=str(ground_mode),
                show_progress=False,
            )

            score_kwargs = dict(
                pre=pre,
                imu_stride_res=imu_stride_res,
                ground_mode=str(ground_mode),
            )
            if score_supports_quiet:
                score_kwargs["quiet"] = True

            scored = core.score_offset_against_mocap(**score_kwargs)
            if not scored:
                continue

            rm = scored.get("rmse_m", None)
            bi = scored.get("bias_m", None)
            if rm is None or bi is None:
                continue

            rmses.append(float(rm))
            biases.append(float(bi))
            n_total += int(scored.get("n", 0))

        if not rmses:
            return float("nan"), float("nan"), 0

        bias_mm = 1000.0 * float(np.mean(biases))
        rmse_mm = 1000.0 * float(np.mean(rmses))
        return bias_mm, rmse_mm, int(n_total)

    # --- table header ---
    x_hdr = "y\\x | " + " ".join([f"{x:>6.0f}" for x in x_vals_cm])
    print("\nRMSE (mm):")
    print(x_hdr)
    print("-" * len(x_hdr))

    best = {"rmse_mm": float("inf"), "x": None, "y": None, "bias_mm": None}

    total = len(x_vals_cm) * len(y_vals_cm)
    done = 0

    rmse_table = {}
    bias_table = {}

    for iy, y in enumerate(y_vals_cm, start=1):
        row_rmse = []
        row_bias = []

        for ix, x in enumerate(x_vals_cm, start=1):
            done += 1
            # light progress ping every row start + every ~10% if you want
            # (kept minimal)
            bias_mm, rmse_mm, n_total = eval_offset(x, y, z_cm)

            rmse_table[(x, y)] = rmse_mm
            bias_table[(x, y)] = bias_mm

            row_rmse.append(f"{rmse_mm:6.2f}")
            row_bias.append(f"{bias_mm:6.2f}")

            if np.isfinite(rmse_mm) and rmse_mm < best["rmse_mm"]:
                best.update({"rmse_mm": rmse_mm, "x": x, "y": y, "bias_mm": bias_mm})

        print(f"{y:>3.0f}  | " + " ".join(row_rmse))

    print("\nBias (mm):")
    print(x_hdr)
    print("-" * len(x_hdr))
    for y in y_vals_cm:
        row_bias = [f"{bias_table[(x, y)]:6.2f}" for x in x_vals_cm]
        print(f"{y:>3.0f}  | " + " ".join(row_bias))

    if best["x"] is not None:
        print(
            f"\n[SWEEP XY] BEST @ z={z_cm:.2f} cm → "
            f"x={best['x']:.2f} cm, y={best['y']:.2f} cm | "
            f"RMSE={best['rmse_mm']:.2f} mm | Bias={best['bias_mm']:.2f} mm"
        )


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

    
    # 1D sensitivity sweeps around (0,0,0) before coarse-to-fine search
    # sensitivity_sweep(pres, ground_mode="min_vel")


    # sensitivity_sweep_xy(
    #     pres,
    #     ground_mode="min_vel",
    #     x_vals_cm=[-8, -6, -4, -2, 0, 2, 4, 6, 8],
    #     y_vals_cm=[0, 2, 4, 6, 8, 10, 12 ,14, 16],
    #     z_cm=0.0,
    # )


    print("[CAL] Multi-test calibration on:", ", ".join([tr.key.test for tr in trials]))

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
        toe_y_cm=float(best["toe_y_cm"]),
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
