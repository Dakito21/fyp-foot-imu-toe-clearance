#!/usr/bin/env python3
"""
metrics_from_imu_only_summary.py

Reads imu_only_all_summary.csv (or similarly structured file) and prints:
1) Per-participant metrics table (precision/recall + TC/IC timing + MTC accuracy)
2) Table 2 summary stats across participants (precision/recall distribution)
3) Table 4 summary stats across participants (MTC bias/RMSE/MAE distribution)
4) Global (count-weighted) precision/recall + pooled timing/MTC metrics
5) A compact terminal "list summary" per participant

Usage:
  python metrics_from_imu_only_summary.py --csv imu_only_all_summary.csv
Optional:
  python metrics_from_imu_only_summary.py --csv imu_only_all_summary.csv --out_csv per_participant_metrics.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np


REQUIRED_COLS = [
    "participant", "test", "side",
    "det", "gt", "matched", "precision_pct", "recall_pct",
    "tc_rmse_ms", "ic_rmse_ms",
    "mtc_bias_mm", "mtc_rmse_mm", "mtc_mean_abs_diff_mm", "mtc_n",
]


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def _summary_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _pooled_rmse(n: pd.Series, rmse: pd.Series) -> float:
    """Pooled RMSE = sqrt( sum(n * rmse^2) / sum(n) )."""
    n = pd.to_numeric(n, errors="coerce").fillna(0.0).astype(float)
    r = pd.to_numeric(rmse, errors="coerce").astype(float)
    denom = float(n.sum())
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(float((n * (r ** 2)).sum()) / denom))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to imu_only_all_summary.csv")
    ap.add_argument("--out_csv", default="", help="Optional output CSV for per-participant table")
    ap.add_argument("--sort_by", default="mtc_rmse_mm", help="Sort per-participant table by this column")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _require_columns(df, REQUIRED_COLS)

    # ---- Per-participant table (already one row per participant in your file)
    per_cols = [
        "participant", "test", "side",
        "det", "gt", "matched", "precision_pct", "recall_pct",
        "tc_rmse_ms", "ic_rmse_ms",
        "mtc_bias_mm", "mtc_rmse_mm", "mtc_mean_abs_diff_mm", "mtc_n",
    ]
    per = df[per_cols].copy()

    # Strong typing / rounding for clean terminal output
    int_cols = ["det", "gt", "matched", "mtc_n"]
    for c in int_cols:
        per[c] = pd.to_numeric(per[c], errors="coerce").astype("Int64")

    float_cols = [c for c in per.columns if c not in ["participant", "test", "side"] + int_cols]
    for c in float_cols:
        per[c] = pd.to_numeric(per[c], errors="coerce")

    if args.sort_by in per.columns:
        per = per.sort_values(by=args.sort_by, ascending=False, na_position="last")

    # ---- Global (count-based) precision/recall
    total_det = float(pd.to_numeric(df["det"], errors="coerce").fillna(0).sum())
    total_gt = float(pd.to_numeric(df["gt"], errors="coerce").fillna(0).sum())
    total_matched = float(pd.to_numeric(df["matched"], errors="coerce").fillna(0).sum())

    global_precision = (total_matched / total_det * 100.0) if total_det > 0 else float("nan")
    global_recall = (total_matched / total_gt * 100.0) if total_gt > 0 else float("nan")

    # ---- Pooled timing and pooled MTC (weighted by mtc_n)
    pooled_tc_rmse = _pooled_rmse(df["mtc_n"], df["tc_rmse_ms"])
    pooled_ic_rmse = _pooled_rmse(df["mtc_n"], df["ic_rmse_ms"])
    pooled_mtc_rmse = _pooled_rmse(df["mtc_n"], df["mtc_rmse_mm"])
    pooled_mtc_mae = float(
        (pd.to_numeric(df["mtc_n"], errors="coerce").fillna(0).astype(float)
         * pd.to_numeric(df["mtc_mean_abs_diff_mm"], errors="coerce").astype(float)
        ).sum()
        / max(1.0, float(pd.to_numeric(df["mtc_n"], errors="coerce").fillna(0).astype(float).sum()))
    )
    pooled_mtc_bias = float(
        (pd.to_numeric(df["mtc_n"], errors="coerce").fillna(0).astype(float)
         * pd.to_numeric(df["mtc_bias_mm"], errors="coerce").astype(float)
        ).sum()
        / max(1.0, float(pd.to_numeric(df["mtc_n"], errors="coerce").fillna(0).astype(float).sum()))
    )

    # ---- Table 2 (distribution across participants)
    pstats = _summary_stats(df["precision_pct"])
    rstats = _summary_stats(df["recall_pct"])

    table2 = pd.DataFrame(
        [
            ["Precision (%)", pstats["mean"], pstats["sd"], pstats["median"], f'{pstats["q1"]:.2f}–{pstats["q3"]:.2f}', f'{pstats["min"]:.2f}–{pstats["max"]:.2f}'],
            ["Recall (%)",    rstats["mean"], rstats["sd"], rstats["median"], f'{rstats["q1"]:.2f}–{rstats["q3"]:.2f}', f'{rstats["min"]:.2f}–{rstats["max"]:.2f}'],
        ],
        columns=["Metric", "Mean", "SD", "Median", "Q1–Q3", "Min–Max"],
    )

    # ---- Table 4 (MTC accuracy distribution across participants)
    bstats = _summary_stats(df["mtc_bias_mm"])
    rmstats = _summary_stats(df["mtc_rmse_mm"])
    maestats = _summary_stats(df["mtc_mean_abs_diff_mm"])

    table4 = pd.DataFrame(
        [
            ["Bias (mm)", bstats["mean"], bstats["sd"], bstats["median"], f'{bstats["q1"]:.2f}–{bstats["q3"]:.2f}', f'{bstats["min"]:.2f}–{bstats["max"]:.2f}'],
            ["RMSE (mm)", rmstats["mean"], rmstats["sd"], rmstats["median"], f'{rmstats["q1"]:.2f}–{rmstats["q3"]:.2f}', f'{rmstats["min"]:.2f}–{rmstats["max"]:.2f}'],
            ["MAE (mm)",  maestats["mean"], maestats["sd"], maestats["median"], f'{maestats["q1"]:.2f}–{maestats["q3"]:.2f}', f'{maestats["min"]:.2f}–{maestats["max"]:.2f}'],
        ],
        columns=["Metric", "Mean", "SD", "Median", "Q1–Q3", "Min–Max"],
    )

    # ---- Terminal prints
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("\n==============================")
    print("PER-PARTICIPANT METRICS TABLE")
    print("==============================")
    print(per.to_string(index=False, justify="left", float_format=lambda x: f"{x:8.3f}"))

    print("\n==============================")
    print("GLOBAL (COUNT-BASED) EVENT METRICS")
    print("==============================")
    print(f"Total det      : {int(total_det)}")
    print(f"Total gt       : {int(total_gt)}")
    print(f"Total matched  : {int(total_matched)}")
    print(f"Global Precision (matched/det): {global_precision:.2f}%")
    print(f"Global Recall    (matched/gt) : {global_recall:.2f}%")

    print("\n==============================")
    print("POOLED (WEIGHTED BY mtc_n) METRICS")
    print("==============================")
    print(f"Pooled TC RMSE (ms): {pooled_tc_rmse:.2f}")
    print(f"Pooled IC RMSE (ms): {pooled_ic_rmse:.2f}")
    print(f"Pooled MTC Bias (mm): {pooled_mtc_bias:.2f}")
    print(f"Pooled MTC MAE  (mm): {pooled_mtc_mae:.2f}")
    print(f"Pooled MTC RMSE (mm): {pooled_mtc_rmse:.2f}")

    print("\n==============================")
    print("EVENT PRECISION/RECALL DISTRIBUTION (ACROSS PARTICIPANTS)")
    print("==============================")
    print(table2.to_string(index=False, justify="left", float_format=lambda x: f"{x:8.3f}"))

    print("\n==============================")
    print("MTC ACCURACY DISTRIBUTION (ACROSS PARTICIPANTS)")
    print("==============================")
    print(table4.to_string(index=False, justify="left", float_format=lambda x: f"{x:8.3f}"))

    print("\n==============================")
    print("TERMINAL LIST SUMMARY (PER PARTICIPANT)")
    print("==============================")
    for _, r in per.sort_values(["participant"]).iterrows():
        p = r["participant"]
        prec = float(r["precision_pct"]) if pd.notna(r["precision_pct"]) else float("nan")
        rec = float(r["recall_pct"]) if pd.notna(r["recall_pct"]) else float("nan")
        tcrmse = float(r["tc_rmse_ms"]) if pd.notna(r["tc_rmse_ms"]) else float("nan")
        icrmse = float(r["ic_rmse_ms"]) if pd.notna(r["ic_rmse_ms"]) else float("nan")
        mtcrmse = float(r["mtc_rmse_mm"]) if pd.notna(r["mtc_rmse_mm"]) else float("nan")
        mtcmae = float(r["mtc_mean_abs_diff_mm"]) if pd.notna(r["mtc_mean_abs_diff_mm"]) else float("nan")
        n = int(r["mtc_n"]) if pd.notna(r["mtc_n"]) else 0
        det = int(r["det"]) if pd.notna(r["det"]) else 0
        gt = int(r["gt"]) if pd.notna(r["gt"]) else 0
        m = int(r["matched"]) if pd.notna(r["matched"]) else 0
        print(
            f"- {p:6s} | Prec={prec:6.2f}% Rec={rec:6.2f}% | "
            f"TC_RMSE={tcrmse:6.1f}ms IC_RMSE={icrmse:6.1f}ms | "
            f"MTC_RMSE={mtcrmse:6.2f}mm MAE={mtcmae:6.2f}mm | "
            f"n={n:3d} (det={det}, gt={gt}, matched={m})"
        )

    # ---- Optional output CSV for thesis tables
    if args.out_csv:
        out_path = Path(args.out_csv)
        per.to_csv(out_path, index=False)
        print(f"\nWrote per-participant metrics CSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
