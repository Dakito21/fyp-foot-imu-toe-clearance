#!/usr/bin/env python3
"""
Stage-A ML gait event detector (stance vs swing) for SensorPositionComparison2019Mocap.

- Learns stance probability p_stance[t] from instep IMU.
- Labels are derived from mocap events converted to IMU sample indices using convert_events_with_padding.
- Exports a single param.pt bundle usable by your clean.py inference.

Why this formulation?
- It generalizes better across normal/slow/fast/long because stance/swing is more stable than sparse IC/TC peaks.
- IC and TC are recovered later as stance transitions (swing->stance = IC, stance->swing = TC).

Requires:
  pip install gaitmap-datasets torch joblib numpy pandas
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Memory

from gaitmap_datasets.sensor_position_comparison_2019 import SensorPositionComparison2019Mocap


# -------------------------
# Utilities: robust normalize
# -------------------------
def robust_normalize_per_trial(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    x: (C, T)
    Returns normalized x and stats.
    """
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    scale = 1.4826 * mad + eps
    x_n = (x - med) / scale
    stats = {"median": med.squeeze(-1), "scale": scale.squeeze(-1)}
    return x_n, stats


def build_features(imu_df: pd.DataFrame) -> np.ndarray:
    """
    imu_df is the instep sensor dataframe (columns include acc_x/y/z, gyr_x/y/z).
    Returns X with shape (C, T).
    """
    acc = imu_df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32)
    gyr = imu_df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=np.float32)

    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True).astype(np.float32)
    gyr_norm = np.linalg.norm(gyr, axis=1, keepdims=True).astype(np.float32)

    X = np.concatenate([acc, gyr, acc_norm, gyr_norm], axis=1)  # (T, 8)
    return X.T  # (8, T)


def stance_labels_from_events(events_imu: pd.DataFrame, T: int) -> np.ndarray:
    """
    events_imu columns: start,end,tc,ic (in IMU samples for THIS test segment)
    Returns y stance mask shape (T,), dtype float32 with:
      stance=1, swing=0.
    Inside each stride: swing=[tc, ic], stance elsewhere inside [start,end].
    """
    y = np.full((T,), fill_value=np.nan, dtype=np.float32)

    for _, r in events_imu.iterrows():
        s = int(r["start"]); e = int(r["end"])
        tc = int(r["tc"]); ic = int(r["ic"])

        # clamp
        s = max(0, min(T - 1, s))
        e = max(0, min(T, e))
        tc = max(0, min(T, tc))
        ic = max(0, min(T, ic))

        if not (s < e):
            continue

        # initialize stride region as stance
        y[s:e] = 1.0

        # mark swing as 0 between tc..ic if order is valid
        if s <= tc < ic <= e:
            y[tc:ic] = 0.0
        else:
            # if ordering is off (rare), skip this stride region entirely to avoid corrupt labels
            y[s:e] = np.nan

    return y


# -------------------------
# Model: small 1D CNN (fully convolutional)
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 7, d: int = 1):
        super().__init__()
        pad = (k // 2) * d
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class EventNetStageA(nn.Module):
    def __init__(self, c_in: int = 8, c_hidden: int = 64):
        super().__init__()
        self.b1 = ConvBlock(c_in, c_hidden, k=7, d=1)
        self.b2 = ConvBlock(c_hidden, c_hidden, k=7, d=2)
        self.b3 = ConvBlock(c_hidden, c_hidden, k=7, d=4)
        self.head = nn.Conv1d(c_hidden, 1, kernel_size=1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        logits = self.head(x)  # (B,1,T)
        return logits


# -------------------------
# Dataset: random crops from each trial
# -------------------------
@dataclass
class Trial:
    X: np.ndarray  # (C,T)
    y: np.ndarray  # (T,) stance labels with NaNs outside labeled regions
    pid: str
    test: str
    side: str
    fs_hz: float


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, trials: List[Trial], crop_len: int, min_valid_frac: float = 0.7):
        self.trials = trials
        self.crop_len = crop_len
        self.min_valid_frac = min_valid_frac

    def __len__(self):
        return len(self.trials) * 200  # "virtual length"

    def __getitem__(self, idx):
        trial = self.trials[idx % len(self.trials)]
        X, y = trial.X, trial.y
        T = X.shape[1]
        L = self.crop_len

        if T <= L + 1:
            start = 0
        else:
            # try multiple attempts to find a crop with enough valid labels
            for _ in range(20):
                start = random.randint(0, T - L - 1)
                yc = y[start:start + L]
                valid = np.isfinite(yc)
                if valid.mean() >= self.min_valid_frac:
                    break

        Xc = X[:, start:start + L]
        yc = y[start:start + L]

        valid = np.isfinite(yc)
        # fill NaNs (will be masked out)
        yc_f = np.where(valid, yc, 0.0).astype(np.float32)

        return (
            torch.from_numpy(Xc).float(),                # (C,L)
            torch.from_numpy(yc_f).float(),              # (L,)
            torch.from_numpy(valid.astype(np.float32)),  # (L,)
        )


def masked_bce_with_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, pos_weight: float = 1.0):
    """
    logits: (B,1,T)
    y:      (B,T)
    mask:   (B,T) float 0/1 where label is valid
    """
    # compute per-sample loss
    bce = F.binary_cross_entropy_with_logits(
        logits.squeeze(1), y, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    bce = bce * mask
    denom = mask.sum().clamp_min(1.0)
    return bce.sum() / denom


# -------------------------
# Data loading via gaitmap_datasets
# -------------------------
def load_trials(
    data_folder: str,
    padding_s: float,
    sides: list[str],
    tests: list[str] | None,
    participants: list[str] | None,
):
    from joblib import Memory
    from gaitmap_datasets.sensor_position_comparison_2019 import SensorPositionComparison2019Mocap

    dataset = SensorPositionComparison2019Mocap(
        memory=Memory("./cache"),
        data_folder=data_folder,
        data_padding_s=float(padding_s),
    )

    idx = dataset.create_index()  # columns: participant, test
    trials = []

    for _, r in idx.iterrows():
        pid = str(r["participant"])
        test = str(r["test"])

        if participants is not None and pid not in participants:
            continue
        if tests is not None and test not in tests:
            continue

        subset = dataset.get_subset(participant=[pid], test=[test])
        if len(subset) == 0:
            continue

        dp = subset[0]
        fs_hz = float(dp.sampling_rate_hz)
        imu_all = dp.data

        for side in sides:
            sensor = "l_instep" if side == "left" else "r_instep"
            if sensor not in imu_all.columns.get_level_values(0):
                continue

            imu_df = imu_all[sensor].copy()
            X = build_features(imu_df)
            X, _ = robust_normalize_per_trial(X)

            events_mocap = dp.mocap_events_[side]
            if len(events_mocap) == 0:
                continue

            events_imu = dp.convert_events_with_padding(
                events_mocap,
                from_time_axis="mocap",
                to_time_axis="imu",
            )

            T = X.shape[1]
            y = stance_labels_from_events(events_imu, T=T)

            if np.isfinite(y).mean() < 0.2:
                continue

            trials.append(
                Trial(
                    X=X.astype(np.float32),
                    y=y.astype(np.float32),
                    pid=pid,
                    test=test,
                    side=side,
                    fs_hz=fs_hz,
                )
            )

    if len(trials) == 0:
        raise RuntimeError("No trials loaded. Check filters and dataset path.")

    print(f"[INFO] Loaded {len(trials)} trials")
    return trials


# -------------------------
# Train / Val split (participant-held-out)
# -------------------------
def split_by_participant(trials: List[Trial], val_participant: str | None):
    if val_participant is None:
        # pick one participant deterministically
        pids = sorted(set(t.pid for t in trials))
        val_participant = pids[-1]

    train = [t for t in trials if t.pid != val_participant]
    val = [t for t in trials if t.pid == val_participant]
    if len(val) == 0 or len(train) == 0:
        raise RuntimeError("Bad split. Provide --val_participant that exists and leaves training data.")
    return train, val, val_participant


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True, help="Path to sensorpositoncomparison-v1.0.0-beta folder")
    ap.add_argument("--out_pt", default="param_stageA.pt")
    ap.add_argument("--padding_s", type=float, default=3.0)

    ap.add_argument("--sides", default="left", choices=["left", "right", "both"])
    ap.add_argument("--tests", default="all", help="Comma list (e.g. normal_10,slow_10) or 'all'")
    ap.add_argument("--participants", default="all", help="Comma list (e.g. 4d91,6e2e) or 'all'")
    ap.add_argument("--val_participant", default=None, help="Participant id held out for validation")

    ap.add_argument("--crop_s", type=float, default=1.25, help="Training crop length in seconds")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--pos_weight", type=float, default=1.0, help="BCE pos_weight for stance class")
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sides = ["left", "right"] if args.sides == "both" else [args.sides]

    tests = None if args.tests == "all" else [t.strip() for t in args.tests.split(",") if t.strip()]
    participants = None if args.participants == "all" else [p.strip() for p in args.participants.split(",") if p.strip()]

    trials = load_trials(
        data_folder=args.data_folder,
        padding_s=args.padding_s,
        sides=sides,
        tests=tests,
        participants=participants,
    )

    train_trials, val_trials, val_pid = split_by_participant(trials, args.val_participant)

    fs_hz = train_trials[0].fs_hz
    crop_len = int(round(args.crop_s * fs_hz))
    crop_len = max(crop_len, 128)

    train_ds = CropDataset(train_trials, crop_len=crop_len)
    val_ds = CropDataset(val_trials, crop_len=crop_len)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EventNetStageA(c_in=8, c_hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for X, y, m in train_dl:
            X = X.to(device)          # (B,C,L)
            y = y.to(device)          # (B,L)
            m = m.to(device)          # (B,L)

            logits = model(X)         # (B,1,L)
            loss = masked_bce_with_logits(logits, y, m, pos_weight=args.pos_weight)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += float(loss.item())
            tr_n += 1

        tr_loss /= max(tr_n, 1)

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for X, y, m in val_dl:
                X = X.to(device)
                y = y.to(device)
                m = m.to(device)
                logits = model(X)
                loss = masked_bce_with_logits(logits, y, m, pos_weight=args.pos_weight)
                va_loss += float(loss.item())
                va_n += 1

        va_loss /= max(va_n, 1)
        print(f"[EP {ep:02d}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_pid={val_pid} | crop_len={crop_len}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    bundle = {
        "state_dict": best_state,
        "model_cfg": {"name": "EventNetStageA", "c_in": 8, "c_hidden": args.hidden},
        "feature_cfg": {
            "channels": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","acc_norm","gyr_norm"],
            "norm": "median_mad_per_trial",
        },
        "fs_hz": float(fs_hz),
        "postproc_cfg": {
            "smooth_ms": 35.0,
            "th_on": 0.55,
            "th_off": 0.45,
            "min_stance_s": 0.08,
            "min_swing_s": 0.08,
            "refractory_s": 0.25,
        },
        "train_info": {
            "val_participant": val_pid,
            "best_val_loss": float(best_val),
            "padding_s": float(args.padding_s),
            "crop_s": float(args.crop_s),
            "tests": tests if tests is not None else "all",
            "participants": participants if participants is not None else "all",
            "sides": sides,
            "seed": int(args.seed),
        },
    }

    torch.save(bundle, args.out_pt)
    print(f"[OK] Saved: {args.out_pt}")


if __name__ == "__main__":
    main()
