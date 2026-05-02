#!/usr/bin/env python3
"""
Stage 2 one-click recorder + calibrator for XIAO BMI160 ACK/NACK firmware.

Records the shoe-mounted still trial, saves CSV, then immediately computes:
- gyro_bias_dps
- mounted gravity vector
- stillness/quality metrics

Firmware expected:
    GyroAndAccel_XIAO_RAM_100Hz_CSV_DUMP_ACK.ino

Run:
    python stage2_record_then_calibrate.py --stage1-accel-calib stage1_container_accel_calibration.json

Optional apply to walking CSV:
    python stage2_record_then_calibrate.py --stage1-accel-calib stage1_container_accel_calibration.json --apply walking.csv --apply-out walking_calibrated.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd
from bleak import BleakClient, BleakScanner


DEVICE_NAME = "XIAO_BATTERY"
NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

CSV_HEADER = "seq,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"
REQUIRED_COLUMNS = CSV_HEADER.split(",")
ACCEL_COLS = ["ax_g", "ay_g", "az_g"]
GYRO_COLS = ["gx_dps", "gy_dps", "gz_dps"]

SCAN_TIMEOUT = 5.0
DUMP_TIMEOUT_S = 90.0


def compact_ranges(values: Iterable[int]) -> str:
    values = sorted(set(int(v) for v in values))
    if not values:
        return "none"
    ranges = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
        else:
            ranges.append((start, prev))
            start = prev = v
    ranges.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())


def is_valid_sample(line: str) -> bool:
    parts = line.split(",")
    if len(parts) != len(REQUIRED_COLUMNS):
        return False
    try:
        int(parts[0])
        int(parts[1])
        for value in parts[2:]:
            float(value)
        return True
    except ValueError:
        return False


def parse_stream_stopped_samples(line: str) -> Optional[int]:
    match = re.search(r"STREAM_STOPPED,(\d+) samples", line)
    return int(match.group(1)) if match else None


def load_validate_csv(path: Path, strict_seq: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} missing columns {missing_cols}")

    df = df[REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = df[df[REQUIRED_COLUMNS].isna().any(axis=1)]
    if len(bad):
        raise ValueError(f"{path} has {len(bad)} bad/non-numeric rows.")
    if len(df) == 0:
        raise ValueError(f"{path} has no sample rows.")

    df["seq"] = df["seq"].astype(int)
    df["t_ms"] = df["t_ms"].astype(int)

    seq_min, seq_max = int(df.seq.min()), int(df.seq.max())
    expected = set(range(seq_min, seq_max + 1))
    seen = set(map(int, df.seq.to_numpy()))
    missing = sorted(expected - seen)
    dup = int(df.seq.duplicated().sum())
    oo = int((df.seq.diff().fillna(1) < 0).sum())

    validation = {
        "path": str(path),
        "rows": int(len(df)),
        "seq_min": seq_min,
        "seq_max": seq_max,
        "missing_seq_count": int(len(missing)),
        "missing_seq_ranges": compact_ranges(missing),
        "duplicate_seq_count": dup,
        "out_of_order_count": oo,
        "seq_pass": len(missing) == 0 and dup == 0 and oo == 0,
    }

    if strict_seq and not validation["seq_pass"]:
        raise ValueError(
            f"Sequence check failed: missing={validation['missing_seq_ranges']}, dup={dup}, out_of_order={oo}"
        )

    return df, validation


def trim_startup(df: pd.DataFrame, drop_first_s: float) -> pd.DataFrame:
    if drop_first_s <= 0:
        return df.copy()
    keep_from = int(df.t_ms.iloc[0]) + int(round(drop_first_s * 1000.0))
    out = df[df.t_ms >= keep_from].copy()
    if len(out) < 100:
        raise ValueError(f"Too few rows after startup trim: {len(out)}")
    return out


def load_stage1(path: Optional[Path]):
    if path is None:
        return None, None
    cal = json.loads(path.read_text(encoding="utf-8"))
    off = np.array([
        float(cal["accel_offset_g"]["ax"]),
        float(cal["accel_offset_g"]["ay"]),
        float(cal["accel_offset_g"]["az"]),
    ])
    scale = np.array([
        float(cal["accel_scale_factor"]["ax"]),
        float(cal["accel_scale_factor"]["ay"]),
        float(cal["accel_scale_factor"]["az"]),
    ])
    return off, scale


def apply_accel_calibration(df: pd.DataFrame, off, scale) -> pd.DataFrame:
    out = df.copy()
    if off is None or scale is None:
        return out
    raw = out[ACCEL_COLS].to_numpy(float)
    corr = (raw - off.reshape(1, 3)) * scale.reshape(1, 3)
    out["ax_raw_g"] = out["ax_g"]
    out["ay_raw_g"] = out["ay_g"]
    out["az_raw_g"] = out["az_g"]
    out["ax_g"] = corr[:, 0]
    out["ay_g"] = corr[:, 1]
    out["az_g"] = corr[:, 2]
    return out


def calculate_stage2(df: pd.DataFrame, validation: Dict[str, Any], csv_path: Path, drop_first_s: float, stage1_path: Optional[Path]) -> Dict[str, Any]:
    gm = df[GYRO_COLS].mean()
    gs = df[GYRO_COLS].std(ddof=1)
    gnorm = np.linalg.norm(df[GYRO_COLS].to_numpy(float), axis=1)

    am = df[ACCEL_COLS].mean()
    ast = df[ACCEL_COLS].std(ddof=1)
    acc = df[ACCEL_COLS].to_numpy(float)
    anorm = np.linalg.norm(acc, axis=1)

    mean_vec = am.to_numpy(float)
    mean_norm = float(np.linalg.norm(mean_vec))
    grav = mean_vec / mean_norm if mean_norm > 0 else np.array([math.nan, math.nan, math.nan])

    t0, t1 = int(df.t_ms.iloc[0]), int(df.t_ms.iloc[-1])
    dur = (t1 - t0) / 1000.0
    hz = (len(df) - 1) / dur if dur > 0 and len(df) > 1 else None

    flags = []
    if float(gnorm.std(ddof=1)) > 0.5:
        flags.append("High gyro variation; shoe/container may have moved.")
    if float(anorm.std(ddof=1)) > 0.02:
        flags.append("High acceleration norm variation; shoe/container may have moved/vibrated.")
    if abs(float(anorm.mean()) - 1.0) > 0.08:
        flags.append("Acceleration norm far from 1 g. Check Stage 1 calibration or stillness.")

    return {
        "stage": "stage2_record_then_calibrate_shoe_mounted_gyro_bias_and_gravity_reference",
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source_csv": str(csv_path),
        "stage1_accel_calibration_used": None if stage1_path is None else str(stage1_path),
        "usage_note": "Use gyro_bias_dps to correct walking trials. Do not subtract accel_static_mean_g from walking data; it contains gravity.",
        "recording_validation": validation,
        "processing": {
            "drop_first_seconds": float(drop_first_s),
            "samples_used": int(len(df)),
            "duration_s_after_trim": float(dur),
            "effective_hz_after_trim": None if hz is None else float(hz),
        },
        "gyro_bias_dps": {"gx": float(gm.gx_dps), "gy": float(gm.gy_dps), "gz": float(gm.gz_dps)},
        "gyro_noise_std_dps": {"gx": float(gs.gx_dps), "gy": float(gs.gy_dps), "gz": float(gs.gz_dps)},
        "gyro_norm_stats_dps": {"mean": float(gnorm.mean()), "std": float(gnorm.std(ddof=1)), "max": float(gnorm.max())},
        "accel_static_mean_g": {"ax": float(am.ax_g), "ay": float(am.ay_g), "az": float(am.az_g)},
        "accel_noise_std_g": {"ax": float(ast.ax_g), "ay": float(ast.ay_g), "az": float(ast.az_g)},
        "accel_norm_stats_g": {
            "mean": float(anorm.mean()), "std": float(anorm.std(ddof=1)),
            "min": float(anorm.min()), "max": float(anorm.max()),
            "norm_of_mean_vector": mean_norm,
        },
        "mounted_gravity_unit_vector_sensor_frame": {"x": float(grav[0]), "y": float(grav[1]), "z": float(grav[2])},
        "quality": {"pass": len(flags) == 0, "flags": flags},
    }


def apply_to_walking(input_csv: Path, output_csv: Path, cal: Dict[str, Any], off, scale, strict_seq: bool) -> None:
    df, val = load_validate_csv(input_csv, strict_seq=strict_seq)
    df = apply_accel_calibration(df, off, scale)
    gb = cal["gyro_bias_dps"]
    df["gx_raw_dps"] = df["gx_dps"]
    df["gy_raw_dps"] = df["gy_dps"]
    df["gz_raw_dps"] = df["gz_dps"]
    df["gx_dps"] = df["gx_dps"] - float(gb["gx"])
    df["gy_dps"] = df["gy_dps"] - float(gb["gy"])
    df["gz_dps"] = df["gz_dps"] - float(gb["gz"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nApplied calibration to walking CSV: {output_csv}")
    print("Raw gyro preserved as gx_raw_dps, gy_raw_dps, gz_raw_dps")


@dataclass
class RecordingState:
    out_dir: Path
    temp_path: Optional[Path] = None
    csv_file: Optional[object] = None
    csv_writer: Optional[csv.writer] = None
    recording_active: bool = False
    csv_started: bool = False
    csv_done: bool = False
    sample_count: int = 0
    bad_line_count: int = 0
    seen_sequences: Set[int] = field(default_factory=set)
    current_chunk_rows: Dict[int, str] = field(default_factory=dict)
    ack_count: int = 0
    nack_count: int = 0
    chunk_tx_count: int = 0
    stream_stopped_line: Optional[str] = None


def open_temp_csv(st: RecordingState, label: str) -> Path:
    st.out_dir.mkdir(parents=True, exist_ok=True)
    st.temp_path = st.out_dir / f"TEMP_{safe_name(label)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.csv_file = st.temp_path.open("w", newline="", encoding="utf-8")
    st.csv_writer = csv.writer(st.csv_file)
    st.csv_writer.writerow(REQUIRED_COLUMNS)
    st.csv_file.flush()
    return st.temp_path


def close_csv(st: RecordingState) -> None:
    if st.csv_file is not None:
        st.csv_file.flush()
        st.csv_file.close()
    st.csv_file = None
    st.csv_writer = None


async def send_cmd(client: BleakClient, cmd: str, quiet: bool = False) -> None:
    await client.write_gatt_char(NUS_RX_UUID, (cmd.strip() + "\n").encode("utf-8"), response=False)
    if not quiet:
        print(f"\nSent command: {cmd.strip()}")


def make_notification_queue():
    queue: asyncio.Queue[str] = asyncio.Queue()
    rx_buffer = ""

    def notify_handler(sender, data: bytearray):
        nonlocal rx_buffer
        rx_buffer += data.decode("utf-8", errors="replace")
        while "\n" in rx_buffer:
            line, rx_buffer = rx_buffer.split("\n", 1)
            line = line.rstrip("\r").strip()
            if line:
                queue.put_nowait(line)

    return queue, notify_handler


async def drain_queue(queue: asyncio.Queue, max_wait_s: float = 0.25) -> None:
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        try:
            await asyncio.wait_for(queue.get(), timeout=0.02)
        except asyncio.TimeoutError:
            break
    while not queue.empty():
        queue.get_nowait()


async def handle_chunk_end(st: RecordingState, client: BleakClient, start: int, end: int) -> None:
    expected = set(range(start, end))
    got = set(st.current_chunk_rows.keys())
    missing = sorted(expected - got)

    if missing:
        st.nack_count += 1
        print(f"\nChunk {start}-{end-1}: missing {compact_ranges(missing)} -> NACK,{start}")
        await send_cmd(client, f"NACK,{start}", quiet=True)
        return

    for seq in range(start, end):
        if seq in st.seen_sequences:
            continue
        line = st.current_chunk_rows[seq]
        if st.csv_writer is not None:
            st.csv_writer.writerow(line.split(","))
        st.seen_sequences.add(seq)
        st.sample_count += 1

    if st.csv_file is not None:
        st.csv_file.flush()

    st.ack_count += 1
    await send_cmd(client, f"ACK,{end}", quiet=True)


async def process_line(st: RecordingState, client: BleakClient, line: str) -> None:
    if line == "CSV_BEGIN":
        st.csv_started = True
        return
    if line == CSV_HEADER:
        return
    if line.startswith("STREAM_STARTED") or line.startswith("DUMP_INFO") or line.startswith("RETRY"):
        print("\n" + line)
        return
    if line.startswith("CHUNK_BEGIN"):
        st.current_chunk_rows = {}
        st.chunk_tx_count += 1
        return
    if line.startswith("CHUNK_END"):
        parts = line.split(",")
        if len(parts) >= 3:
            await handle_chunk_end(st, client, int(parts[1]), int(parts[2]))
        return
    if line.startswith("STREAM_STOPPED"):
        st.stream_stopped_line = line
        return
    if line == "CSV_END":
        st.csv_done = True
        st.recording_active = False
        st.csv_started = False
        return
    if line.startswith("STATUS") or line.startswith("ERR") or line.startswith("CANCELLED"):
        print("\n" + line)
        return
    if line.startswith("RX_CMD"):
        return

    if st.csv_started:
        if is_valid_sample(line):
            seq = int(line.split(",", 1)[0])
            st.current_chunk_rows[seq] = line
        else:
            st.bad_line_count += 1


async def find_device(device_name: str):
    while True:
        print("Scanning...")
        devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT)
        for d in devices:
            if d.name == device_name:
                print(f"Found {d.name} [{d.address}]")
                return d
        print("Device not found. Retrying...")
        await asyncio.sleep(2.0)


async def wait_until_csv_end(st: RecordingState, client: BleakClient, queue: asyncio.Queue, disconnected_event: asyncio.Event) -> None:
    deadline = time.monotonic() + DUMP_TIMEOUT_S
    last_print = time.monotonic()

    while not st.csv_done:
        if disconnected_event.is_set():
            raise ConnectionError("BLE disconnected before CSV_END.")
        if time.monotonic() > deadline:
            raise TimeoutError("Timeout waiting for CSV_END.")

        try:
            line = await asyncio.wait_for(queue.get(), timeout=0.5)
            await process_line(st, client, line)
        except asyncio.TimeoutError:
            pass

        if time.monotonic() - last_print >= 1.0:
            phase = "dump" if st.csv_started else "record"
            print(f"\rBLE | saved={st.sample_count} | phase={phase} | ACK={st.ack_count} NACK={st.nack_count}", end="", flush=True)
            last_print = time.monotonic()


async def process_available(st: RecordingState, client: BleakClient, queue: asyncio.Queue) -> None:
    while not queue.empty():
        await process_line(st, client, queue.get_nowait())


async def record_csv(args) -> Path:
    device = await find_device(args.device_name)
    disconnected_event = asyncio.Event()

    def on_disconnect(client):
        print("\nDisconnected.")
        disconnected_event.set()

    queue, notify_handler = make_notification_queue()

    async with BleakClient(device, disconnected_callback=on_disconnect) as client:
        print(f"Connected to {device.name} [{device.address}]")
        await client.start_notify(NUS_TX_UUID, notify_handler)
        await drain_queue(queue)

        st = RecordingState(out_dir=args.out_dir, recording_active=True)
        temp_path = open_temp_csv(st, args.label)

        print("\nStage 2 condition:")
        print("  Mount casing on shoe exactly like walking setup.")
        print("  Shoe flat and completely still.")
        print("  Do not walk.")
        input("Press ENTER to start Stage 2 recording... ")

        print(f"Recording {args.seconds}s. Temporary CSV: {temp_path}")
        await send_cmd(client, "start")

        start_time = time.monotonic()
        last_remaining = None
        while True:
            if disconnected_event.is_set():
                close_csv(st)
                raise ConnectionError("BLE disconnected during recording.")
            elapsed = time.monotonic() - start_time
            remaining = max(0, args.seconds - int(elapsed))
            if remaining != last_remaining:
                print(f"\rRecording shoe-mounted still... {remaining:>3}s left", end="", flush=True)
                last_remaining = remaining
            await process_available(st, client, queue)
            if elapsed >= args.seconds:
                break
            await asyncio.sleep(0.05)

        await send_cmd(client, "stop")
        print("Waiting for ACK/NACK CSV dump + CSV_END...")
        await wait_until_csv_end(st, client, queue, disconnected_event)
        close_csv(st)

        try:
            await client.stop_notify(NUS_TX_UUID)
        except Exception:
            pass

    if st.stream_stopped_line:
        print("\n" + st.stream_stopped_line)

    reported = parse_stream_stopped_samples(st.stream_stopped_line or "")
    if reported is not None and reported != st.sample_count:
        print(f"WARNING: XIAO reported {reported}, Python saved {st.sample_count}.")

    if st.sample_count <= 0:
        raise RuntimeError("No samples saved.")

    final_path = args.out_dir / f"{safe_name(args.label)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.replace(temp_path, final_path)

    print("\nRecording complete:")
    print(f"  CSV: {final_path}")
    print(f"  saved={st.sample_count}, ACK={st.ack_count}, NACK={st.nack_count}, bad_lines={st.bad_line_count}")

    return final_path


async def async_main(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    off, scale = load_stage1(args.stage1_accel_calib)

    csv_path = await record_csv(args)

    df_raw, validation = load_validate_csv(csv_path, strict_seq=not args.allow_bad_seq)
    df = trim_startup(df_raw, args.drop_first_s)
    df = apply_accel_calibration(df, off, scale)

    cal = calculate_stage2(df, validation, csv_path, args.drop_first_s, args.stage1_accel_calib)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(cal, indent=2), encoding="utf-8")

    gb = cal["gyro_bias_dps"]
    am = cal["accel_static_mean_g"]
    an = cal["accel_norm_stats_g"]
    gu = cal["mounted_gravity_unit_vector_sensor_frame"]

    print("\nStage 2 calibration saved:")
    print(f"  {args.out_json}")
    print(f"Gyro bias[dps]: gx={gb['gx']:.6f}, gy={gb['gy']:.6f}, gz={gb['gz']:.6f}")
    print(f"Accel mean[g]: ax={am['ax']:.6f}, ay={am['ay']:.6f}, az={am['az']:.6f}; norm={an['mean']:.6f}")
    print(f"Mounted gravity unit vector: [{gu['x']:.6f}, {gu['y']:.6f}, {gu['z']:.6f}]")
    print("Quality:", "PASS" if cal["quality"]["pass"] else "WARNING")
    for flag in cal["quality"]["flags"]:
        print(" -", flag)

    if args.apply is not None:
        out = args.apply_out or args.apply.with_name(args.apply.stem + "_calibrated.csv")
        apply_to_walking(args.apply, out, cal, off, scale, strict_seq=not args.allow_bad_seq)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 one-click shoe-mounted still recording + calibration.")
    p.add_argument("--device-name", default=DEVICE_NAME)
    p.add_argument("--seconds", default=20, type=int)
    p.add_argument("--drop-first-s", default=1.0, type=float)
    p.add_argument("--out-dir", default=Path("imu_stage2"), type=Path)
    p.add_argument("--label", default="shoe_flat_still")
    p.add_argument("--stage1-accel-calib", default=None, type=Path)
    p.add_argument("--out-json", default=Path("stage2_shoe_mounted_session_calibration.json"), type=Path)
    p.add_argument("--allow-bad-seq", action="store_true")
    p.add_argument("--apply", default=None, type=Path)
    p.add_argument("--apply-out", default=None, type=Path)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.seconds < 5:
        raise ValueError("Use at least 5 seconds. Recommended: 20 seconds.")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
