#!/usr/bin/env python3
"""
Stage 0: Face auto-axis detector for XIAO BMI160 ACK/NACK firmware.

Robust version:
- Uses one sequential BLE line queue.
- Does NOT create one asyncio task per received line.
- Drains stale notification lines between faces.
- Detects empty CSV files before trying axis detection.
- Supports resuming from a specific face list, e.g. --faces Back Top Bottom Left Right.

Firmware expected:
    GyroAndAccel_XIAO_RAM_100Hz_CSV_DUMP_ACK.ino

Protocol expected:
    start
    stop
    cancel
    ACK,<next_seq>
    NACK,<chunk_start>

Arduino dump markers:
    CSV_BEGIN
    seq,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
    DUMP_INFO,total=998,chunk=20
    CHUNK_BEGIN,20,40
    ...
    CHUNK_END,20,40
    STREAM_STOPPED,998 samples,missed=0,read_failures=0
    CSV_END

Important:
    CHUNK_BEGIN,start,end and CHUNK_END,start,end use END, not COUNT.
    CHUNK_END,20,40 means expected seq 20..39 and ACK,40.

Run:
    python stage0_face_auto_axis_detector_ROBUST.py

Resume after Front:
    python stage0_face_auto_axis_detector_ROBUST.py --faces Back Top Bottom Left Right
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from bleak import BleakClient, BleakScanner


DEVICE_NAME = "XIAO_BATTERY"
NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

CSV_HEADER = "seq,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"
EXPECTED_COLUMNS = 8

SCAN_TIMEOUT = 5.0
RETRY_DELAY = 2.0
DUMP_TIMEOUT_S = 90.0

DEFAULT_FACES = ["Front", "Back", "Top", "Bottom", "Left", "Right"]

FACE_HELP = {
    "Front": "toe-side face",
    "Back": "heel-side face",
    "Top": "top face",
    "Bottom": "bottom face",
    "Left": "left side face",
    "Right": "right side face",
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())


def compact_ranges(values: List[int]) -> str:
    values = sorted(set(values))
    if not values:
        return "none"

    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
        else:
            ranges.append((start, prev))
            start = prev = value
    ranges.append((start, prev))

    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def is_valid_sample(line: str) -> bool:
    parts = line.split(",")
    if len(parts) != EXPECTED_COLUMNS:
        return False
    try:
        int(parts[0])
        int(parts[1])
        for value in parts[2:]:
            float(value)
        return True
    except ValueError:
        return False


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def detect_axis_direction(csv_path: Path, drop_first_s: float = 1.0) -> Dict:
    df = pd.read_csv(csv_path)
    required = CSV_HEADER.split(",")

    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} missing columns: {missing_columns}")

    if len(df) == 0:
        raise ValueError(
            f"{csv_path} contains zero sample rows. "
            "The BLE connection probably dropped or recording did not complete."
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_rows = df[df[required].isna().any(axis=1)]
    if len(bad_rows):
        raise ValueError(f"{csv_path} contains {len(bad_rows)} non-numeric/missing rows.")

    df["seq"] = df["seq"].astype(int)
    df["t_ms"] = df["t_ms"].astype(int)

    seq_min = int(df["seq"].min())
    seq_max = int(df["seq"].max())
    expected = set(range(seq_min, seq_max + 1))
    seen = set(int(x) for x in df["seq"].to_numpy())
    missing_seq = sorted(expected - seen)
    duplicate_seq_count = int(df["seq"].duplicated().sum())
    out_of_order_count = int((df["seq"].diff().fillna(1) < 0).sum())

    t0 = int(df["t_ms"].iloc[0])
    df_use = df[df["t_ms"] >= t0 + int(round(drop_first_s * 1000.0))].copy()

    if len(df_use) < 100:
        raise ValueError(
            f"Too few samples after dropping first {drop_first_s:.2f}s: {len(df_use)} samples."
        )

    means = df_use[["ax_g", "ay_g", "az_g"]].mean()
    stds = df_use[["ax_g", "ay_g", "az_g"]].std(ddof=1)

    vec = np.array([float(means["ax_g"]), float(means["ay_g"]), float(means["az_g"])], dtype=float)
    abs_vec = np.abs(vec)
    axis_idx = int(np.argmax(abs_vec))
    axis = ["x", "y", "z"][axis_idx]
    direction = "plus" if vec[axis_idx] >= 0 else "minus"
    axis_direction = f"{axis}_{direction}"

    sorted_abs = np.sort(abs_vec)
    dominant_abs = float(sorted_abs[-1])
    second_abs = float(sorted_abs[-2])
    dominance_margin = dominant_abs - second_abs

    norms = np.linalg.norm(df_use[["ax_g", "ay_g", "az_g"]].to_numpy(dtype=float), axis=1)
    norm_mean = float(norms.mean())
    norm_std = float(norms.std(ddof=1))

    flags = []
    if missing_seq:
        flags.append(f"Missing sequence values: {compact_ranges(missing_seq)}")
    if duplicate_seq_count:
        flags.append(f"Duplicate sequence count: {duplicate_seq_count}")
    if out_of_order_count:
        flags.append(f"Out-of-order sequence count: {out_of_order_count}")
    if abs(norm_mean - 1.0) > 0.08:
        flags.append(f"Acceleration norm mean {norm_mean:.4f} g is far from 1 g.")
    if norm_std > 0.02:
        flags.append(f"Acceleration norm std {norm_std:.4f} g is high; container may have moved.")
    if dominant_abs < 0.70:
        flags.append(f"Dominant axis magnitude {dominant_abs:.4f} g is low.")
    if dominance_margin < 0.25:
        flags.append(f"Dominant axis margin {dominance_margin:.4f} g is small.")

    return {
        "axis": axis,
        "direction": direction,
        "axis_direction": axis_direction,
        "mean_accel_g": {
            "ax": float(vec[0]),
            "ay": float(vec[1]),
            "az": float(vec[2]),
        },
        "std_accel_g": {
            "ax": float(stds["ax_g"]),
            "ay": float(stds["ay_g"]),
            "az": float(stds["az_g"]),
        },
        "norm_g": {
            "mean": norm_mean,
            "std": norm_std,
        },
        "dominant_abs_g": dominant_abs,
        "second_abs_g": second_abs,
        "dominance_margin_g": dominance_margin,
        "samples_total": int(len(df)),
        "samples_used": int(len(df_use)),
        "seq_min": seq_min,
        "seq_max": seq_max,
        "missing_seq_count": int(len(missing_seq)),
        "missing_seq_ranges": compact_ranges(missing_seq),
        "duplicate_seq_count": duplicate_seq_count,
        "out_of_order_count": out_of_order_count,
        "quality_pass": len(flags) == 0,
        "quality_flags": flags,
    }


@dataclass
class FaceState:
    face: str
    out_dir: Path
    temp_path: Optional[Path] = None
    csv_file: Optional[object] = None
    csv_writer: Optional[csv.writer] = None

    recording_active: bool = False
    csv_started: bool = False
    csv_done: bool = False

    sample_count: int = 0
    bad_line_count: int = 0

    byte_count: int = 0
    stat_t0: float = field(default_factory=time.monotonic)

    seen_sequences: Set[int] = field(default_factory=set)
    duplicate_seq_count: int = 0
    out_of_order_count: int = 0
    last_written_seq: Optional[int] = None

    dump_total: Optional[int] = None
    chunk_size: Optional[int] = None
    stream_stopped_line: Optional[str] = None

    current_chunk_start: Optional[int] = None
    current_chunk_end: Optional[int] = None
    current_chunk_rows: Dict[int, str] = field(default_factory=dict)

    ack_count: int = 0
    nack_count: int = 0
    chunk_tx_count: int = 0


def open_temp_csv(st: FaceState) -> Path:
    st.out_dir.mkdir(parents=True, exist_ok=True)
    st.temp_path = st.out_dir / f"TEMP_{safe_name(st.face)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.csv_file = st.temp_path.open("w", newline="", encoding="utf-8")
    st.csv_writer = csv.writer(st.csv_file)
    st.csv_writer.writerow(CSV_HEADER.split(","))
    st.csv_file.flush()
    return st.temp_path


def close_csv(st: FaceState) -> None:
    if st.csv_file is not None:
        st.csv_file.flush()
        st.csv_file.close()
    st.csv_file = None
    st.csv_writer = None


def parse_dump_info(line: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"DUMP_INFO,total=(\d+),chunk=(\d+)", line)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_stream_stopped_samples(line: str) -> Optional[int]:
    match = re.search(r"STREAM_STOPPED,(\d+) samples", line)
    if not match:
        return None
    return int(match.group(1))


async def send_cmd(client: BleakClient, cmd: str, quiet: bool = False) -> None:
    await client.write_gatt_char(NUS_RX_UUID, (cmd.strip() + "\n").encode("utf-8"), response=False)
    if not quiet:
        print(f"\nSent command: {cmd.strip()}")


def make_notification_queue() -> Tuple[asyncio.Queue, callable]:
    queue: asyncio.Queue[str] = asyncio.Queue()
    rx_buffer = ""

    def notify_handler(sender, data: bytearray) -> None:
        nonlocal rx_buffer
        rx_buffer += data.decode("utf-8", errors="replace")

        while "\n" in rx_buffer:
            line, rx_buffer = rx_buffer.split("\n", 1)
            line = line.rstrip("\r").strip()
            if line:
                queue.put_nowait(line)

    return queue, notify_handler


async def drain_queue(queue: asyncio.Queue, max_wait_s: float = 0.25) -> int:
    drained = 0
    deadline = time.monotonic() + max_wait_s

    while time.monotonic() < deadline:
        try:
            await asyncio.wait_for(queue.get(), timeout=0.02)
            drained += 1
        except asyncio.TimeoutError:
            break

    while not queue.empty():
        queue.get_nowait()
        drained += 1

    return drained


async def handle_chunk_end(st: FaceState, client: BleakClient, start: int, end: int) -> None:
    # Arduino sends start,end where end is exclusive.
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
            st.duplicate_seq_count += 1
            continue

        line = st.current_chunk_rows[seq]
        if st.csv_writer is not None:
            st.csv_writer.writerow(line.split(","))

        st.seen_sequences.add(seq)
        if st.last_written_seq is not None and seq < st.last_written_seq:
            st.out_of_order_count += 1
        if st.last_written_seq is None or seq > st.last_written_seq:
            st.last_written_seq = seq

        st.sample_count += 1

    if st.csv_file is not None:
        st.csv_file.flush()

    st.ack_count += 1
    await send_cmd(client, f"ACK,{end}", quiet=True)


async def process_line(st: FaceState, client: BleakClient, line: str) -> None:
    now = time.monotonic()
    if now - st.stat_t0 >= 1.0:
        if st.recording_active:
            phase = "dump" if st.csv_started else "record"
            print(
                f"\rBLE | saved: {st.sample_count} | phase: {phase} | ACK={st.ack_count} NACK={st.nack_count}",
                end="",
                flush=True,
            )
        st.stat_t0 = now

    if line == "CSV_BEGIN":
        st.csv_started = True
        return

    if line == CSV_HEADER:
        return

    if line.startswith("STREAM_STARTED"):
        print("\n" + line)
        return

    if line.startswith("DUMP_INFO"):
        total, chunk = parse_dump_info(line)
        st.dump_total = total
        st.chunk_size = chunk
        print(f"\n{line}")
        return

    if line.startswith("CHUNK_BEGIN"):
        parts = line.split(",")
        if len(parts) >= 3:
            st.current_chunk_start = int(parts[1])
            st.current_chunk_end = int(parts[2])
            st.current_chunk_rows = {}
            st.chunk_tx_count += 1
        return

    if line.startswith("CHUNK_END"):
        parts = line.split(",")
        if len(parts) >= 3:
            start = int(parts[1])
            end = int(parts[2])
            await handle_chunk_end(st, client, start, end)
        return

    if line.startswith("STREAM_STOPPED"):
        st.stream_stopped_line = line
        return

    if line == "CSV_END":
        st.csv_done = True
        st.recording_active = False
        st.csv_started = False
        return

    if line.startswith("STATUS") or line.startswith("ERR") or line.startswith("CANCELLED") or line.startswith("RETRY"):
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

        print("Device not found. Retrying...\n")
        await asyncio.sleep(RETRY_DELAY)


async def wait_for_lines_until_csv_end(
    st: FaceState,
    client: BleakClient,
    queue: asyncio.Queue,
    disconnected_event: asyncio.Event,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s

    while not st.csv_done:
        if disconnected_event.is_set():
            raise ConnectionError("BLE disconnected before CSV_END.")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timeout waiting for CSV_END.")

        try:
            line = await asyncio.wait_for(queue.get(), timeout=min(1.0, remaining))
        except asyncio.TimeoutError:
            continue

        await process_line(st, client, line)


async def process_record_phase_lines(st: FaceState, client: BleakClient, queue: asyncio.Queue) -> None:
    while not queue.empty():
        line = queue.get_nowait()
        await process_line(st, client, line)


async def record_one_face(
    client: BleakClient,
    queue: asyncio.Queue,
    disconnected_event: asyncio.Event,
    face: str,
    seconds: int,
    out_dir: Path,
    drop_first_s: float,
) -> Dict:
    face_clean = safe_name(face)
    face_desc = FACE_HELP.get(face, "physical face")

    print("\n" + "=" * 70)
    print(f"FACE: {face}")
    print(f"Instruction: place the {face} face ({face_desc}) facing UP toward the ceiling.")
    print("Keep the container completely still during recording.")
    input("Press ENTER when ready... ")

    drained = await drain_queue(queue, max_wait_s=0.30)
    if drained:
        print(f"Drained {drained} stale BLE line(s) before recording.")

    st = FaceState(face=face, out_dir=out_dir, recording_active=True)
    temp_path = open_temp_csv(st)

    print(f"Recording {seconds}s. Temporary CSV: {temp_path}")
    await send_cmd(client, "start")

    start_time = time.monotonic()
    last_remaining = None

    while True:
        if disconnected_event.is_set():
            close_csv(st)
            raise ConnectionError("BLE disconnected during recording phase.")

        elapsed = time.monotonic() - start_time
        remaining = max(0, seconds - int(elapsed))
        if remaining != last_remaining:
            print(f"\rRecording {face}... {remaining:>3}s left", end="", flush=True)
            last_remaining = remaining

        await process_record_phase_lines(st, client, queue)

        if elapsed >= seconds:
            break

        await asyncio.sleep(0.05)

    await send_cmd(client, "stop")
    print("Waiting for ACK/NACK CSV dump + CSV_END...")

    await wait_for_lines_until_csv_end(
        st=st,
        client=client,
        queue=queue,
        disconnected_event=disconnected_event,
        timeout_s=DUMP_TIMEOUT_S,
    )

    close_csv(st)

    if st.stream_stopped_line:
        print("\n" + st.stream_stopped_line)

    reported = parse_stream_stopped_samples(st.stream_stopped_line or "")
    if reported is not None and reported != st.sample_count:
        print(f"WARNING: XIAO reported {reported} samples, Python saved {st.sample_count} unique rows.")

    if st.sample_count <= 0:
        raise RuntimeError(
            f"No samples were saved for face {face}. "
            "Reset the XIAO, reconnect BLE, and rerun this face."
        )

    detection = detect_axis_direction(temp_path, drop_first_s=drop_first_s)
    axis_direction = detection["axis_direction"]

    final_path = unique_path(out_dir / f"{face_clean}_{axis_direction}.csv")
    os.replace(temp_path, final_path)

    print(f"\nSaved: {final_path}")
    print(
        "Detected:",
        f"{axis_direction} | mean=[{detection['mean_accel_g']['ax']:.4f}, "
        f"{detection['mean_accel_g']['ay']:.4f}, {detection['mean_accel_g']['az']:.4f}] g | "
        f"norm={detection['norm_g']['mean']:.4f} g",
    )
    print(
        f"Transport: saved={st.sample_count}, ACK={st.ack_count}, NACK={st.nack_count}, "
        f"bad_lines={st.bad_line_count}, chunks={st.chunk_tx_count}"
    )

    if detection["quality_flags"]:
        print("Quality warnings:")
        for flag in detection["quality_flags"]:
            print(f"  - {flag}")
    else:
        print("Quality: PASS")

    # Give Windows BLE stack / XIAO time to settle before next face.
    await asyncio.sleep(1.0)
    await drain_queue(queue, max_wait_s=0.20)

    return {
        "face": face,
        "face_description": face_desc,
        "csv_path": str(final_path),
        "detected_axis_direction": axis_direction,
        "detected_axis": detection["axis"],
        "detected_direction": detection["direction"],
        "detection": detection,
        "transport": {
            "xiao_reported_samples": reported,
            "python_saved_unique_samples": st.sample_count,
            "bad_line_count": st.bad_line_count,
            "ack_count": st.ack_count,
            "nack_count": st.nack_count,
            "chunk_tx_count": st.chunk_tx_count,
            "duplicates_seen": st.duplicate_seq_count,
            "out_of_order_count": st.out_of_order_count,
            "stream_stopped_line": st.stream_stopped_line,
        },
    }


def check_face_axis_duplicates(results: List[Dict]) -> List[str]:
    warnings = []
    mapping: Dict[str, List[str]] = {}

    for result in results:
        mapping.setdefault(result["detected_axis_direction"], []).append(result["face"])

    for axis_dir, faces in mapping.items():
        if len(faces) > 1:
            warnings.append(f"{axis_dir} was detected for multiple faces: {faces}")

    required = {"x_plus", "x_minus", "y_plus", "y_minus", "z_plus", "z_minus"}
    found = set(mapping.keys())
    missing = sorted(required - found)
    if missing and len(results) >= 6:
        warnings.append(f"Missing expected axis directions: {missing}")

    return warnings


async def run(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = await find_device(args.device_name)
    disconnected_event = asyncio.Event()

    def on_disconnect(client):
        print("\nDisconnected.")
        disconnected_event.set()

    queue, notify_handler = make_notification_queue()

    async with BleakClient(device, disconnected_callback=on_disconnect) as client:
        print(f"Connected to {device.name} [{device.address}]")
        await client.start_notify(NUS_TX_UUID, notify_handler)

        results = []
        try:
            for face in args.faces:
                if disconnected_event.is_set():
                    raise ConnectionError("BLE disconnected. Restart script and XIAO, then resume remaining faces.")

                result = await record_one_face(
                    client=client,
                    queue=queue,
                    disconnected_event=disconnected_event,
                    face=face,
                    seconds=args.seconds,
                    out_dir=args.out_dir,
                    drop_first_s=args.drop_first_s,
                )
                results.append(result)

        except Exception:
            # Try to leave firmware in a safe state.
            try:
                if client.is_connected:
                    await send_cmd(client, "cancel", quiet=True)
            except Exception:
                pass
            raise

        finally:
            try:
                await client.stop_notify(NUS_TX_UUID)
            except Exception:
                pass

    warnings = check_face_axis_duplicates(results)

    summary = {
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "device_name": args.device_name,
        "record_seconds_per_face": args.seconds,
        "drop_first_seconds_for_detection": args.drop_first_s,
        "face_definition": FACE_HELP,
        "results": results,
        "overall_quality": {
            "pass": not warnings and all(r["detection"]["quality_pass"] for r in results),
            "warnings": warnings,
        },
    }

    summary_path = args.out_dir / "face_axis_mapping.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("FACE AXIS SUMMARY")
    for result in results:
        print(f"  {result['face']:<8} -> {result['detected_axis_direction']:<8} | {Path(result['csv_path']).name}")

    if warnings:
        print("\nOverall warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nOverall check: PASS")

    print(f"\nSaved summary JSON: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 0 face recorder with automatic accelerometer axis/sign detection."
    )
    parser.add_argument("--device-name", default=DEVICE_NAME)
    parser.add_argument("--seconds", default=10, type=int)
    parser.add_argument("--drop-first-s", default=1.0, type=float)
    parser.add_argument("--out-dir", default=Path("imu_face_calibration"), type=Path)
    parser.add_argument("--faces", nargs="+", default=DEFAULT_FACES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seconds < 3:
        raise ValueError("Use at least 3 seconds per face.")
    asyncio.run(run(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
