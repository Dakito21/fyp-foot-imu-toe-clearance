import asyncio
import csv
import os
import re
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "XIAO_BATTERY"
NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # Python -> XIAO
NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # XIAO -> Python

SCAN_TIMEOUT = 5.0
RETRY_DELAY = 2.0
SAVE_DIR = Path("imu_recordings")
CSV_HEADER = "seq,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"
EXPECTED_COLUMNS = 8
DUMP_TIMEOUT_S = 120.0

# Calibration files.
# Stage 1 = container accelerometer offset/scale calibration.
# Stage 2 = shoe-mounted gyroscope bias calibration.
APPLY_CALIBRATION_ON_SAVE = True
STAGE1_CALIB_JSON = Path("stage1_container_accel_calibration.json")
STAGE2_CALIB_JSON = Path("stage2_shoe_mounted_session_calibration.json")

# Runtime state
rx_buffer = ""
recording_active = False          # True from start command until CSV_END/finalize/cancel
csv_started = False
recording_done = False
waiting_for_dump = False

temp_path: Optional[Path] = None
keep_path: Optional[Path] = None

bad_line_count = 0
byte_count = 0
stat_t0 = time.monotonic()

first_t_ms: Optional[int] = None
last_saved_t_ms: Optional[int] = None
stream_stopped_line: Optional[str] = None
csv_ended_event: Optional[asyncio.Event] = None

received_rows: Dict[int, List[str]] = {}
duplicate_seq_count = 0
out_of_order_count = 0
max_seq_seen: Optional[int] = None
reported_sample_count: Optional[int] = None

active_chunk_start: Optional[int] = None
active_chunk_end: Optional[int] = None
chunk_count = 0
nack_count = 0
ack_count = 0

ack_queue: Optional[asyncio.Queue] = None
write_lock: Optional[asyncio.Lock] = None
event_loop: Optional[asyncio.AbstractEventLoop] = None


def is_valid_sample(line: str) -> bool:
    parts = line.split(",")
    if len(parts) != EXPECTED_COLUMNS:
        return False
    try:
        int(parts[0])  # seq
        int(parts[1])  # t_ms
        for v in parts[2:]:
            float(v)
        return True
    except ValueError:
        return False


def compact_ranges(values):
    values = sorted(values)
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


def parse_record_seconds(cmd: str) -> int:
    cmd = cmd.strip().lower()
    if cmd == "r":
        return 10
    if cmd.startswith("r "):
        return int(cmd.split()[1])
    if cmd.startswith("r") and len(cmd) > 1:
        return int(cmd[1:])
    raise ValueError


def open_temp_csv() -> Path:
    global temp_path
    SAVE_DIR.mkdir(exist_ok=True)
    temp_path = SAVE_DIR / datetime.now().strftime("imu_TEMP_%Y%m%d_%H%M%S.csv")
    # Do not write rows yet. With ACK/NACK retransmits, rows can arrive more than once.
    # We write one clean, de-duplicated, seq-sorted CSV at finalize.
    temp_path.write_text(CSV_HEADER + "\n", encoding="utf-8")
    return temp_path


def delete_temp_file() -> None:
    global temp_path, keep_path
    if temp_path is not None and temp_path.exists():
        temp_path.unlink()
    temp_path = None
    keep_path = None


def reset_recording_state_for_new_run() -> None:
    global rx_buffer, recording_active, csv_started, recording_done, waiting_for_dump
    global bad_line_count, byte_count, stat_t0
    global first_t_ms, last_saved_t_ms, stream_stopped_line, csv_ended_event
    global received_rows, duplicate_seq_count, out_of_order_count, max_seq_seen, reported_sample_count
    global active_chunk_start, active_chunk_end, chunk_count, nack_count, ack_count

    rx_buffer = ""
    recording_active = True
    csv_started = False
    recording_done = False
    waiting_for_dump = False

    bad_line_count = 0
    byte_count = 0
    stat_t0 = time.monotonic()

    first_t_ms = None
    last_saved_t_ms = None
    stream_stopped_line = None
    csv_ended_event = asyncio.Event()

    received_rows = {}
    duplicate_seq_count = 0
    out_of_order_count = 0
    max_seq_seen = None
    reported_sample_count = None

    active_chunk_start = None
    active_chunk_end = None
    chunk_count = 0
    nack_count = 0
    ack_count = 0


def extract_xiao_reported_samples(line: Optional[str]) -> Optional[int]:
    if not line:
        return None
    match = re.search(r"STREAM_STOPPED,(\d+) samples", line)
    if not match:
        return None
    return int(match.group(1))


def parse_chunk_begin(line: str):
    # CHUNK_BEGIN,start,end_exclusive
    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def parse_chunk_end(line: str):
    # CHUNK_END,start,end_exclusive
    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def queue_ble_command(cmd: str) -> None:
    """Safe from Bleak notification callback: queues ACK/NACK to async writer task."""
    if ack_queue is None:
        return
    if event_loop is not None and event_loop.is_running():
        event_loop.call_soon_threadsafe(ack_queue.put_nowait, cmd)


def write_clean_csv() -> None:
    if temp_path is None:
        return
    with temp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER.split(","))
        for seq in sorted(received_rows):
            writer.writerow(received_rows[seq])


def finalize_recording_file(reason: str = "CSV_END") -> None:
    global recording_active, csv_started, recording_done, waiting_for_dump, keep_path, temp_path

    if recording_done:
        return

    write_clean_csv()

    recording_active = False
    csv_started = False
    waiting_for_dump = False
    recording_done = True
    keep_path = temp_path

    sample_count = len(received_rows)

    duration_ms = None
    effective_hz = None
    if first_t_ms is not None and last_saved_t_ms is not None and sample_count > 1:
        duration_ms = last_saved_t_ms - first_t_ms
        if duration_ms > 0:
            effective_hz = (sample_count - 1) / (duration_ms / 1000.0)

    print(f"\nRecording finished by {reason}: {sample_count} unique valid samples, {bad_line_count} bad/clipped lines ignored.")

    reported = reported_sample_count
    if stream_stopped_line:
        print(stream_stopped_line)
        if reported is None:
            reported = extract_xiao_reported_samples(stream_stopped_line)
        if reported is not None and reported != sample_count:
            print(f"WARNING: XIAO reported {reported} samples, but Python saved {sample_count}.")

    if reported is not None:
        expected_sequences = set(range(reported))
    elif received_rows:
        expected_sequences = set(range(min(received_rows), max(received_rows) + 1))
    else:
        expected_sequences = set()

    missing_sequences = sorted(expected_sequences - set(received_rows.keys()))
    if missing_sequences:
        print(f"SEQ CHECK: FAIL | missing {len(missing_sequences)} seq value(s): {compact_ranges(missing_sequences)}")
    else:
        print("SEQ CHECK: PASS | no missing sequence values detected.")

    print(f"ACK STATS: chunks={chunk_count}, ACKs={ack_count}, NACKs={nack_count}, duplicates_seen={duplicate_seq_count}, out_of_order={out_of_order_count}")

    if effective_hz is not None:
        print(f"IMU timestamp span: {duration_ms} ms | effective sample rate: {effective_hz:.1f} Hz")

    print(f"Temporary CSV: {keep_path}")
    print("Type 's' to keep/rename CSV, or 'd' to delete it.")

    if csv_ended_event is not None and not csv_ended_event.is_set():
        csv_ended_event.set()


def handle_notify(sender, data: bytearray) -> None:
    global rx_buffer, csv_started, bad_line_count, byte_count, stat_t0
    global first_t_ms, last_saved_t_ms, stream_stopped_line, reported_sample_count
    global duplicate_seq_count, out_of_order_count, max_seq_seen
    global active_chunk_start, active_chunk_end, chunk_count, nack_count, ack_count

    byte_count += len(data)
    now = time.monotonic()
    if now - stat_t0 >= 1.0:
        if recording_active:
            phase = "dump" if csv_started else "record"
            print(f"\rBLE: {byte_count:5d} B/s | unique saved: {len(received_rows)} | phase: {phase} | ACK={ack_count} NACK={nack_count}", end="", flush=True)
        byte_count = 0
        stat_t0 = now

    rx_buffer += data.decode("utf-8", errors="replace")

    while "\n" in rx_buffer:
        line, rx_buffer = rx_buffer.split("\n", 1)
        line = line.rstrip("\r").strip()
        if not line:
            continue

        if line == "CSV_BEGIN":
            csv_started = True
            continue

        if line == CSV_HEADER:
            continue

        if line.startswith("DUMP_INFO"):
            print("\n" + line)
            continue

        if line.startswith("STREAM_STARTED"):
            print("\n" + line)
            continue

        if line.startswith("CHUNK_BEGIN"):
            parsed = parse_chunk_begin(line)
            if parsed is None:
                bad_line_count += 1
                continue
            active_chunk_start, active_chunk_end = parsed
            chunk_count += 1
            continue

        if line.startswith("CHUNK_END"):
            parsed = parse_chunk_end(line)
            if parsed is None:
                bad_line_count += 1
                continue
            end_start, end_exclusive = parsed

            # Verify the exact chunk range using the de-duplicated received row dictionary.
            expected = set(range(end_start, end_exclusive))
            missing = sorted(expected - set(received_rows.keys()))
            if missing:
                nack_count += 1
                print(f"\nChunk {end_start}-{end_exclusive - 1}: missing {compact_ranges(missing)} -> NACK,{end_start}")
                queue_ble_command(f"NACK,{end_start}")
            else:
                ack_count += 1
                queue_ble_command(f"ACK,{end_exclusive}")
            continue

        if line.startswith("STREAM_STOPPED"):
            stream_stopped_line = line
            reported_sample_count = extract_xiao_reported_samples(line)
            continue

        if line == "CSV_END":
            finalize_recording_file("CSV_END")
            continue

        if line.startswith("STATUS") or line.startswith("ERR") or line.startswith("CANCELLED") or line.startswith("RETRY"):
            print("\n" + line)
            continue

        if line.startswith("RX_CMD"):
            continue

        if recording_active and csv_started:
            if is_valid_sample(line):
                parts = line.split(",")
                seq = int(parts[0])
                t_ms = int(parts[1])

                if first_t_ms is None:
                    first_t_ms = t_ms

                if seq in received_rows:
                    duplicate_seq_count += 1
                else:
                    received_rows[seq] = parts

                if max_seq_seen is not None and seq < max_seq_seen:
                    out_of_order_count += 1
                if max_seq_seen is None or seq > max_seq_seen:
                    max_seq_seen = seq

                if last_saved_t_ms is None or seq >= max(received_rows.keys()):
                    last_saved_t_ms = t_ms
            else:
                bad_line_count += 1
        elif not recording_active:
            print(line)
        # During RAM recording phase, ignore metadata that is not CSV.


def _load_json_file(path: Path, label: str) -> Optional[dict]:
    if not path.exists():
        print(f"CALIBRATION WARNING: {label} file not found: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"CALIBRATION WARNING: could not read {label} file {path}: {e}")
        return None


def apply_stage1_stage2_calibration_to_csv(raw_csv_path: Path, calibrated_csv_path: Path) -> bool:
    """Create a calibrated copy of a raw XIAO IMU CSV.

    The raw CSV is preserved unchanged.

    Stage 1 correction:
        ax_g = (ax_raw_g - accel_offset_g.ax) * accel_scale_factor.ax
        ay_g = (ay_raw_g - accel_offset_g.ay) * accel_scale_factor.ay
        az_g = (az_raw_g - accel_offset_g.az) * accel_scale_factor.az

    Stage 2 correction:
        gx_dps = gx_raw_dps - gyro_bias_dps.gx
        gy_dps = gy_raw_dps - gyro_bias_dps.gy
        gz_dps = gz_raw_dps - gyro_bias_dps.gz
    """
    import pandas as pd

    stage1 = _load_json_file(STAGE1_CALIB_JSON, "Stage 1 accel calibration")
    stage2 = _load_json_file(STAGE2_CALIB_JSON, "Stage 2 gyro calibration")

    if stage1 is None or stage2 is None:
        print("CALIBRATION WARNING: calibrated CSV was not created because calibration JSON is missing.")
        return False

    required_stage1 = ["accel_offset_g", "accel_scale_factor"]
    required_stage2 = ["gyro_bias_dps"]

    if any(k not in stage1 for k in required_stage1):
        print("CALIBRATION WARNING: Stage 1 JSON does not contain accel_offset_g and accel_scale_factor.")
        return False

    if any(k not in stage2 for k in required_stage2):
        print("CALIBRATION WARNING: Stage 2 JSON does not contain gyro_bias_dps.")
        return False

    df = pd.read_csv(raw_csv_path)
    expected = CSV_HEADER.split(",")
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        print(f"CALIBRATION WARNING: raw CSV missing columns: {missing_cols}")
        return False

    # Preserve raw values.
    df["ax_raw_g"] = df["ax_g"]
    df["ay_raw_g"] = df["ay_g"]
    df["az_raw_g"] = df["az_g"]
    df["gx_raw_dps"] = df["gx_dps"]
    df["gy_raw_dps"] = df["gy_dps"]
    df["gz_raw_dps"] = df["gz_dps"]

    off = stage1["accel_offset_g"]
    scale = stage1["accel_scale_factor"]
    gb = stage2["gyro_bias_dps"]

    # Apply Stage 1 accelerometer correction.
    df["ax_g"] = (df["ax_g"].astype(float) - float(off["ax"])) * float(scale["ax"])
    df["ay_g"] = (df["ay_g"].astype(float) - float(off["ay"])) * float(scale["ay"])
    df["az_g"] = (df["az_g"].astype(float) - float(off["az"])) * float(scale["az"])

    # Apply Stage 2 gyroscope bias correction.
    df["gx_dps"] = df["gx_dps"].astype(float) - float(gb["gx"])
    df["gy_dps"] = df["gy_dps"].astype(float) - float(gb["gy"])
    df["gz_dps"] = df["gz_dps"].astype(float) - float(gb["gz"])

    calibrated_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(calibrated_csv_path, index=False)

    print("Calibration applied:")
    print(f"  Stage 1 accel: {STAGE1_CALIB_JSON}")
    print(f"  Stage 2 gyro : {STAGE2_CALIB_JSON}")
    print(f"  Calibrated CSV: {calibrated_csv_path}")
    print("  Raw columns preserved: ax_raw_g, ay_raw_g, az_raw_g, gx_raw_dps, gy_raw_dps, gz_raw_dps")
    return True


async def find_device():
    while True:
        print("Scanning...")
        devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT)
        for d in devices:
            if d.name == DEVICE_NAME:
                print(f"Found {d.name} [{d.address}]")
                return d
        print("Device not found. Retrying...\n")
        await asyncio.sleep(RETRY_DELAY)


async def send_cmd(client: BleakClient, cmd: str, verbose: bool = True) -> None:
    global write_lock
    msg = (cmd.strip() + "\n").encode("utf-8")
    if write_lock is None:
        await client.write_gatt_char(NUS_RX_UUID, msg, response=False)
    else:
        async with write_lock:
            await client.write_gatt_char(NUS_RX_UUID, msg, response=False)
    if verbose:
        print(f"\nSent command: {cmd.strip()}")


async def ack_sender_loop(client: BleakClient) -> None:
    while True:
        cmd = await ack_queue.get()
        if cmd is None:
            return
        await send_cmd(client, cmd, verbose=False)


async def countdown_stop_and_wait_for_dump(client: BleakClient, seconds: int) -> None:
    global waiting_for_dump

    start = time.monotonic()
    last_print = None

    while recording_active:
        elapsed = time.monotonic() - start
        remaining = max(0, seconds - int(elapsed))
        if remaining != last_print:
            print(f"\rRecording on XIAO RAM... {remaining:>3}s left", end="", flush=True)
            last_print = remaining
        if elapsed >= seconds:
            break
        await asyncio.sleep(0.05)

    if recording_active:
        waiting_for_dump = True
        await send_cmd(client, "stop")
        print("Waiting for chunked CSV dump + CSV_END from XIAO...")
        try:
            if csv_ended_event is not None:
                await asyncio.wait_for(csv_ended_event.wait(), timeout=DUMP_TIMEOUT_S)
        except asyncio.TimeoutError:
            print("\nTimeout waiting for CSV_END. Writing whatever complete rows arrived.")
            finalize_recording_file("TIMEOUT")


async def input_loop(client: BleakClient) -> None:
    global recording_active, recording_done, waiting_for_dump, csv_started
    global temp_path, keep_path

    print("\nCommands:")
    print("  r 10 / r10 = record 10 seconds on XIAO RAM, then chunked ACK CSV dump over BLE")
    print("  r          = record 10 seconds default")
    print("  status     = ask XIAO status")
    print("  s          = keep completed temp CSV as RAW and auto-create CALIBRATED copy if JSON files exist")
    print("  d          = delete completed temp CSV")
    print("  c          = cancel active recording/dump and delete temp CSV")
    print("  q          = quit\n")

    while True:
        cmd = await asyncio.to_thread(input, "> ")
        cmd = cmd.strip().lower()
        if not cmd:
            continue

        if cmd == "q":
            if recording_active:
                await send_cmd(client, "cancel")
                delete_temp_file()
            print("Exiting...")
            return

        if cmd == "status":
            await send_cmd(client, "status")
            continue

        if cmd.startswith("r"):
            if recording_active:
                print("Already recording/dumping. Press 'c' to cancel first.")
                continue
            if recording_done and keep_path is not None:
                print("Previous recording waiting. Type 's' to keep or 'd' to delete first.")
                continue
            try:
                seconds = parse_record_seconds(cmd)
                if seconds <= 0:
                    raise ValueError
            except (IndexError, ValueError):
                print("Format: r 10, r10, or r")
                continue

            if seconds > 60:
                print("This Arduino script buffers up to 60 seconds at 100 Hz. Use 60 or less.")
                continue

            reset_recording_state_for_new_run()
            path = open_temp_csv()
            print(f"Recording {seconds}s on XIAO RAM. CSV dump target: {path}")
            await send_cmd(client, "start")
            asyncio.create_task(countdown_stop_and_wait_for_dump(client, seconds))
            continue

        if cmd == "c":
            if recording_active:
                await send_cmd(client, "cancel")
                delete_temp_file()
                recording_active = False
                waiting_for_dump = False
                csv_started = False
                recording_done = False
                print("\nCancelled. Temp CSV deleted.")
            else:
                print("Not currently recording.")
            continue

        if cmd == "s":
            if recording_active:
                print("Recording/dump still running. Wait, or press 'c' to cancel.")
                continue
            if not recording_done or keep_path is None or not keep_path.exists():
                print("No completed recording to keep.")
                continue
            final_path = SAVE_DIR / datetime.now().strftime("imu_%Y%m%d_%H%M%S_RAW.csv")
            os.replace(keep_path, final_path)
            print(f"Kept RAW CSV: {final_path}")

            if APPLY_CALIBRATION_ON_SAVE:
                calibrated_path = final_path.with_name(final_path.stem.replace("_RAW", "_CALIBRATED") + final_path.suffix)
                apply_stage1_stage2_calibration_to_csv(final_path, calibrated_path)

            keep_path = None
            temp_path = None
            recording_done = False
            continue

        if cmd == "d":
            if recording_active:
                print("Recording/dump still running. Use 'c' to cancel.")
                continue
            if keep_path is not None and keep_path.exists():
                keep_path.unlink()
                print("Deleted completed temp CSV.")
            else:
                print("No completed recording to delete.")
            keep_path = None
            temp_path = None
            recording_done = False
            continue

        print("Unknown command. Use r 10, r10, r, status, s, d, c, or q.")


async def connect_and_run(device) -> None:
    global rx_buffer, ack_queue, write_lock, event_loop
    rx_buffer = ""
    event_loop = asyncio.get_running_loop()
    ack_queue = asyncio.Queue()
    write_lock = asyncio.Lock()
    disconnected_event = asyncio.Event()

    def on_disconnect(client):
        print("\nDisconnected.")
        disconnected_event.set()

    async with BleakClient(device, disconnected_callback=on_disconnect) as client:
        print(f"Connected to {device.name} [{device.address}]")
        await client.start_notify(NUS_TX_UUID, handle_notify)

        ack_task = asyncio.create_task(ack_sender_loop(client))
        input_task = asyncio.create_task(input_loop(client))
        disconnect_task = asyncio.create_task(disconnected_event.wait())

        done, pending = await asyncio.wait(
            {input_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if ack_queue is not None:
            await ack_queue.put(None)
        await ack_task

        try:
            await client.stop_notify(NUS_TX_UUID)
        except Exception:
            pass


async def main() -> None:
    while True:
        device = await find_device()
        try:
            await connect_and_run(device)
        except Exception as e:
            print(f"Connection error: {e}")
        await asyncio.sleep(RETRY_DELAY)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
