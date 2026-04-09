import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "XIAO_BATTERY"
NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

SCAN_TIMEOUT = 5.0
RETRY_DELAY = 2.0

rx_buffer = ""
last_acc = None
last_gyr = None


def handle_notify(sender, data):
    global rx_buffer, last_acc, last_gyr

    rx_buffer += data.decode("utf-8", errors="replace")

    while "\n" in rx_buffer:
        line, rx_buffer = rx_buffer.split("\n", 1)
        line = line.rstrip("\r").strip()

        if not line:
            continue

        if line.startswith("A,"):
            last_acc = line
        elif line.startswith("G,"):
            last_gyr = line
        else:
            print(line)
            continue

        if last_acc and last_gyr:
            print(f"{last_acc} | {last_gyr}")
            last_acc = None
            last_gyr = None


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


async def connect_and_listen(device):
    disconnected_event = asyncio.Event()

    def on_disconnect(client):
        print("\nDisconnected. Waiting to reconnect...\n")
        disconnected_event.set()

    global rx_buffer
    rx_buffer = ""

    try:
        async with BleakClient(device, disconnected_callback=on_disconnect) as client:
            print(f"Connected to {device.name} [{device.address}]")
            await client.start_notify(NUS_TX_UUID, handle_notify)
            print("Listening...\n")
            await disconnected_event.wait()

    except Exception as e:
        print(f"Connection error: {e}\n")


async def main():
    while True:
        device = await find_device()
        await connect_and_listen(device)
        await asyncio.sleep(RETRY_DELAY)


if __name__ == "__main__":
    asyncio.run(main())