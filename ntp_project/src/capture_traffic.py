import psutil
import time
import csv
import datetime
from pathlib import Path

data_file = Path("data/live_traffic.csv")

# Create file with header if not exists
if not data_file.exists():
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "bytes_in", "bytes_out"])

print("📡 Capturing live traffic... Press CTRL+C to stop")

try:
    while True:
        net1 = psutil.net_io_counters()
        time.sleep(1)  # capture every 1 sec
        net2 = psutil.net_io_counters()

        bytes_in = net2.bytes_recv - net1.bytes_recv
        bytes_out = net2.bytes_sent - net1.bytes_sent

        with open(data_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.datetime.now(), bytes_in, bytes_out])

        print(f"{datetime.datetime.now()} | In: {bytes_in} bytes | Out: {bytes_out} bytes")

except KeyboardInterrupt:
    print("\n✅ Traffic capture stopped. Data saved to data/live_traffic.csv")




# # src/capture_traffic.py
# import psutil, time, csv, datetime
# from pathlib import Path

# OUT = Path("../data") if Path(".").resolve().name == "src" else Path("./data")
# OUT.mkdir(parents=True, exist_ok=True)
# OUTFILE = OUT / "live_local_traffic.csv"

# if not OUTFILE.exists():
#     with open(OUTFILE, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["timestamp", "bytes_in", "bytes_out"])

# print("Capturing local network traffic (press Ctrl+C to stop)...")
# try:
#     while True:
#         a = psutil.net_io_counters()
#         time.sleep(60)
#         b = psutil.net_io_counters()
#         bytes_in = b.bytes_recv - a.bytes_recv
#         bytes_out = b.bytes_sent - a.bytes_sent
#         with open(OUTFILE, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([datetime.datetime.now(), bytes_in, bytes_out])
#         print(f"{datetime.datetime.now()} | in={bytes_in} out={bytes_out}")
# except KeyboardInterrupt:
#     print("Stopped capture.")
