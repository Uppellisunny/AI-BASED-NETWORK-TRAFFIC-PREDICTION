import pandas as pd
from pathlib import Path
import numpy as np

DATA_PATH = Path("data/sample_traffic.csv")
OUTPUT_PATH = Path("data/sample_traffic_hyd.csv")

df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.set_index('timestamp').resample('1T').mean().ffill()

# Create synthetic traffic for Hyderabad sub-locations
np.random.seed(42)
df['aramghar'] = df['bytes_in'] * np.random.uniform(0.8, 1.2, len(df))
df['panjagutta'] = df['bytes_in'] * np.random.uniform(0.7, 1.3, len(df))
df['secunderabad'] = df['bytes_in'] * np.random.uniform(0.9, 1.1, len(df))
df['uppal'] = df['bytes_in'] * np.random.uniform(0.85, 1.25, len(df))
df['lbnagar'] = df['bytes_in'] * np.random.uniform(0.8, 1.15, len(df))

df.to_csv(OUTPUT_PATH)
print(f"✅ New dataset with Hyderabad locations saved at {OUTPUT_PATH}")


# # src/prepare_locations.py
# import pandas as pd
# import numpy as np
# from pathlib import Path

# OUT = Path("../data") if Path(".").resolve().name == "src" else Path("./data")
# OUT.mkdir(parents=True, exist_ok=True)
# OUTFILE = OUT / "sample_traffic_hyd.csv"

# def generate_15_locations():
#     periods = 24*60  # 1 day minute-level
#     idx = pd.date_range("2025-01-01", periods=periods, freq="1min")
#     base = 5000 + 2000*np.sin(np.linspace(0, 4*np.pi, periods))  # diurnal pattern
#     rng = np.random.RandomState(42)

#     locations = {
#         "Aramghar": 0.9,
#         "Panjagutta": 1.05,
#         "Secunderabad": 1.2,
#         "Uppal": 1.0,
#         "LB_Nagar": 0.95,
#         "Kukatpally": 1.02,
#         "Madhapur": 1.4,
#         "Banjara_Hills": 1.1,
#         "Ameerpet": 0.98,
#         "Koti": 0.88,
#         "Dilsukhnagar": 0.92,
#         "Mehdipatnam": 0.97,
#         "Gachibowli": 1.45,
#         "Hitech_City": 1.5,
#         "Begumpet": 0.85
#     }

#     df = pd.DataFrame(index=idx)
#     for name, factor in locations.items():
#         noise = rng.normal(0, 300, periods)
#         df[name] = (base * factor + noise).clip(min=0).astype(int)
#     df.reset_index(inplace=True)
#     df = df.rename(columns={"index": "timestamp"})
#     df.to_csv(OUTFILE, index=False)
#     print(f"Generated sample dataset at {OUTFILE}")

# if __name__ == "__main__":
#     generate_15_locations()
