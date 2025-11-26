import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib


def generate_sample(path):
    """Generate a synthetic sample dataset if not provided."""
    timestamps = pd.date_range("2025-01-01", periods=1440, freq="1min")  # 1 day, per minute
    bytes_in = np.random.randint(100, 1000, size=len(timestamps))
    bytes_out = np.random.randint(50, 900, size=len(timestamps))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out
    })
    df.to_csv(path, index=False)
    print(f"✅ Sample dataset generated at {path}")


def preprocess(infile, outdir, lookback=10):
    """Preprocess traffic data into training sequences."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(infile, parse_dates=['timestamp'])

    # Resample to 1-minute intervals
    df = df.set_index('timestamp').resample('1min').mean().ffill()

    # Use bytes_in as target
    values = df['bytes_in'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled) - lookback):
        X.append(scaled[i:i+lookback].flatten())
        y.append(scaled[i+lookback, 0])

    X = np.array(X)
    y = np.array(y)

    # Save processed data
    np.save(outdir / 'X.npy', X)
    np.save(outdir / 'y.npy', y)
    joblib.dump(scaler, outdir / 'scaler.pkl')

    print(f"✅ Preprocessing complete!")
    print(f"   X shape = {X.shape}")
    print(f"   y shape = {y.shape}")
    print(f"   Data saved to {outdir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--generate', action='store_true', help='Generate sample dataset')
    # p.add_argument('--infile', default='data/sample_traffic.csv')
    p.add_argument('--infile', default='data/live_traffic.csv')

    p.add_argument('--outdir', default='data/processed')
    p.add_argument('--lookback', type=int, default=10)
    args = p.parse_args()

    infile = Path(args.infile)

    if args.generate or not infile.exists():
        generate_sample(infile)

    preprocess(infile, args.outdir, args.lookback)




# # src/preprocess.py
# import argparse
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import joblib

# def preprocess(infile="data/sample_traffic_hyd.csv", col="aramghar", outdir="data/processed", lookback=10):
#     infile = Path(infile)
#     df = pd.read_csv(infile, parse_dates=['timestamp'])
#     df = df.set_index('timestamp').resample('1min').mean().ffill()

#     if col not in df.columns:
#         raise ValueError(f"Column '{col}' not found in {infile}. Available: {list(df.columns)}")

#     values = df[col].values.reshape(-1, 1)
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(values)

#     X, y = [], []
#     for i in range(len(scaled) - lookback):
#         X.append(scaled[i:i+lookback])
#         y.append(scaled[i+lookback, 0])

#     X = np.array(X)   # (samples, timesteps, 1)
#     y = np.array(y)

#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     # Save using column name so different locations don't overwrite each other
#     np.save(outdir / f'X_{col}.npy', X)
#     np.save(outdir / f'y_{col}.npy', y)
#     joblib.dump(scaler, outdir / f'scaler_{col}.pkl')
#     print(f"Saved processed for {col}: X.shape={X.shape}, y.shape={y.shape} in {outdir}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument('--infile', default='data/sample_traffic_hyd.csv')
#     p.add_argument('--col', default='Aramghar')
#     p.add_argument('--outdir', default='data/processed')
#     p.add_argument('--lookback', type=int, default=10)
#     args = p.parse_args()
#     preprocess(args.infile, args.col, args.outdir, args.lookback)
