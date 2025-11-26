import pandas as pd
import statsmodels.api as sm
from pathlib import Path

if __name__ == '__main__':
    # Load dataset from correct path
    df = pd.read_csv('data/sample_traffic.csv', parse_dates=['timestamp'], index_col='timestamp')

    # Resample to 1-minute intervals
    ts = df['bytes_in'].resample('1min').mean().ffill()

    # Simple ARIMA(5,1,0) model
    print("⏳ Training ARIMA model...")
    model = sm.tsa.ARIMA(ts, order=(5,1,0))
    res = model.fit()

    # Forecast next 60 minutes
    fc = res.forecast(60)

    # Save output
    out = Path('models')
    out.mkdir(exist_ok=True)
    fc.to_csv(out / 'arima_forecast.csv')

    print("✅ ARIMA forecast saved to models/arima_forecast.csv")

# # src/baseline_arima.py
# import argparse
# import pandas as pd
# import statsmodels.api as sm
# from pathlib import Path

# def run(col="Aramghar", infile="data/sample_traffic_hyd.csv", outdir="models"):
#     df = pd.read_csv(infile, parse_dates=['timestamp'], index_col='timestamp')
#     if col not in df.columns:
#         raise ValueError(f"{col} not found.")
#     ts = df[col].resample('1min').mean().ffill()
#     model = sm.tsa.ARIMA(ts, order=(5,1,0))
#     res = model.fit()
#     fc = res.forecast(120)  # 2 hours default
#     outdir = Path(outdir)
#     outdir.mkdir(exist_ok=True)
#     fc.to_csv(outdir / f"arima_{col}.csv")
#     print(f"ARIMA forecast saved at {outdir / f'arima_{col}.csv'}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument('--col', default='Aramghar')
#     args = p.parse_args()
#     run(args.col)
