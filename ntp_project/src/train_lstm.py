
# import numpy as np
# from pathlib import Path
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import joblib

# def build_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(64, input_shape=input_shape))
#     model.add(Dropout(0.1))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

# if __name__ == '__main__':
#     proc = Path('../data/processed')
#     X = np.load(proc / 'X.npy')
#     y = np.load(proc / 'y.npy')
#     # reshape X to (samples, timesteps, features)
#     lookback = 10
#     X = X.reshape((X.shape[0], lookback, 1))
#     model = build_model((lookback,1))
#     es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
#     model.fit(X, y, epochs=20, batch_size=32, callbacks=[es])
#     outdir = Path('../models')
#     outdir.mkdir(exist_ok=True)
#     model.save(outdir / 'lstm_model.h5')
#     joblib.dump('trained', outdir / 'train.flag')
#     print('Model trained and saved to models/lstm_model.h5')


import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Settings
LOOKBACK = 10
FORECAST_STEPS = 120   # 🔹 Predict 2 hours (120 mins)

def load_data():
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")
    scaler = joblib.load("data/processed/scaler.pkl")
    return X, y, scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def recursive_forecast(model, last_sequence, steps, scaler):
    """Predict multiple future steps by feeding predictions back"""
    preds = []
    seq = last_sequence.copy()

    for _ in range(steps):
        x_input = seq.reshape(1, LOOKBACK)
        yhat = model.predict(x_input, verbose=0)[0][0]
        preds.append(yhat)

        # slide window
        seq = np.append(seq[1:], yhat)

    # inverse transform (back to original scale)
    preds = np.array(preds).reshape(-1, 1)
    preds = scaler.inverse_transform(preds)
    return preds.flatten()

if __name__ == "__main__":
    print("📥 Loading data...")
    X, y, scaler = load_data()

    # Reshape for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], LOOKBACK, 1))

    print("🧠 Building model...")
    model = build_model((LOOKBACK, 1))

    print("🚀 Training model...")
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=20, batch_size=32, verbose=1, callbacks=[es])

    # Save model
    outdir = Path("models")
    outdir.mkdir(exist_ok=True)
    model.save(outdir / "lstm_model.h5")
    print("✅ LSTM model saved to models/lstm_model.h5")

    # Recursive forecast for 2 hours
    last_seq = X[-1].flatten()
    preds = recursive_forecast(model, last_seq, FORECAST_STEPS, scaler)

    # Save forecast
    future_index = pd.date_range(start=pd.Timestamp.now(), periods=FORECAST_STEPS, freq="1min")
    forecast_df = pd.DataFrame({"timestamp": future_index, "lstm_forecast": preds})
    forecast_df.to_csv(outdir / "lstm_forecast.csv", index=False)

    print(f"✅ LSTM forecast (next {FORECAST_STEPS} mins) saved to models/lstm_forecast.csv")




# # src/train_lstm.py
# import argparse
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Input
# from tensorflow.keras.callbacks import EarlyStopping
# from pathlib import Path

# def train(col="Aramghar", processed_dir="data/processed", models_dir="models", epochs=20):
#     proc = Path(processed_dir)
#     Xp = proc / f"X_{col}.npy"
#     yp = proc / f"y_{col}.npy"
#     if not Xp.exists():
#         raise FileNotFoundError(f"Processed files not found for {col}. Run preprocess.py --col {col}")

#     X = np.load(Xp)
#     y = np.load(yp)
#     print("Data shapes:", X.shape, y.shape)
#     if X.shape[0] < 10:
#         raise ValueError("Not enough samples. Capture more data before training.")

#     timesteps = X.shape[1]
#     model = Sequential([
#         Input(shape=(timesteps, 1)),
#         LSTM(64, return_sequences=False),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse')

#     es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

#     model.fit(X, y, epochs=epochs, batch_size=32, verbose=1, callbacks=[es])

#     models_dir = Path(models_dir)
#     models_dir.mkdir(parents=True, exist_ok=True)
#     out_file = models_dir / f"lstm_{col}.keras"
#     model.save(out_file)   # Keras v3 .keras format
#     print(f"Saved model to {out_file}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument('--col', default='Aramghar')
#     p.add_argument('--epochs', type=int, default=20)
#     args = p.parse_args()
#     train(args.col, epochs=args.epochs)
