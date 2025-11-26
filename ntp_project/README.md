# AI-based Network Traffic Prediction (Mini Project)

This is a ready-to-run mini project scaffold that demonstrates:
- synthetic network-traffic CSV generator
- preprocessing script
- baseline ARIMA script
- LSTM training script (TensorFlow)
- Streamlit dashboard to visualize predictions vs actuals

## Structure
- data/sample_traffic.csv        : synthetic dataset (1-minute intervals)
- src/preprocess.py              : preprocessing & feature generation
- src/train_lstm.py              : LSTM training script (saves model to models/)
- src/baseline_arima.py          : ARIMA baseline script
- src/app_streamlit.py           : Streamlit app to visualize and compare
- requirements.txt               : Python deps

## Quick start (local)
1. Create and activate a virtualenv (optional)
2. `pip install -r requirements.txt`
3. Generate data and preprocess:
   `python src/preprocess.py --generate`
4. Train LSTM (will create models/lstm_model.h5):
   `python src/train_lstm.py`
5. Run Streamlit dashboard:
   `streamlit run src/app_streamlit.py`

Note: training parameters are small by default for quick runs on laptop. For better results, increase epochs and dataset length.