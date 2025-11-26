# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Correct dataset path
# data_path = "data/sample_traffic.csv"

# st.title("📊 AI-based Network Traffic Prediction")

# # Load dataset
# df = pd.read_csv(data_path, parse_dates=['timestamp'])
# df = df.set_index('timestamp').resample('1min').mean().ffill()

# st.subheader("Raw Traffic Data (bytes_in)")
# st.line_chart(df['bytes_in'])

# # Show stats
# st.subheader("Dataset Summary")
# st.write(df.describe())

# # Load ARIMA results if available
# arima_path = Path("models/arima_forecast.csv")
# if arima_path.exists():
#     st.subheader("📈 ARIMA Forecast (Next 60 mins)")
#     arima_df = pd.read_csv(arima_path, index_col=0)
#     st.line_chart(arima_df)
# else:
#     st.info("ARIMA forecast not found. Run `python src/baseline_arima.py` first.")

# # Load LSTM results if available
# lstm_path = Path("models/lstm_model.h5")
# if lstm_path.exists():
#     st.subheader("🤖 LSTM Model Status")
#     st.success("LSTM model is trained and saved in models/lstm_model.h5")
# else:
#     st.warning("No LSTM model found. Run `python src/train_lstm.py` to train it.")





# import streamlit as st
# import requests
# import os

# # ==============================
# # CONFIG
# # ==============================
# API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"   # 🔑 Replace with your key
# SAVE_DIR = "maps"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # Define Hyderabad locations with coordinates
# LOCATIONS = {
#     "Aramghar": "17.3341,78.4018",
#     "Panjagutta": "17.4270,78.4510",
#     "Secunderabad": "17.4399,78.4983",
#     "Uppal": "17.4059,78.5591",
#     "LB Nagar": "17.3415,78.5521"
# }

# # ==============================
# # STREAMLIT UI
# # ==============================
# st.set_page_config(page_title="Hyderabad Live Traffic", layout="wide")

# st.title("🚦 Hyderabad Live Traffic Dashboard")

# location = st.selectbox("📍 Choose a location:", list(LOCATIONS.keys()))

# # ==============================
# # FETCH GOOGLE MAP IMAGE
# # ==============================
# coords = LOCATIONS[location]
# url = (
#     f"https://maps.googleapis.com/maps/api/staticmap?"
#     f"center={coords}&zoom=14&size=800x600&maptype=roadmap"
#     f"&markers=color:red%7Clabel:{location[0]}%7C{coords}"
#     f"&key={API_KEY}"
# )

# # Save image locally
# img_path = os.path.join(SAVE_DIR, f"{location}.png")
# try:
#     response = requests.get(url, timeout=10)
#     if response.status_code == 200:
#         with open(img_path, "wb") as f:
#             f.write(response.content)
#     else:
#         st.error("❌ Failed to fetch map image from Google API.")
# except Exception as e:
#     st.error(f"⚠️ Error fetching map: {e}")

# # ==============================
# # SHOW IMAGE
# # ==============================
# st.image(img_path, caption=f"Live Traffic at {location}", use_column_width=True)

# # ==============================
# # FUTURE TRAFFIC PREDICTION PLACEHOLDER
# # ==============================
# st.subheader("📊 Predicted Traffic (Next 2 Hours)")
# st.info("This section will display LSTM prediction results for traffic flow in future.")

# import streamlit as st
# import folium
# from streamlit_folium import st_folium
# import numpy as np
# import matplotlib.pyplot as plt

# # ---------------------------
# # Fake traffic dataset
# # ---------------------------
# locations = {
#     "Aramghar": {"lat": 17.3616, "lon": 78.4467},
#     "Panjagutta": {"lat": 17.4239, "lon": 78.4483},
#     "Secunderabad": {"lat": 17.4399, "lon": 78.4983},
#     "Uppal": {"lat": 17.4050, "lon": 78.5591},
#     "LB Nagar": {"lat": 17.3474, "lon": 78.5570},
#     "Kukatpally": {"lat": 17.4933, "lon": 78.3995},
#     "Madhapur": {"lat": 17.4504, "lon": 78.3911},
#     "Banjara Hills": {"lat": 17.4202, "lon": 78.4483},
#     "Ameerpet": {"lat": 17.4375, "lon": 78.4487},
#     "Koti": {"lat": 17.3840, "lon": 78.4800},
#     "Dilsukhnagar": {"lat": 17.3686, "lon": 78.5247},
#     "Mehdipatnam": {"lat": 17.3945, "lon": 78.4306},
#     "Gachibowli": {"lat": 17.4401, "lon": 78.3489},
#     "Hitech City": {"lat": 17.4435, "lon": 78.3772},
#     "Begumpet": {"lat": 17.4477, "lon": 78.4897}
# }

# # ---------------------------
# # Streamlit Layout
# # ---------------------------
# st.set_page_config(layout="wide")
# st.title("🚦 Hyderabad Live Traffic Dashboard")

# # Select location
# location = st.selectbox("Select a location", list(locations.keys()))

# # Generate fake traffic data
# traffic_value = np.random.randint(2000, 15000)  # bytes per minute
# if traffic_value < 5000:
#     color, status = "green", "🟢 Low Traffic"
# elif traffic_value < 10000:
#     color, status = "orange", "🟠 Medium Traffic"
# else:
#     color, status = "red", "🔴 Heavy Traffic"

# col1, col2 = st.columns([2, 2])

# # ---------------------------
# # Graph Section (Area Chart)
# # ---------------------------
# with col1:
#     st.subheader(f"📈 Traffic Trend at {location}")
#     times = np.arange(30)  # last 30 mins
#     traffic_load = np.random.normal(loc=traffic_value, scale=800, size=30)

#     fig, ax = plt.subplots(figsize=(6, 3))
#     ax.plot(times, traffic_load, color=color, linewidth=2)
#     ax.fill_between(times, traffic_load, color=color, alpha=0.3)

#     ax.set_title("Traffic in last 30 mins", fontsize=12)
#     ax.set_xlabel("Minutes ago")
#     ax.set_ylabel("Bytes / min")
#     ax.grid(True, linestyle="--", alpha=0.5)

#     st.pyplot(fig)

# # ---------------------------
# # Map Section
# # ---------------------------
# with col2:
#     st.subheader("🗺️ Location Map")
#     coords = locations[location]

#     map_ = folium.Map(location=[coords["lat"], coords["lon"]], zoom_start=13)
#     folium.CircleMarker(
#         [coords["lat"], coords["lon"]],
#         radius=25,
#         popup=f"{location} - {status}",
#         color=color,
#         fill=True,
#         fill_color=color,
#     ).add_to(map_)

#     st_folium(map_, width=500, height=350)

# # ---------------------------
# # Info Section
# # ---------------------------
# st.markdown("### 📌 Location Information")
# st.info(
#     f"""
#     **Location:** {location}  
#     **Traffic Status:** {status}  
#     **Current Load:** {traffic_value} bytes/min  

#     ✅ Green = Smooth traffic  
#     🟠 Orange = Moderate traffic  
#     🔴 Red = Heavy congestion
#     """
# )




# # src/app_streamlit.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# from streamlit_folium import st_folium
# import folium
# from utils import apply_ar_scenario

# # Config
# DATA_FILE = Path("data/sample_traffic_hyd.csv")
# PROCESSED_DIR = Path("data/processed")
# MODELS_DIR = Path("models")

# # Check data
# if not DATA_FILE.exists():
#     st.error("Data file not found. Run src/prepare_locations.py to generate sample data.")
#     st.stop()

# # Load dataset
# df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
# df = df.set_index('timestamp').resample('1min').mean().ffill()

# # Locations available
# locations = [c for c in df.columns if c != 'timestamp']
# # Provide a friendly name mapping (underscores -> spaces)
# locations_pretty = {loc: loc.replace('_', ' ') for loc in locations}

# st.set_page_config(layout="wide")
# st.title("AI-based Network Traffic Prediction — AR Scenario Simulator")

# # Sidebar
# st.sidebar.header("Controls")
# selected = st.sidebar.selectbox("Select location", locations, format_func=lambda x: locations_pretty[x])
# lookback = st.sidebar.slider("Model lookback (timesteps)", 5, 30, 10)
# forecast_minutes = st.sidebar.slider("Forecast horizon (minutes)", 30, 180, 120, step=30)

# st.sidebar.markdown("---")
# st.sidebar.header("AR Scenario (what-if)")
# scenario = st.sidebar.selectbox("Scenario", ["none", "bicycle", "bus", "no_signals"])
# if scenario in ["bicycle", "bus"]:
#     pct = st.sidebar.slider("Adoption (%)", 5, 80, 20)
#     param = pct / 100.0
# else:
#     param = 0.0

# # Selected series
# series = df[selected].dropna()

# # Layout: left (graph) / right (map)
# col1, col2 = st.columns([1.2, 1])

# # Left: historical + prediction graphs
# with col1:
#     st.subheader(f"Traffic at {locations_pretty[selected]}")
#     st.write(f"Latest value: **{int(series.iloc[-1])} bytes/min**")

#     # Plot historical last 120 minutes
#     hist = series.tail(120)
#     fig, ax = plt.subplots(figsize=(8, 3.5))
#     ax.plot(hist.index, hist.values, label="Historical", color='blue')
#     ax.set_title("Historical Traffic (last 120 mins)")
#     ax.set_ylabel("Bytes per minute")
#     ax.grid(alpha=0.3)
#     ax.tick_params(axis='x', rotation=45)
#     st.pyplot(fig)

#     # Load model for this location if exists
#     model_file = MODELS_DIR / f"lstm_{selected}.keras"
#     scaler_file = PROCESSED_DIR / f"scaler_{selected}.pkl"
#     preds = None
#     if model_file.exists() and scaler_file.exists():
#         model = load_model(model_file, compile=False)
#         scaler = joblib.load(scaler_file)

#         # prepare last sequence
#         # try to load X_selected if exists; else build sequence from series
#         Xfile = PROCESSED_DIR / f"X_{selected}.npy"
#         if Xfile.exists():
#             X = np.load(Xfile)
#             last_seq = X[-1].reshape(1, X.shape[1], 1)
#         else:
#             # build last_seq from raw series
#             last_vals = series.values[-lookback:].reshape(-1,1)
#             last_scaled = scaler.transform(last_vals)
#             last_seq = last_scaled.reshape(1, lookback, 1)

#         # recursive forecasting
#         preds_scaled = []
#         seq = last_seq.copy()
#         for _ in range(forecast_minutes):
#             p = model.predict(seq, verbose=0)[0][0]
#             preds_scaled.append(p)
#             seq = np.roll(seq, -1)
#             seq[0, -1, 0] = p

#         preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
#     else:
#         st.info("No trained LSTM model found for this location. You can train using src/train_lstm.py --col <location>")

#     # Also run ARIMA baseline if exists
#     arima_file = MODELS_DIR / f"arima_{selected}.csv"
#     arima_preds = None
#     if arima_file.exists():
#         arima_df = pd.read_csv(arima_file, index_col=0, parse_dates=True)
#         arima_preds = arima_df.values.flatten()[:forecast_minutes]

#     # Apply AR scenario to the LSTM prediction if present
#     if preds is not None:
#         preds_ar = apply_ar_scenario(preds, scenario, param)
#         idx_future = pd.date_range(series.index[-1], periods=forecast_minutes+1, freq='1min')[1:]
#         df_preds = pd.DataFrame({
#             "LSTM": preds,
#             "LSTM_AR": preds_ar
#         }, index=idx_future)
#         st.subheader("Predicted Traffic (LSTM)")
#         fig2, ax2 = plt.subplots(figsize=(8,3.5))
#         ax2.plot(df_preds.index, df_preds["LSTM"], label="LSTM (base)", color='green', alpha=0.6)
#         ax2.plot(df_preds.index, df_preds["LSTM_AR"], label=f"LSTM (scenario: {scenario})", color='magenta')
#         if arima_preds is not None:
#             ax2.plot(idx_future, arima_preds, label="ARIMA baseline", color='orange', alpha=0.6)
#         ax2.set_title(f"Forecast Next {forecast_minutes} mins")
#         ax2.set_ylabel("Bytes / minute")
#         ax2.legend(loc='upper left')
#         ax2.grid(alpha=0.2)
#         st.pyplot(fig2)

#         st.subheader("Scenario Summary")
#         reduction = 100.0*(1 - (np.mean(df_preds["LSTM_AR"]) / np.mean(df_preds["LSTM"])))
#         st.write(f"- Mean predicted traffic (base LSTM): **{int(np.mean(df_preds['LSTM']))} bytes/min**")
#         st.write(f"- Mean predicted traffic (scenario {scenario}): **{int(np.mean(df_preds['LSTM_AR']))} bytes/min**")
#         st.write(f"- Estimated reduction: **{reduction:.1f}%**")
#     else:
#         st.info("Predictions not available (model missing). The AR scenario will show relative effects once model predictions exist.")

# # Right: Map + location info
# with col2:
#     st.subheader("Location Map")
#     coords = (series.index, )  # placeholder

#     # Create map centered on location
#     # find lat/lon from sample file (we used names as columns with lat/lon in prepare script)
#     # here we read first row for lat/lons stored in metadata? For synthetic dataset, we know column names correspond.
#     # For simplicity, we use a small mapping (must match prepare_locations)
#     location_coords = {
#         "Aramghar": (17.3616, 78.4467),
#         "Panjagutta": (17.4239, 78.4483),
#         "Secunderabad": (17.4399, 78.4983),
#         "Uppal": (17.4050, 78.5591),
#         "LB_Nagar": (17.3474, 78.5570),
#         "Kukatpally": (17.4933, 78.3995),
#         "Madhapur": (17.4504, 78.3911),
#         "Banjara_Hills": (17.4202, 78.4483),
#         "Ameerpet": (17.4375, 78.4487),
#         "Koti": (17.3840, 78.4800),
#         "Dilsukhnagar": (17.3686, 78.5247),
#         "Mehdipatnam": (17.3945, 78.4306),
#         "Gachibowli": (17.4401, 78.3489),
#         "Hitech_City": (17.4435, 78.3772),
#         "Begumpet": (17.4477, 78.4897)
#     }
#     lat, lon = location_coords.get(selected, (17.3850, 78.4867))
#     m = folium.Map(location=[lat, lon], zoom_start=14)

#     # Show only selected location and its traffic indicator
#     latest_val = int(series.iloc[-1])
#     if latest_val > 9000:
#         color = "red"
#     elif latest_val > 6000:
#         color = "orange"
#     else:
#         color = "green"

#     folium.CircleMarker([lat, lon],
#                         radius=18,
#                         color=color, fill=True, fill_color=color, fill_opacity=0.7,
#                         popup=f"{selected}: {latest_val} bytes/min").add_to(m)

#     # For scenario: also show AR indicator (e.g., predicted scenario mean)
#     if preds is not None:
#         mean_base = int(np.mean(preds))
#         mean_scn = int(np.mean(apply_ar_scenario(preds, scenario, param)))
#         folium.map.Marker([lat, lon],
#                           icon=folium.DivIcon(html=f"""<div style="font-size:12pt">
#                             <b>Base:{mean_base}</b><br><span style="color:blue">Scenario:{mean_scn}</span></div>""")
#                          ).add_to(m)

#     st_folium(m, width=450, height=350)

#     st.subheader("Location Info")
#     st.write(f"- **Location:** {selected.replace('_',' ')}")
#     st.write(f"- **Latest measured bytes/min:** {int(series.iloc[-1])}")
#     if preds is not None:
#         st.write(f"- **Predicted (mean next {forecast_minutes} min):** {int(np.mean(preds))}")
#         st.write(f"- **Predicted under scenario '{scenario}':** {int(np.mean(apply_ar_scenario(preds, scenario, param)))}")
#     else:
#         st.info("Train a model for this location to get LSTM predictions (see scripts).")




# import streamlit as st
# import folium
# from streamlit_folium import st_folium
# import numpy as np
# import matplotlib.pyplot as plt

# # ---------------------------
# # Fake traffic dataset
# # ---------------------------
# locations = {
#     "Aramghar": {"lat": 17.3616, "lon": 78.4467},
#     "Panjagutta": {"lat": 17.4239, "lon": 78.4483},
#     "Secunderabad": {"lat": 17.4399, "lon": 78.4983},
#     "Uppal": {"lat": 17.4050, "lon": 78.5591},
#     "LB Nagar": {"lat": 17.3474, "lon": 78.5570},
#     "Kukatpally": {"lat": 17.4933, "lon": 78.3995},
#     "Madhapur": {"lat": 17.4504, "lon": 78.3911},
#     "Banjara Hills": {"lat": 17.4202, "lon": 78.4483},
#     "Ameerpet": {"lat": 17.4375, "lon": 78.4487},
#     "Koti": {"lat": 17.3840, "lon": 78.4800},
#     "Dilsukhnagar": {"lat": 17.3686, "lon": 78.5247},
#     "Mehdipatnam": {"lat": 17.3945, "lon": 78.4306},
#     "Gachibowli": {"lat": 17.4401, "lon": 78.3489},
#     "Hitech City": {"lat": 17.4435, "lon": 78.3772},
#     "Begumpet": {"lat": 17.4477, "lon": 78.4897}
# }

# # ---------------------------
# # Streamlit Layout
# # ---------------------------
# st.set_page_config(layout="wide")
# st.title("🚦 Hyderabad Live Traffic Dashboard")

# # Select location
# location = st.selectbox("Select a location", list(locations.keys()))

# # Generate fake traffic data
# traffic_value = np.random.randint(2000, 15000)  # bytes per minute
# if traffic_value < 5000:
#     color, status = "green", "🟢 Low Traffic"
# elif traffic_value < 10000:
#     color, status = "orange", "🟠 Medium Traffic"
# else:
#     color, status = "red", "🔴 Heavy Traffic"

# col1, col2 = st.columns([2, 2])

# # ---------------------------
# # Graph Section (Area Chart)
# # ---------------------------
# with col1:
#     st.subheader(f"📈 Traffic Trend at {location}")
#     times = np.arange(30)  # last 30 mins
#     traffic_load = np.random.normal(loc=traffic_value, scale=800, size=30)

#     fig, ax = plt.subplots(figsize=(6, 3))
#     ax.plot(times, traffic_load, color=color, linewidth=2)
#     ax.fill_between(times, traffic_load, color=color, alpha=0.3)

#     ax.set_title("Traffic in last 30 mins", fontsize=12)
#     ax.set_xlabel("Minutes ago")
#     ax.set_ylabel("Bytes / min")
#     ax.grid(True, linestyle="--", alpha=0.5)

#     st.pyplot(fig)

# # ---------------------------
# # Map Section
# # ---------------------------
# with col2:
#     st.subheader("🗺️ Location Map")
#     coords = locations[location]

#     map_ = folium.Map(location=[coords["lat"], coords["lon"]], zoom_start=13)
#     folium.CircleMarker(
#         [coords["lat"], coords["lon"]],
#         radius=25,
#         popup=f"{location} - {status}",
#         color=color,
#         fill=True,
#         fill_color=color,
#     ).add_to(map_)

#     st_folium(map_, width=500, height=350)

# # ---------------------------
# # Info Section
# # ---------------------------
# st.markdown("### 📌 Location Information")
# st.info(
#     f"""
#     **Location:** {location}  
#     **Traffic Status:** {status}  
#     **Current Load:** {traffic_value} bytes/min  

#     ✅ Green = Smooth traffic  
#     🟠 Orange = Moderate traffic  
#     🔴 Red = Heavy congestion
#     """
# )





# import streamlit as st
# import requests
# import folium
# from streamlit_folium import st_folium

# API_KEY = "OLyXRwevA68dRR4PlbUjjIPSYoPUrJ84"

# st.set_page_config(page_title="Traffic Prediction with AI Assistant", layout="wide")

# st.title("🚦 Real-Time Traffic Dashboard with AI Assistant")

# # -------------------- FUNCTION: Convert City Name → Coordinates --------------------
# def get_coordinates(place):
#     url = f"https://api.tomtom.com/search/2/geocode/{place}.json?key={API_KEY}"
#     res = requests.get(url).json()
#     try:
#         lat = res['results'][0]['position']['lat']
#         lon = res['results'][0]['position']['lon']
#         return lat, lon
#     except:
#         return None, None

# # -------------------- FUNCTION: Traffic Flow Data (Congestion Level) --------------------
# def get_traffic_flow(lat, lon):
#     url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={lat},{lon}&key={API_KEY}"
#     res = requests.get(url).json()
#     try:
#         freeSpeed = res['flowSegmentData']['freeFlowSpeed']
#         currentSpeed = res['flowSegmentData']['currentSpeed']
#         jamFactor = res['flowSegmentData']['currentTravelTime']
#         return freeSpeed, currentSpeed, jamFactor
#     except:
#         return None, None, None

# # -------------------- FUNCTION: Traffic Incidents --------------------
# def get_incidents(lat, lon):
#     url = f"https://api.tomtom.com/traffic/services/5/incidentDetails?bbox={lon-0.05},{lat-0.05},{lon+0.05},{lat+0.05}&key={API_KEY}&fields=description"
#     res = requests.get(url).json()
#     try:
#         incidents = [i['description'] for i in res['incidents']]
#         return incidents
#     except:
#         return []

# # -------------------- UI INPUT --------------------
# place = st.text_input("Enter Area / City (Example: Begumpet, Hyderabad):", "Begumpet Hyderabad")

# lat, lon = get_coordinates(place)

# if lat and lon:
#     free, current, jam = get_traffic_flow(lat, lon)
    
#     st.subheader(f"📍 Location: {place}")
#     st.write(f"Latitude: {lat}, Longitude: {lon}")

#     col1, col2, col3 = st.columns(3)
#     col1.metric("🚗 Free Flow Speed", f"{free} km/h")
#     col2.metric("🐢 Current Speed", f"{current} km/h")
#     col3.metric("⚠️ Delay Factor", jam)

#     incidents = get_incidents(lat, lon)
#     st.subheader("🛑 Traffic Incidents Nearby")
#     if incidents:
#         for i in incidents:
#             st.write("- ", i)
#     else:
#         st.write("✅ No major incidents reported.")

#     # -------------------- MAP --------------------
#     m = folium.Map(location=[lat, lon], zoom_start=13)
#     folium.Marker([lat, lon], popup=place).add_to(m)
#     st_folium(m, width=700, height=450)

# else:
#     st.error("❗ Could not find location. Try a more specific name.")


# # -------------------- AI ASSISTANT (Inside Sidebar) --------------------
# st.sidebar.title("🤖 Traffic AI Assistant")

# user_input = st.sidebar.text_input("Ask about traffic (Example: What is traffic like in Begumpet?)")

# def ai_response(query):
#     # Extract location from user query
#     words = query.split()
#     detected_place = words[-1]  # last word used as location

#     lat, lon = get_coordinates(detected_place)

#     if not lat:
#         return "❗ I couldn't identify the location. Try again with a valid place."

#     free, current, jam = get_traffic_flow(lat, lon)

#     response = f"""
# Traffic Report for **{detected_place}**:
# - Free Flow Speed: {free} km/h
# - Current Speed: {current} km/h
# - Delay Impact: {jam}

# Conclusion: {"✅ Smooth Traffic" if current >= free*0.8 else "🚦 Heavy Traffic, Expect Delays"}
#     """
#     return response

# if st.sidebar.button("Ask"):
#     answer = ai_response(user_input)
#     st.sidebar.write(answer)













# import streamlit as st
# import requests
# import folium
# from streamlit_folium import st_folium

# # -------------------- CONFIG --------------------
# TOMTOM_API_KEY = "OLyXRwevA68dRR4PlbUjjIPSYoPUrJ84"
# st.set_page_config(page_title="Smart Traffic Monitor", layout="wide")

# # -------------------- PAGE TITLE --------------------
# st.title("🚦 Real-Time Traffic & Route Visualization (TomTom API)")
# st.markdown("Check live traffic and alternate routes for any place in the world.")

# # -------------------- INPUTS --------------------
# col1, col2 = st.columns(2)

# with col1:
#     location_query = st.text_input("📍 Search Location (City / Area)", "Hyderabad")

# with col2:
#     col2_1, col2_2 = st.columns(2)
#     with col2_1:
#         source = st.text_input("🛣️ Source", "Begumpet")
#     with col2_2:
#         destination = st.text_input("🏁 Destination", "Koti")

# # -------------------- FUNCTIONS --------------------
# def get_coordinates(place):
#     """Get latitude & longitude of a place"""
#     url = f"https://api.tomtom.com/search/2/geocode/{place}.json?key={TOMTOM_API_KEY}"
#     res = requests.get(url).json()
#     if res.get("results"):
#         pos = res["results"][0]["position"]
#         return pos["lat"], pos["lon"]
#     return None, None

# def get_traffic_flow(lat, lon):
#     """Get traffic flow data (speed, congestion)"""
#     url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={TOMTOM_API_KEY}&point={lat},{lon}"
#     res = requests.get(url).json()
#     return res.get("flowSegmentData")

# def get_alternate_routes(src_lat, src_lon, dest_lat, dest_lon):
#     """Get main and alternative routes"""
#     url = f"https://api.tomtom.com/routing/1/calculateRoute/{src_lat},{src_lon}:{dest_lat},{dest_lon}/json?key={TOMTOM_API_KEY}&computeBestOrder=true&routeRepresentation=polyline&maxAlternatives=2"
#     res = requests.get(url).json()
#     return res.get("routes", [])

# # -------------------- MAIN LOGIC --------------------
# st.divider()
# map_col, info_col = st.columns([2, 1])

# with map_col:
#     lat, lon = get_coordinates(location_query)
#     if lat and lon:
#         traffic_data = get_traffic_flow(lat, lon)

#         # Create Map
#         m = folium.Map(location=[lat, lon], zoom_start=13)

#         # Color based on congestion
#         if traffic_data:
#             frc = traffic_data.get("currentSpeed", 0)
#             free_flow = traffic_data.get("freeFlowSpeed", 1)
#             ratio = frc / free_flow if free_flow else 0
#             if ratio >= 0.8:
#                 color = "green"
#                 traffic_level = "Low"
#             elif 0.5 <= ratio < 0.8:
#                 color = "orange"
#                 traffic_level = "Medium"
#             else:
#                 color = "red"
#                 traffic_level = "High"

#             # Draw traffic line for the road segment
#             coords = traffic_data.get("coordinates", {}).get("coordinate", [])
#             folium.PolyLine(
#                 [(c["latitude"], c["longitude"]) for c in coords],
#                 color=color, weight=6, opacity=0.8
#             ).add_to(m)

#         # Routing visualization
#         src_lat, src_lon = get_coordinates(source)
#         dest_lat, dest_lon = get_coordinates(destination)
#         if src_lat and dest_lat:
#             routes = get_alternate_routes(src_lat, src_lon, dest_lat, dest_lon)
#             colors = ["blue", "purple", "cyan"]
#             for i, route in enumerate(routes[:3]):
#                 points = route["legs"][0]["points"]
#                 folium.PolyLine(
#                     [(p["latitude"], p["longitude"]) for p in points],
#                     color=colors[i % len(colors)],
#                     weight=5, opacity=0.7,
#                     tooltip=f"Route {i+1}"
#                 ).add_to(m)
#                 folium.Marker([src_lat, src_lon], popup="Source", icon=folium.Icon(color="green")).add_to(m)
#                 folium.Marker([dest_lat, dest_lon], popup="Destination", icon=folium.Icon(color="red")).add_to(m)

#         # Display map
#         st_folium(m, width=900, height=550)
#     else:
#         st.warning("Enter a valid place name.")

# with info_col:
#     st.subheader("📊 Traffic Insights")
#     if traffic_data:
#         st.markdown(f"""
#         **Location:** {location_query}  
#         **Current Speed:** {traffic_data['currentSpeed']} km/h  
#         **Free Flow Speed:** {traffic_data['freeFlowSpeed']} km/h  
#         **Traffic Level:** {traffic_level}  
#         """)
#         st.progress(1 - ratio)
#     else:
#         st.info("No live traffic data available for this area.")

#     if source and destination:
#         st.subheader("🗺️ Route Information")
#         st.markdown("""
#         - 🟦 Main Route  
#         - 🟪 Alternate Route 1  
#         - 🟦 Alternate Route 2  
#         """)
#         st.markdown("TomTom data updates every few minutes for real-time accuracy.")




import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

# -------------------- CONFIG --------------------
TOMTOM_API_KEY = "OLyXRwevA68dRR4PlbUjjIPSYoPUrJ84"
st.set_page_config(page_title="Real-Time Traffic Viewer", layout="wide")

# -------------------- PAGE TITLE --------------------
st.title("🚦 Real-Time Traffic Viewer (TomTom API)")
st.markdown("Search any place and view real-time traffic on the roads.")

# -------------------- SEARCH BAR --------------------
location_query = st.text_input("🔍 Search a Location (City / Area)", "Hyderabad")

# -------------------- FUNCTIONS --------------------
def get_coordinates(place):
    """Convert location name → latitude, longitude"""
    url = f"https://api.tomtom.com/search/2/geocode/{place}.json?key={TOMTOM_API_KEY}"
    res = requests.get(url).json()
    if res.get("results"):
        pos = res["results"][0]["position"]
        return pos["lat"], pos["lon"]
    return None, None


def get_traffic_flow(lat, lon):
    """Fetch traffic flow info for location"""
    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/"
        f"10/json?key={TOMTOM_API_KEY}&point={lat},{lon}"
    )
    res = requests.get(url).json()
    return res.get("flowSegmentData")


# -------------------- LAYOUT --------------------
st.divider()
map_col, info_col = st.columns([2, 1])

# -------------------- MAIN LOGIC --------------------
with map_col:
    lat, lon = get_coordinates(location_query)
    if lat and lon:

        # create the map
        m = folium.Map(location=[lat, lon], zoom_start=13)

        # get traffic
        traffic_data = get_traffic_flow(lat, lon)

        if traffic_data:
            curr = traffic_data.get("currentSpeed", 0)
            free = traffic_data.get("freeFlowSpeed", 1)
            ratio = curr / free if free else 0

            if ratio >= 0.8:
                color = "green"
                traffic_level = "Low"
            elif ratio >= 0.5:
                color = "orange"
                traffic_level = "Medium"
            else:
                color = "red"
                traffic_level = "High"

            # draw full-road colored traffic segment
            coords = traffic_data.get("coordinates", {}).get("coordinate", [])
            if coords:
                folium.PolyLine(
                    [(c["latitude"], c["longitude"]) for c in coords],
                    color=color,
                    weight=7,
                    opacity=0.9
                ).add_to(m)

        # show map
        st_folium(m, width=900, height=550)

    else:
        st.warning("❗ Please enter a valid place name.")

# -------------------- TRAFFIC INFORMATION --------------------
with info_col:
    st.subheader("📊 Traffic Insights")

    if lat and lon and traffic_data:
        st.markdown(f"""
        ### 📍 Location: **{location_query}**
        **Current Speed:** {traffic_data['currentSpeed']} km/h  
        **Free Flow Speed:** {traffic_data['freeFlowSpeed']} km/h  
        **Traffic Condition:** **{traffic_level}**  
        """)

        # progress bar (higher bar = more traffic)
        st.markdown("### Traffic Congestion Level")
        st.progress(1 - ratio)

        st.info("Live traffic updated every few minutes.")
    else:
        st.info("Traffic data will appear here after you search.")
