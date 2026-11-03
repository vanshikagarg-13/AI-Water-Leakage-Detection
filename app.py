"""
AI-Based Water Leakage Detection
Beginner-level Streamlit dashboard using MQTT and Isolation Forest.

Run:
    pip install streamlit pandas numpy scikit-learn paho-mqtt
    streamlit run app.py

The app uses water_meter_demo_dataset.csv by default.
For MQTT testing, enable "Use MQTT live data" in the sidebar and publish
JSON messages to test.mosquitto.org topic: water/meter/readings

Expected MQTT JSON:
{"timestamp":"2026-11-03 12:00:00","meter_id":"WM001",
 "flow_rate_lpm":25.5,"consumption_liters":0.38,"pressure_bar":3.0}
"""

import json
import os
import queue
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# MQTT is optional. The dashboard still works with the CSV if paho-mqtt
# is not installed or the broker is unavailable.
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

DATA_FILE = "water_meter_demo_dataset.csv"
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "water/meter/readings"

st.set_page_config(page_title="AI-Based Water Leakage Detection",
                   page_icon="💧", layout="wide")

st.title("💧 AI-Based Water Leakage Detection")
st.write("A simple dashboard to monitor water meter readings and identify unusual patterns.")
st.caption("Technology: MQTT | Isolation Forest | Streamlit")

# Keep recent MQTT readings in memory while the app is running.
if "mqtt_readings" not in st.session_state:
    st.session_state.mqtt_readings = []
if "mqtt_queue" not in st.session_state:
    st.session_state.mqtt_queue = queue.Queue()

def receive_message(client, userdata, message):
    """Place a valid JSON reading in a thread-safe queue."""
    try:
        reading = json.loads(message.payload.decode("utf-8"))
        required = ["meter_id", "flow_rate_lpm", "consumption_liters", "pressure_bar"]
        if all(key in reading for key in required):
            reading.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            userdata.put(reading)
    except Exception:
        pass

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE)
    else:
        data = pd.DataFrame(columns=[
            "timestamp", "meter_id", "flow_rate_lpm",
            "consumption_liters", "pressure_bar"
        ])
    return data

data = load_dataset()

st.sidebar.header("Dashboard Options")
use_mqtt = st.sidebar.checkbox("Use MQTT live data", value=False)
st.sidebar.write("CSV data is used by default.")
st.sidebar.caption("MQTT broker: test.mosquitto.org")
st.sidebar.caption("Topic: water/meter/readings")

if use_mqtt:
    if mqtt is None:
        st.sidebar.error("Install paho-mqtt to use MQTT.")
    else:
        # Start one MQTT client for this Streamlit session.
        if "mqtt_client" not in st.session_state:
            try:
                client = mqtt.Client()
                client.user_data_set(st.session_state.mqtt_queue)
                client.on_message = receive_message
                client.connect(MQTT_BROKER, MQTT_PORT, 60)
                client.subscribe(MQTT_TOPIC)
                client.loop_start()
                st.session_state.mqtt_client = client
                st.sidebar.success("MQTT connected")
            except Exception as error:
                st.sidebar.warning("MQTT connection could not be started.")
                st.sidebar.caption(str(error))
        # Move messages from the MQTT background thread into Streamlit state.
        while not st.session_state.mqtt_queue.empty():
            st.session_state.mqtt_readings.append(st.session_state.mqtt_queue.get())
        st.session_state.mqtt_readings = st.session_state.mqtt_readings[-100:]
        if st.session_state.mqtt_readings:
            mqtt_data = pd.DataFrame(st.session_state.mqtt_readings)
            data = pd.concat([data, mqtt_data], ignore_index=True)

if data.empty:
    st.warning("Dataset not found. Keep water_meter_demo_dataset.csv in the same folder as app.py.")
    st.stop()

# Prepare data and run Isolation Forest on the two main water-use readings.
data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
for column in ["flow_rate_lpm", "consumption_liters", "pressure_bar"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")
data = data.dropna(subset=["flow_rate_lpm", "consumption_liters", "pressure_bar"])
data = data.sort_values("timestamp")

features = data[["flow_rate_lpm", "consumption_liters", "pressure_bar"]]
model = IsolationForest(contamination=0.05, random_state=42)
data["prediction"] = model.fit_predict(features)
data["status"] = np.where(data["prediction"] == -1, "Possible Leakage", "Normal")

total_readings = len(data)
active_meters = data["meter_id"].nunique()
possible_leaks = int((data["status"] == "Possible Leakage").sum())
latest_flow = float(data.iloc[-1]["flow_rate_lpm"])

# Simple summary cards.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Readings", total_readings)
col2.metric("Active Water Meters", active_meters)
col3.metric("Possible Leakage Alerts", possible_leaks)
col4.metric("Latest Flow Rate", f"{latest_flow:.2f} L/min")

st.subheader("Water Flow Monitoring")
selected_meter = st.selectbox(
    "Select Water Meter",
    ["All Meters"] + sorted(data["meter_id"].astype(str).unique().tolist())
)

view = data if selected_meter == "All Meters" else data[data["meter_id"] == selected_meter]
chart_data = view.set_index("timestamp")[["flow_rate_lpm"]].tail(100)
st.line_chart(chart_data, y="flow_rate_lpm")

st.subheader("Water Meter Readings")
st.dataframe(
    view[["timestamp", "meter_id", "flow_rate_lpm",
          "consumption_liters", "pressure_bar", "status"]].tail(20),
    use_container_width=True
)

st.subheader("Possible Leakage Alerts")
alerts = view[view["status"] == "Possible Leakage"]
if alerts.empty:
    st.success("No unusual readings found for this selection.")
else:
    st.warning("The AI model found unusual water readings. These may need inspection.")
    st.dataframe(
        alerts[["timestamp", "meter_id", "flow_rate_lpm",
                "consumption_liters", "pressure_bar", "status"]].tail(15),
        use_container_width=True
    )

st.caption(
    "Note: Isolation Forest identifies unusual readings; it does not confirm a physical leak. "
    "The included CSV is a small demonstration dataset. Replace it with verified meter data "
    "for real testing."
)
