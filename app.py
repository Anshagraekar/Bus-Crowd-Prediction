import streamlit as st
import cv2
import numpy as np
from PIL import Image

from predict_utils import predict_crowd, time_to_hour
from ocr_utils import extract_text_from_image, parse_ticket_text

# --------------------------------------------------
# ROUTE → STOPS MAPPING (STRICT, NO MIXING)
# --------------------------------------------------
ROUTES = {
    "61": [
        "Burdi", "Panchasheel Square", "Rahate Colony", "Ajni Square",
        "Chatrapati Square", "Ujjwal Nagar", "Airport",
        "Chinchbhavan", "Khapri", "Ashokwan",
        "Mohgaon", "Butibori"
    ],
    "35": [
        "Pardi", "HB Town", "Wathoda Square", "Kharbi Square",
        "Dighori", "Manewada", "Chatrapati Square",
        "Pratap Nagar", "Trimurti Nagar",
        "Dharampeth", "Hingna", "YCCE"
    ]
}

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(page_title="Smart Bus Crowd Prediction", layout="centered")

# Clear any cached data on app load
if 'initialized' not in st.session_state:
    st.session_state.initialized = True

st.title("🚌 Smart Bus Crowd Prediction")
st.caption("Route 61 & Route 35 | Manual + OCR Based")

input_mode = st.radio("Choose Input Mode", ["Manual Input", "Upload Ticket (OCR)"])

route_id = boarding_stop = alighting_stop = time_str = None

# --------------------------------------------------
# MANUAL INPUT MODE
# --------------------------------------------------
if input_mode == "Manual Input":
    route_id = st.selectbox("Select Route", ["61", "35"])

    stops = ROUTES[route_id]

    boarding_stop = st.selectbox("Boarding Stop", stops)
    
    # Allow selecting any stop as destination (both forward and backward)
    available_destinations = [s for s in stops if s != boarding_stop]
    alighting_stop = st.selectbox(
        "Destination Stop",
        available_destinations
    )

    time_str = st.time_input("Boarding Time").strftime("%H:%M")

# --------------------------------------------------
# OCR INPUT MODE
# --------------------------------------------------
else:
    uploaded_file = st.file_uploader("Upload Bus Ticket Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Ticket", use_container_width=True)

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        text = extract_text_from_image(image_cv)

        st.subheader("📄 OCR Extracted Text")
        st.code(text)

        route_id, boarding_stop, alighting_stop, time_str = parse_ticket_text(text)

        st.subheader("🧾 Parsed Ticket Details")
        st.write({
            "Route": route_id,
            "From": boarding_stop,
            "To": alighting_stop,
            "Time": time_str
        })

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🚦 Predict Crowd", key="predict_button"):
    if not all([route_id, boarding_stop, alighting_stop, time_str]):
        st.error("❌ Please provide all required inputs")
    else:
        hour = time_to_hour(time_str)

        # Force fresh prediction (no caching)
        crowd_level, insight = predict_crowd(
            route_id,
            boarding_stop,
            alighting_stop,
            hour
        )

        # Add debug output in UI
        st.write(f"**Debug Info:** Predicted crowd level = `{crowd_level}`")

        # ---------------- OUTPUT ----------------
        st.subheader("🚥 Current Crowd Level")

        # Handle all crowd levels with PEAK as the highest severity
        if crowd_level == "PEAK":
            st.error("🔴🔴 PEAK HOUR – Extremely Crowded! Consider alternate route/timing")
        elif crowd_level == "HIGH":
            st.error("🔴 High Crowd – Overcrowded, standing room only")
        elif crowd_level == "MEDIUM":
            st.warning("🟡 Medium Crowd – Standing passengers expected")
        elif crowd_level == "LOW":
            st.success("🟢 Low Crowd – Seats available")
        else:
            # Fallback for any unexpected values
            st.info(f"ℹ️ Crowd Level: {crowd_level}")

        st.subheader("🧠 Journey Insight")
        st.info(insight)

        st.subheader("📌 Trip Summary")
        
        # Determine direction
        stops = ROUTES[route_id]
        boarding_idx = stops.index(boarding_stop)
        alighting_idx = stops.index(alighting_stop)
        direction = "➡️ Forward" if boarding_idx < alighting_idx else "⬅️ Backward"
        
        st.write(f"**Route:** {route_id}")
        st.write(f"**Direction:** {direction}")
        st.write(f"**From:** {boarding_stop}")
        st.write(f"**To:** {alighting_stop}")
        st.write(f"**Time:** {time_str}")