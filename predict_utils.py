import joblib
import pandas as pd

# ---------------- ROUTE STOPS (ORDER MATTERS) ----------------
ROUTE_STOPS = {
    35: [
        "Pardi", "Hb Town", "Wathoda Square", "Kharbi Square",
        "Dighori", "Manewada", "Chatrapati Square",
        "Pratap Nagar", "Trimurti Nagar",
        "Dharampeth", "Hingna", "Ycce"
    ],
    61: [
        "Burdi", "Panchsheel Square", "Rahate Colony",
        "Ajni Square", "Chatrapati Square", "Ujjwal Nagar",
        "Airport", "Chinchbhavan", "Khapri",
        "Ashokwan", "Dongargaon", "Mohgaon", "Butibori"
    ]
}

# ---------------- LOAD MODEL & ENCODER ----------------
model = joblib.load("crowd_rf_model.pkl")
label_encoder = joblib.load("crowd_label_encoder.pkl")  # FIX: correct filename

# Load dataset ONCE
DATASET = pd.read_csv("bus_crowd_with_alighting_14_days.csv")


# ---------------- TIME HANDLING ----------------
def time_to_hour(time_str):
    """
    Converts 'HH:MM' or 'HH:MM AM/PM' → hour (0–23)
    """
    time_str = time_str.strip().upper()
    if "AM" in time_str or "PM" in time_str:
        from datetime import datetime
        return datetime.strptime(time_str, "%I:%M %p").hour
    return int(time_str.split(":")[0])


# ---------------- ESTIMATE BOARDING & ALIGHTING ----------------
def estimate_boarding_alighting(route_id, stop_name, hour):
    # Convert route_id to int for comparison
    route_id = int(route_id)
    
    subset = DATASET[
        (DATASET["route_id"] == route_id) &
        (DATASET["stop_name"].str.lower() == stop_name.lower()) &
        (DATASET["hour"] == hour)
    ]

    if subset.empty:
        # Fallback: try without hour constraint
        subset = DATASET[
            (DATASET["route_id"] == route_id) &
            (DATASET["stop_name"].str.lower() == stop_name.lower())
        ]
        
        if subset.empty:
            print(f"Warning: No data found for route {route_id}, stop {stop_name}")
            return 5, 3  # more realistic fallback

    boarding = int(subset["boarding_count"].mean())
    alighting = int(subset["alighting_count"].mean())

    return max(boarding, 1), max(alighting, 1)


# ---------------- CROWD PREDICTION ----------------
def predict_crowd(route_id, boarding_stop, destination_stop, hour):
    """
    Final crowd prediction function
    """
    # Convert route_id to int
    route_id = int(route_id)

    boarding_count, alighting_count = estimate_boarding_alighting(
        route_id, boarding_stop, hour
    )

    print(f"Debug - Route: {route_id}, Hour: {hour}")
    print(f"Debug - Boarding: {boarding_count}, Alighting: {alighting_count}")

    input_df = pd.DataFrame([{
        "route_id": route_id,
        "hour": hour,
        "boarding_count": boarding_count,
        "alighting_count": alighting_count
    }])

    pred_encoded = model.predict(input_df)[0]
    ml_crowd = label_encoder.inverse_transform([pred_encoded])[0]
    
    print(f"Debug - ML Prediction: {ml_crowd}")

    # ---------------- USE ML PREDICTION DIRECTLY ----------------
    # The model is trained to recognize hour patterns:
    # PEAK: hours 8-10, HIGH: hours 17-20, MEDIUM: 6,7,11,16,21, LOW: 12-15
    crowd = ml_crowd

    print(f"Debug - Final Crowd: {crowd}")

    # ---------------- CROWD REDUCTION STOP ----------------
    reduction_stop = find_crowd_reduction_stop(
        route_id,
        boarding_stop,
        destination_stop,
        hour
    )

    insight = (
        f"Crowd likely reduces after **{reduction_stop}**"
        if reduction_stop else
        "No major crowd drop expected on this stretch"
    )

    return crowd, insight


# ---------------- CROWD REDUCTION LOGIC ----------------
def find_crowd_reduction_stop(route_id, boarding_stop, destination_stop, hour):
    # Convert route_id to int
    route_id = int(route_id)
    
    if route_id not in ROUTE_STOPS:
        return None

    stops = ROUTE_STOPS[route_id]

    if boarding_stop not in stops or destination_stop not in stops:
        return None

    start_idx = stops.index(boarding_stop)
    end_idx = stops.index(destination_stop)

    # Determine direction and create journey
    if start_idx < end_idx:
        # Forward direction
        journey = stops[start_idx:end_idx + 1]
    else:
        # Backward direction (reverse the route)
        journey = stops[end_idx:start_idx + 1][::-1]

    # Check each stop in the journey for crowd reduction
    for stop in journey[1:]:  # Skip the boarding stop
        subset = DATASET[
            (DATASET["route_id"] == route_id) &
            (DATASET["stop_name"].str.lower() == stop.lower()) &
            (DATASET["hour"] == hour)
        ]

        if subset.empty:
            continue

        # If more people alight than board, crowd reduces
        if subset["alighting_count"].mean() > subset["boarding_count"].mean():
            return stop

    return None