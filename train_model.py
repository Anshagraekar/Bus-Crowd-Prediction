import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("bus_crowd_with_alighting_14_days.csv")

print("Dataset shape:", df.shape)
print(df.head())
print("\nUnique crowd levels:", df["crowd_level"].unique())
print("Crowd level distribution:")
print(df["crowd_level"].value_counts())

# Encode crowd levels
label_encoder = LabelEncoder()
df["crowd_encoded"] = label_encoder.fit_transform(df["crowd_level"])

print("\nCrowd classes:", label_encoder.classes_)

# Features
X = df[["route_id", "hour", "boarding_count", "alighting_count"]]
y = df["crowd_encoded"]

print("\nFeature statistics:")
print(X.describe())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model and encoder with CORRECT names
joblib.dump(model, "crowd_rf_model.pkl")
joblib.dump(label_encoder, "crowd_label_encoder.pkl")  # FIX: consistent naming

print("\nModel and encoder saved successfully")
print(f"Saved files: crowd_rf_model.pkl, crowd_label_encoder.pkl")