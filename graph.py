import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("bus_crowd_with_alighting_14_days.csv")

model = joblib.load("crowd_rf_model.pkl")
label_encoder = joblib.load("crowd_label_encoder.pkl")

X = df[["route_id", "hour", "boarding_count", "alighting_count"]]
y = label_encoder.transform(df["crowd_level"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

classes = label_encoder.classes_
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

crowd_mapping = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "PEAK": 4
}

df["crowd_numeric"] = df["crowd_level"].map(crowd_mapping)

hourly_crowd = df.groupby("hour")["crowd_numeric"].mean()

plt.figure()
plt.plot(hourly_crowd.index, hourly_crowd.values, marker='o')
plt.xlabel("Hour of Day")
plt.ylabel("Average Crowd Level")
plt.title("Crowd Level vs Time of Day")
plt.show()
