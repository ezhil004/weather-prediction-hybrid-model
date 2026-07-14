import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = "weather_classification_data.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please provide the dataset.")
    exit(1)

# Convert 'Weather Type' to string format (just in case)
df['Weather Type'] = df['Weather Type'].astype(str)

# Encode the weather labels
label_encoder = LabelEncoder()
df['Weather Type'] = label_encoder.fit_transform(df['Weather Type'])

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Print label encoding mapping
print("Weather Labels Mapping:", dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)))

# Drop missing values
df = df.dropna()

# Split features and target
X = df.drop(columns=['Weather Type'])
y = df['Weather Type']

# Drop non-numeric columns from X
X = X.select_dtypes(include=[np.number])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Function to predict weather from new input
def predict_weather(temp, humidity, wind_speed, pressure, uv_index, visibility, precipitation):
    input_data = np.array([[temp, humidity, wind_speed, precipitation, pressure, uv_index, visibility]])
    feature_names = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Standardize
    input_scaled = scaler.transform(input_df)
    
    # Predict
    encoded_prediction = model.predict(input_scaled)
    predicted_weather = label_encoder.inverse_transform([encoded_prediction[0]])
    
    return predicted_weather[0]

# Save model
with open("weather_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
