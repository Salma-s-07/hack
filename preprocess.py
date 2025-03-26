import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("pcos_dataset.csv")

# Print column names to verify
#print("Columns in dataset:", df.columns)

# Update target column name based on your dataset
# Change this if needed

# Drop target column from features
X = df.drop(columns=['PCOS_Diagnosis'])
y = df['PCOS_Diagnosis']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model and scaler
joblib.dump(model, "pcos_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")
