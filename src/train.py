import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow

# Load dataset
df = pd.read_csv("data/matches.csv")

# Encode categorical teams
le = LabelEncoder()
df["batting_team"] = le.fit_transform(df["batting_team"])
df["bowling_team"] = le.fit_transform(df["bowling_team"])

# Features & target
X = df.drop("result", axis=1)
y = df["result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
mlflow.start_run()

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)

# Save model
joblib.dump(model, "models/win_predictor.pkl")
mlflow.log_artifact("models/win_predictor.pkl")

mlflow.end_run()

print("✅ Model trained successfully")
