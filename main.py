from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Cricket Win Probability API")


# -----------------------------
# Input schema (DEFINE FIRST)
# -----------------------------
class MatchInput(BaseModel):
    batting_team: int
    bowling_team: int
    runs: int
    overs: float
    wickets: int
    target: int


# -----------------------------
# Model loading
# -----------------------------
MODEL_PATH = "models/win_predictor.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_win_probability(data: MatchInput):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model file not found. Train the model to create models/win_predictor.pkl"
        )

    input_data = [[
        data.batting_team,
        data.bowling_team,
        data.runs,
        data.overs,
        data.wickets,
        data.target
    ]]

    probability = model.predict_proba(input_data)[0][1]
    return {"win_probability_percent": round(probability * 100, 2)}
