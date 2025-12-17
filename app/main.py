from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize app
app = FastAPI(title="Cricket Win Probability API")

# Load trained model
model = joblib.load("models/win_predictor.pkl")

# Input schema
class MatchInput(BaseModel):
    batting_team: int
    bowling_team: int
    runs: int
    overs: float
    wickets: int
    target: int

# Prediction endpoint
@app.post("/predict")
def predict_win_probability(data: MatchInput):
    input_data = [[
        data.batting_team,
        data.bowling_team,
        data.runs,
        data.overs,
        data.wickets,
        data.target
    ]]
    
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "win_probability_percent": round(probability * 100, 2)
    }
