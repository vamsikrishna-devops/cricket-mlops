from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "batting_team": 1,
        "bowling_team": 2,
        "runs": 140,
        "overs": 17.3,
        "wickets": 5,
        "target": 170
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "win_probability_percent" in data
    assert 0 <= data["win_probability_percent"] <= 100
