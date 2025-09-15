import fastapi
import joblib
import streamlit
from pydantic import BaseModel
import uvicorn

app = fastapi.FastAPI()

class HouseFeatures(BaseModel):
    size: int
    bedrooms: int
    garden: int

@app.post("/predict")
def predict(house: HouseFeatures):
    model = joblib.load("regression.joblib")
    try:
        size = streamlit.number_input("Size", 10, 1000, house.size)
        bedrooms = streamlit.number_input("Number of bedrooms", 1, 10, house.bedrooms)
        garden = streamlit.number_input("Has garden?", 0, 1, house.garden)
    except:
        raise fastapi.HTTPException(
            status_code=400,
            detail="Invalid input. Size must be between 10 and 1000. Bedrooms must be between 1 and 10. Garden must be 0 or 1."
        )
    features = [[size, bedrooms, garden]]
    prediction = model.predict(features)[0]
    return {"y_pred": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5242, log_level="info")

