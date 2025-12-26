import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc
import numpy as np
from typing import List
import uvicorn

app = FastAPI(title="Iris Classifier API")

model = None
current_model_version = None


class PredictRequest(BaseModel):
    features: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[int]
    model_version: str


class UpdateModelRequest(BaseModel):
    run_id: str


class UpdateModelResponse(BaseModel):
    old_version: str
    new_version: str


def load_model_from_mlflow(run_id: str = None):
    global model, current_model_version
    mlflow.set_tracking_uri("file:./mlflow/mlruns")

    try:
        if run_id is None:
            run_id = "0a6022e77b0e4206b7914737f308ead1"

        model_uri = f"runs:/{run_id}/iris_classifier"
        print(mlflow.search_runs("iris_classifier"))
        model = mlflow.pyfunc.load_model(model_uri)
        current_model_version = run_id
        print(f"Model loaded successfully from run: {run_id}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e


@app.on_event("startup")
def startup_event():
    try:
        load_model_from_mlflow()
        print("Server started with model loaded from MLflow")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {str(e)}")
        print("Model needs to be loaded via /update-model endpoint")


@app.get("/")
def root():
    return {
        "message": "Hello World!",
        "model_loaded": model is not None,
        "current_model_version": current_model_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded.")

    try:
        features_array = np.array(request.features)

        if features_array.shape[1] != 4:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 4 features per sample, got {features_array.shape[1]}",
            )

        predictions = model.predict(features_array)

        predictions_list = [int(pred) for pred in predictions]

        return PredictResponse(
            predictions=predictions_list, model_version=current_model_version
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/update-model", response_model=UpdateModelResponse)
def update_model(request: UpdateModelRequest):
    old_version = current_model_version

    try:
        load_model_from_mlflow(run_id=request.run_id)

        return UpdateModelResponse(
            old_version=old_version if old_version else "none",
            new_version=current_model_version,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
