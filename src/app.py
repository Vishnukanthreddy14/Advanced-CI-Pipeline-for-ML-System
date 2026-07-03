"""
FastAPI Application for ML Model Serving
Exposes trained model as REST API for predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List


# Initialize FastAPI app
app = FastAPI(
    title="ML Prediction API",
    description="REST API for ML model predictions",
    version="1.0.0"
)


# Load model
model_path = os.path.join(os.path.dirname(__file__), '../models/model.joblib')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None


# Pydantic model for input validation
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: int
    confidence: float
    class_name: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "RandomForestClassifier"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction using trained model.
    
    Args:
        request: PredictionRequest containing feature values
        
    Returns:
        PredictionResponse with prediction and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate input length
    if len(request.features) != 4:
        raise HTTPException(
            status_code=400,
            detail="Expected 4 features (sepal_length, sepal_width, petal_length, petal_width)"
        )
    
    # Convert to numpy array and reshape
    features_array = np.array(request.features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    confidence = float(max(probabilities))
    
    # Map prediction to class name
    class_names = ['setosa', 'versicolor', 'virginica']
    class_name = class_names[prediction]
    
    return PredictionResponse(
        prediction=int(prediction),
        confidence=confidence,
        class_name=class_name
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make prediction (POST)",
            "/docs": "Swagger UI documentation"
        }
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
