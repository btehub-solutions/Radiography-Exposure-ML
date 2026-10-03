"""
Radiographic Exposure Parameter Prediction API
Clinical Decision Support Prototype (kVp and mAs Estimator)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import logging
import time

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("radiography-api")

# Directory setup
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Storage for loaded models
model_registry: Dict[str, object] = {}


def get_model_path(filename: str) -> Path:
    return MODELS_DIR / filename


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and scalers into memory upon application startup."""
    logger.info("Starting up Radiography Exposure ML Service...")
    
    # 1. Load Scaler
    scaler_path = get_model_path("scaler.joblib")
    if not scaler_path.exists():
        logger.error(f"Required scaler not found at: {scaler_path}")
        raise FileNotFoundError(f"Scaler missing: {scaler_path}")
    model_registry["scaler"] = joblib.load(scaler_path)
    logger.info(f"Loaded Scaler successfully from {scaler_path.name}")

    # 2. Model mappings: primary ridge models and alternate architectures
    models_to_load = {
        "ridge_kvp": "kvp_model.joblib",
        "ridge_mas": "mas_model.joblib",
        "random_forest_kvp": "kvp_random_forest.joblib",
        "random_forest_mas": "mas_random_forest.joblib",
        "gradient_boosting_kvp": "kvp_gradient_boosting.joblib",
        "gradient_boosting_mas": "mas_gradient_boosting.joblib",
        "mlp_kvp": "kvp_mlp_neural_network.joblib",
        "mlp_mas": "mas_mlp_neural_network.joblib",
        "linear_regression_kvp": "kvp_linear_regression.joblib",
        "linear_regression_mas": "mas_linear_regression.joblib",
    }

    loaded_count = 0
    for key, filename in models_to_load.items():
        path = get_model_path(filename)
        if path.exists():
            try:
                model_registry[key] = joblib.load(path)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Could not load {filename}: {e}")
        else:
            logger.debug(f"Optional model file {filename} not found.")

    logger.info(f"Loaded {loaded_count} model estimators successfully.")
    
    # Check that primary Ridge models are present
    if "ridge_kvp" not in model_registry or "ridge_mas" not in model_registry:
        raise RuntimeError("Primary Ridge regression models (kvp_model.joblib / mas_model.joblib) could not be loaded!")

    yield

    logger.info("Shutting down Radiography Exposure ML Service...")
    model_registry.clear()


# Initialize FastAPI app
app = FastAPI(
    title="Radiographic Exposure Parameter API",
    description="Machine Learning Decision Support API predicting optimal tube voltage (kVp) and exposure product (mAs) for thoracic radiography.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-platform app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for mobile/web apps
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class PatientInput(BaseModel):
    sex: Union[Literal["Female", "Male", "female", "male", "F", "M", "f", "m"], int] = Field(
        ...,
        description="Biological sex: 'Female' (or 0) / 'Male' (or 1)",
        examples=["Female"],
    )
    age: float = Field(
        ...,
        ge=18.0,
        le=100.0,
        description="Patient age in years (cohort optimized for adults 18-60)",
        examples=[28.0],
    )
    weight_kg: float = Field(
        ...,
        ge=30.0,
        le=250.0,
        description="Patient body weight in kilograms",
        examples=[65.0],
    )
    height_m: float = Field(
        ...,
        ge=1.0,
        le=2.5,
        description="Patient height in meters (e.g. 1.70)",
        examples=[1.68],
    )
    chest_cm: float = Field(
        ...,
        ge=15.0,
        le=45.0,
        description="Anteroposterior chest thickness in centimeters",
        examples=[22.5],
    )
    bmi: Optional[float] = Field(
        None,
        ge=10.0,
        le=70.0,
        description="Body Mass Index in kg/m^2. If omitted, computed automatically from weight and height.",
        examples=[23.03],
    )
    model_architecture: Optional[Literal["ridge", "random_forest", "gradient_boosting", "mlp", "linear_regression"]] = Field(
        "ridge",
        description="ML architecture to use for prediction. Default is 'ridge' (primary validated model).",
        examples=["ridge"],
    )


class PredictionResult(BaseModel):
    status: str = "success"
    model_used: str
    predicted_kvp: float = Field(..., description="Continuous regression output for tube voltage (kVp)")
    predicted_mas: float = Field(..., description="Continuous regression output for tube current-time product (mAs)")
    recommended_kvp: int = Field(..., description="Clinically rounded kVp setting (minimum clamped at 70 kVp)")
    recommended_mas: float = Field(..., description="Clinically rounded mAs setting (rounded to nearest 0.1)")
    computed_bmi: float
    bmi_category: str
    fixed_sid_cm: int = 180
    clinical_disclaimer: str = (
        "RESEARCH PROTOTYPE DECISION SUPPORT ONLY. NOT A REPLACEMENT FOR LICENSED RADIOLOGIC TECHNOLOGIST VERIFICATION."
    )


class BatchPatientInput(BaseModel):
    patients: List[PatientInput]


class BatchPredictionResult(BaseModel):
    status: str = "success"
    count: int
    predictions: List[PredictionResult]


# --- Helper Functions ---
def categorize_bmi(bmi_val: float) -> str:
    if bmi_val < 18.5:
        return "Underweight"
    elif bmi_val < 25.0:
        return "Normal Weight"
    elif bmi_val < 30.0:
        return "Overweight"
    else:
        return "Obese"


def encode_sex(sex_input: Union[str, int]) -> int:
    if isinstance(sex_input, int):
        return 1 if sex_input == 1 else 0
    s = str(sex_input).strip().lower()
    if s in ["male", "m", "1"]:
        return 1
    return 0


def perform_prediction(patient: PatientInput) -> PredictionResult:
    # 1. Encode Sex
    sex_enc = encode_sex(patient.sex)

    # 2. Determine / Calculate BMI
    if patient.bmi is not None and patient.bmi > 0:
        bmi_val = round(float(patient.bmi), 2)
    else:
        bmi_val = round(float(patient.weight_kg) / (float(patient.height_m) ** 2), 2)

    bmi_cat = categorize_bmi(bmi_val)

    # 3. Form continuous biometric features vector in exact scaler order:
    # ['age', 'weight_kg', 'height_m', 'bmi', 'chest_cm']
    cont_features = np.array(
        [[patient.age, patient.weight_kg, patient.height_m, bmi_val, patient.chest_cm]],
        dtype=np.float64,
    )

    # 4. Scale continuous features
    scaler = model_registry.get("scaler")
    if scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model scaler is not initialized.",
        )
    scaled_cont = scaler.transform(cont_features)

    # 5. Combine binary sex_encoded with scaled biometrics (6 features total expected by model):
    # [sex_encoded, scaled_age, scaled_weight, scaled_height, scaled_bmi, scaled_chest]
    feature_vector = np.hstack([[sex_enc], scaled_cont[0]]).reshape(1, -1)

    # 6. Select model estimators
    arch = patient.model_architecture or "ridge"
    kvp_key = f"{arch}_kvp"
    mas_key = f"{arch}_mas"

    kvp_estimator = model_registry.get(kvp_key) or model_registry.get("ridge_kvp")
    mas_estimator = model_registry.get(mas_key) or model_registry.get("ridge_mas")

    if kvp_estimator is None or mas_estimator is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Estimators for architecture '{arch}' are unavailable.",
        )

    # 7. Predict continuous values
    raw_kvp = float(kvp_estimator.predict(feature_vector)[0])
    raw_mas = float(mas_estimator.predict(feature_vector)[0])

    # 7. Apply clinical rounding and constraints
    # Diagnostic adult thoracic standard: floor clamping at 70 kVp based on dataset boundaries
    recommended_kvp = int(round(max(raw_kvp, 70.0)))
    recommended_mas = float(round(max(raw_mas, 1.0), 1))

    return PredictionResult(
        model_used=arch,
        predicted_kvp=round(raw_kvp, 2),
        predicted_mas=round(raw_mas, 2),
        recommended_kvp=recommended_kvp,
        recommended_mas=recommended_mas,
        computed_bmi=bmi_val,
        bmi_category=bmi_cat,
    )


# --- Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Service landing endpoint with API metadata and documentation links."""
    return {
        "service": "Radiographic Exposure Parameter ML API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "primary_model": "Ridge Regularized Linear Regression (Alpha=10.0)",
        "intended_use": "Thoracic radiography exposure optimization decision support (180 cm SID)",
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint for Render keep-alive pings and cold-start verification.
    """
    models_ready = "scaler" in model_registry and "ridge_kvp" in model_registry
    return {
        "status": "healthy" if models_ready else "degraded",
        "models_loaded": models_ready,
        "available_architectures": list({k.split("_")[0] for k in model_registry.keys() if "_" in k}),
        "timestamp": time.time(),
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict_exposure(patient: PatientInput):
    """
    Predict optimal kVp (tube voltage) and mAs (exposure charge) for a single patient.
    """
    try:
        return perform_prediction(patient)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post("/batch-predict", response_model=BatchPredictionResult, tags=["Inference"])
async def batch_predict_exposure(payload: BatchPatientInput):
    """
    Batch predict optimal exposure parameters for multiple patient records.
    """
    results: List[PredictionResult] = []
    for item in payload.patients:
        results.append(perform_prediction(item))
    return BatchPredictionResult(count=len(results), predictions=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
