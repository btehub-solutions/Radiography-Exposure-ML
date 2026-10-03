"""
Radiographic Exposure Parameter Prediction API
Clinical Decision Support Prototype (kVp and mAs Estimator)
"""

from contextlib import asynccontextmanager
import csv
import io
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import joblib
import numpy as np
import openpyxl
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


class FilePatientPrediction(BaseModel):
    row: int
    patient_id: Optional[str] = None
    sex: str
    age: float
    weight_kg: float
    height_m: float
    chest_cm: float
    computed_bmi: float
    bmi_category: str
    recommended_kvp: int
    recommended_mas: float
    predicted_kvp: float
    predicted_mas: float
    status: str = "success"
    error: Optional[str] = None


class FilePredictionResponse(BaseModel):
    status: str = "success"
    filename: str
    total_rows: int
    successful_predictions: int
    failed_rows: int
    predictions: List[FilePatientPrediction]


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


@app.post("/predict-file", tags=["Inference"])
async def predict_from_file(
    file: UploadFile = File(..., description="Upload a CSV (.csv) or Excel (.xlsx) file containing patient records"),
    download_csv: bool = Query(False, description="If True, downloads the results as an enriched CSV file with predicted kVp & mAs appended"),
):
    """
    Upload a CSV or Excel spreadsheet of patients (e.g. 50+ records) and obtain bulk exposure predictions.
    Supports standard column headers: sex/gender, age, weight_kg, height_m, chest_cm, (optional id, bmi).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Please upload a CSV (.csv) or Excel (.xlsx) file.",
        )

    contents = await file.read()
    records: List[Dict[str, Any]] = []

    if ext == ".csv":
        try:
            text = contents.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = contents.decode("latin-1")
        reader = csv.DictReader(io.StringIO(text))
        for r in reader:
            records.append(dict(r))
    else:
        # Excel (.xlsx)
        try:
            wb = openpyxl.load_workbook(filename=io.BytesIO(contents), data_only=True)
            sheet = wb.active
            rows_iter = sheet.iter_rows(values_only=True)
            header_row = next(rows_iter, None)
            if not header_row:
                raise HTTPException(status_code=400, detail="Excel file is empty.")
            headers = [str(col).strip() if col is not None else f"col_{i}" for i, col in enumerate(header_row)]
            for row in rows_iter:
                if any(cell is not None for cell in row):
                    row_dict = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                    records.append(row_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read Excel file: {str(e)}")

    if not records:
        raise HTTPException(status_code=400, detail="The uploaded file contains no data rows.")

    results: List[FilePatientPrediction] = []
    output_rows: List[Dict[str, Any]] = []

    def get_val(row_dict: dict, *keys):
        norm = {str(k).strip().lower(): v for k, v in row_dict.items() if k is not None}
        for k in keys:
            if k in norm and norm[k] not in (None, ""):
                return norm[k]
        return None

    successful = 0
    failed = 0

    for idx, row in enumerate(records, start=1):
        pid = get_val(row, "id", "patient_id", "record_id", "patient")
        sex = get_val(row, "sex", "gender", "patient_sex")
        age = get_val(row, "age", "patient_age")
        weight = get_val(row, "weight_kg", "weight", "wt_kg", "wt")
        height = get_val(row, "height_m", "height", "ht_m", "ht")
        chest = get_val(row, "chest_cm", "chest", "chest_thickness", "thickness_cm")
        bmi = get_val(row, "bmi", "body_mass_index")

        if None in (sex, age, weight, height, chest):
            failed += 1
            missing = [name for name, val in [("sex", sex), ("age", age), ("weight_kg", weight), ("height_m", height), ("chest_cm", chest)] if val is None]
            err_msg = f"Missing required columns: {', '.join(missing)}"
            results.append(
                FilePatientPrediction(
                    row=idx,
                    patient_id=str(pid) if pid is not None else None,
                    sex=str(sex or "Unknown"),
                    age=float(age or 0),
                    weight_kg=float(weight or 0),
                    height_m=float(height or 0),
                    chest_cm=float(chest or 0),
                    computed_bmi=0.0,
                    bmi_category="Invalid",
                    recommended_kvp=0,
                    recommended_mas=0.0,
                    predicted_kvp=0.0,
                    predicted_mas=0.0,
                    status="failed",
                    error=err_msg,
                )
            )
            continue

        try:
            p_input = PatientInput(
                sex=sex,
                age=float(age),
                weight_kg=float(weight),
                height_m=float(height),
                chest_cm=float(chest),
                bmi=float(bmi) if bmi is not None else None,
            )
            pred = perform_prediction(p_input)
            successful += 1
            pred_item = FilePatientPrediction(
                row=idx,
                patient_id=str(pid) if pid is not None else str(idx),
                sex=str(sex),
                age=p_input.age,
                weight_kg=p_input.weight_kg,
                height_m=p_input.height_m,
                chest_cm=p_input.chest_cm,
                computed_bmi=pred.computed_bmi,
                bmi_category=pred.bmi_category,
                recommended_kvp=pred.recommended_kvp,
                recommended_mas=pred.recommended_mas,
                predicted_kvp=pred.predicted_kvp,
                predicted_mas=pred.predicted_mas,
                status="success",
            )
            results.append(pred_item)

            if download_csv:
                annotated_row = dict(row)
                annotated_row["recommended_kvp"] = pred.recommended_kvp
                annotated_row["recommended_mas"] = pred.recommended_mas
                annotated_row["predicted_kvp"] = pred.predicted_kvp
                annotated_row["predicted_mas"] = pred.predicted_mas
                annotated_row["computed_bmi"] = pred.computed_bmi
                annotated_row["bmi_category"] = pred.bmi_category
                output_rows.append(annotated_row)

        except Exception as e:
            failed += 1
            results.append(
                FilePatientPrediction(
                    row=idx,
                    patient_id=str(pid) if pid is not None else None,
                    sex=str(sex),
                    age=float(age or 0),
                    weight_kg=float(weight or 0),
                    height_m=float(height or 0),
                    chest_cm=float(chest or 0),
                    computed_bmi=0.0,
                    bmi_category="Error",
                    recommended_kvp=0,
                    recommended_mas=0.0,
                    predicted_kvp=0.0,
                    predicted_mas=0.0,
                    status="failed",
                    error=str(e),
                )
            )

    if download_csv and output_rows:
        out_stream = io.StringIO()
        fieldnames = list(output_rows[0].keys())
        writer = csv.DictWriter(out_stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
        out_stream.seek(0)
        return StreamingResponse(
            io.BytesIO(out_stream.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predicted_{Path(file.filename).stem}.csv"},
        )

    return FilePredictionResponse(
        status="success",
        filename=file.filename,
        total_rows=len(records),
        successful_predictions=successful,
        failed_rows=failed,
        predictions=results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
