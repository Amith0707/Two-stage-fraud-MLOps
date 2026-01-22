from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.inference.predictor import FraudPredictor
from src.inference.inference_logger import log_inference
from src.db.insert_transaction import insert_transaction
from utils.logger import get_logger

logger = get_logger(__name__)

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

app = FastAPI(
    title="Fraud Detection API",
    description="Two-Stage Fraud Detection System",
    version="1.0.0"
)

templates = Jinja2Templates(directory="src/api/templates")

# Load once per startup
predictor = FraudPredictor()
logger.info("FraudPredictor loaded successfully")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
def predict(transaction: Transaction):
    """
    JSON-based inference endpoint.
    Used by frontend (fetch) and external clients.
    Within this inference call before returning the result 
    the transaction is also logged as well as the transaction as captured in a new table in 
    postgres
    """
    result = predictor.predict(
        transaction=transaction.model_dump()
    )
    insert_transaction(
        features=transaction.model_dump(),
        prediction=result['prediction'],
        probability=result['probability'],
        stage=result['stage_used']
    )
    log_inference(
        features=transaction.model_dump(),
        prediction=result['prediction'],
        probability=result['probability'],
        stage=result['stage_used']
    )
    return result

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }

@app.get("/metadata")
def metadata():
    return {
        "stage_1_model": "logistic_regression",
        "stage_2_model": "xgboost",
        "threshold_range": [0.3, 0.7],
        "problem_type": "binary_classification",
        "use_case": "credit_card_fraud"
    }