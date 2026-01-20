import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger=get_logger(__name__)

class FraudPredictor:
    """
    This is a class mainly to handle inference pipeline and connect to our core ML 
    logic to handle the inference once the API endpoint hits a request.
    """

    def __init__(self,stage1_dir:str="artifacts/stage1",stage2_dir:str="artifacts/stage2"):
        """
        Loads all inference artifacts ONCE.
        This runs at app startup when service is intialized via FastAPI
        """

        stage1_dir=Path(stage1_dir)
        stage2_dir=Path(stage2_dir)

        # Stage-1 Artifacts
        self.stage1_model=joblib.load(stage1_dir/"model.pkl")
        self.scaler=joblib.load(stage1_dir/"scaler.pkl")

        with open(stage1_dir/"config.json","r") as f:
            config=json.load(f)
        
        self.low_threshold=config["low_threshold"]
        self.high_threshold=config["high_threshold"]

        # Stage-2 artifacts
        self.stage2_model=joblib.load(stage2_dir/"model.pkl")

        self.stage1_decision_threshold=0.5

    def _preprocess(self,transaction:dict)->pd.DataFrame:
        """
        Converts raw input dict into a scaled dataframe.
        """
        df=pd.DataFrame([transaction])
        scaled=self.scaler.transform(df)
        return scaled
    
    def predict(self,transaction:dict)->dict:
        """
        Runs full two stage inference for ONE Transaction
        """

        X_scaled=self._preprocess(transaction=transaction)
        stage1_prob=self.stage1_model.predict_proba(X_scaled)[:,1][0]
        
        # Routing prediction to client or escalate to stage 2 (XGBoost Model)
        if (
            stage1_prob <= self.low_threshold or stage1_prob >=self.high_threshold
        ):
            prediction=int(stage1_prob >= self.stage1_decision_threshold)

            return {
                "prediction": prediction,
                "stage_used": "stage1",
                "probability":float(stage1_prob)
            }
        
        stage2_pred=int(self.stage2_model.predict(X_scaled)[0])

        return {
            "prediction":stage2_pred,
            "stage_used":"stage2 (XGBoost Model)",
            "probability":float(stage1_prob)
        }