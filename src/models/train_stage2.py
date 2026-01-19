import os
import json
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import recall_score,precision_score,accuracy_score
from src.models.stage1_routing import route_predictions
from utils.logger import get_logger


logger=get_logger(__name__)

def train_stage2(X_train,y_train,train_probs,X_val,y_val,val_probs,
                 low_threshold:float=0.3,high_threshold:float=0.7):
    
    logger.info("Applying routing logic to training data..")
    train_routing=route_predictions(train_probs,low_threshold=low_threshold,high_threshold=high_threshold)
    val_routing=route_predictions(val_probs,low_threshold=low_threshold,high_threshold=high_threshold)

    # Filtering the uncertain samples
    train_mask=train_routing["uncertain_mask"]
    val_mask=val_routing["uncertain_mask"]

    X_train_Stage2=X_train.loc[train_mask]
    y_train_Stage2=y_train.loc[train_mask]

    X_val_Stage2=X_val.loc[val_mask]
    y_val_Stage2=y_val.loc[val_mask]

    logger.info(f"Stage -2 Training Samples..{len(X_train_Stage2)}")
    logger.info(f"Stage -2 Validation Samples..{len(X_val_Stage2)}")

    if len(X_train_Stage2)==0:
        raise ValueError("No uncertain samples available for stage 2 training..")
    
    # Model Training and evaluation
    logger.info("Training Stage-2 Model (XGBoost)")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train_Stage2,y_train_Stage2)
    logger.info("Evaluating Stage-2 on uncertain validation samples")
    y_val_pred=model.predict(X_val_Stage2)

    recall = recall_score(y_val_Stage2, y_val_pred)
    precision = precision_score(y_val_Stage2, y_val_pred)
    accuracy = accuracy_score(y_val_Stage2, y_val_pred)

    logger.info(f"Stage-2 Recall: {recall:.4f}")
    logger.info(f"Stage-2 Precision: {precision:.4f}")
    logger.info(f"Stage-2 Accuracy: {accuracy:.4f}")


    return{
        "model":model,
        "X_val_stage2":X_val_Stage2,
        "y_val_stage2":y_val_Stage2,
        "y_val_pred":y_val_pred,
        
        "recall":recall,
        "precision":precision,
        "accuracy":accuracy
    }

def save_stage2_artifacts(model,artifact_dir="artifacts/stage2"):
    """
    This function is created to save model parameters as artifacts to run in the 
    inference pipeline when client requests for a model prediction.
    """
    os.makedirs(artifact_dir,exist_ok=True)
    joblib.dump(model,f"{artifact_dir}/model.pkl")
    
    logger.info("Stage-2 Artifacts Saved")