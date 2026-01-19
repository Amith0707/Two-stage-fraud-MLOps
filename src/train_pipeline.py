import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from src.data.load_from_db import load_dataframe
from src.features.preprocess import split_features
from src.models.train_stage1 import train_stage1
from src.models.stage1_routing import route_predictions
from src.models.train_stage2 import train_stage2
from src.models.final_decision import compute_final_predictions
from utils.logger import get_logger

logger=get_logger(__name__)

LOW_THRESHOLD=0.3
HIGH_THRESHOLD=0.7

def run_training_pipeline():
    """
    This function is used to log the ML Experiments perform and log the model performance
    based on their metrics and models themselves as artifacts to enable tracking and easy
    visualization on what parameters the model seems to be performing well
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Monitoring Training Pipelines")
    # Loading Data
    df=load_dataframe("SELECT * FROM transactions")
    X,y=split_features(df)

    with mlflow.start_run(run_name="two_stage_fraud_system"):
        
        mlflow.set_tag("Training Pipeline","To collect information on both stages")
        # Stage-01
        stage1_out=train_stage1(X,y)
        mlflow.set_tag("stage1_model", "LogisticRegression")

        # Logistic Regression model parameters
        mlflow.log_param("stage1_threshold",0.5)
        mlflow.log_params(stage1_out["model"].get_params())

        mlflow.log_param("low_threshold",LOW_THRESHOLD)
        mlflow.log_param("high_threshold",HIGH_THRESHOLD)
        
        # Logistic Regression metrics
        mlflow.log_metric("stage1_recall",stage1_out["recall"])
        mlflow.log_metric("stage1_accuracy",stage1_out["accuracy"])
        mlflow.log_metric("stage1_precision",stage1_out["precision"])

        # Routing
        routing_val=route_predictions(
            stage1_out["val_probs"],
            low_threshold=LOW_THRESHOLD,
            high_threshold=HIGH_THRESHOLD
        )

        escalation_rate=routing_val["escalation_rate"]
        
        # Stage-02
        stage2_out=train_stage2(
            X_train=stage1_out["X_train"],
            X_val=stage1_out["X_val"],
            y_train=stage1_out["y_train"],
            y_val=stage1_out["y_val"],
            train_probs=stage1_out["train_probs"],
            val_probs=stage1_out["val_probs"],
            low_threshold=LOW_THRESHOLD,high_threshold=HIGH_THRESHOLD
        )
        # XGBoost Model Parameters
        model2=stage2_out["model"]
        important_params = {
            "n_estimators": model2.n_estimators,
            "max_depth": model2.max_depth,
            "learning_rate": model2.learning_rate,
            "subsample": model2.subsample,
            "colsample_bytree": model2.colsample_bytree,
            "tree_method": model2.tree_method
        }
        mlflow.log_params(important_params)
        # XGBoost Metrics
        mlflow.log_metric("stage2_recall",stage2_out["recall"])
        mlflow.log_metric("stage2_accuracy",stage2_out["accuracy"])
        mlflow.log_metric("stage2_precision",stage2_out["precision"])

        # system level metrics
        stage1_val_preds=stage1_out["y_val_pred"]
        stage2_val_preds=stage2_out["y_val_pred"]
        final_preds=compute_final_predictions(
            stage1_preds=stage1_val_preds,
            stage2_preds=stage2_out["y_val_pred"],
            y_true=stage1_out["y_val"],
            uncertain_mask=routing_val["uncertain_mask"]
            )
        
        # Logging system metrics
        mlflow.log_metric("system_recall",final_preds["recall"])
        mlflow.log_metric("system_accuracy",final_preds["accuracy"])
        mlflow.log_metric("system_precision",final_preds["precision"])
        mlflow.log_metric("escalation_rate",escalation_rate) 

        # Storing inference signatures for models
        input_example_stage1=stage1_out["X_val"].iloc[:5]
        signature_stage1=infer_signature(input_example_stage1,stage1_val_preds[:5])
        
        input_example_stage2=stage2_out["X_val_stage2"].iloc[:5]
        signature_stage2=infer_signature(input_example_stage2,stage2_val_preds[:5])

        # logging models as artifacts
        mlflow.sklearn.log_model(
            stage1_out["model"],
            name="stage1_model",
            signature=signature_stage1,
            input_example=input_example_stage1
        )

        mlflow.xgboost.log_model(
            stage2_out["model"],
            name="stage2_model",
            signature=signature_stage2,
            input_example=input_example_stage2
        )

if __name__=="__main__":
    print("Starting MLFLOW TRACKING...")
    run_training_pipeline()