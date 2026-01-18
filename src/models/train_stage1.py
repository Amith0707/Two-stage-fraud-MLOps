"""
Stage -1 Model training: Logistic Regression
Responsible for training and evaluating the fast screening model.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_score

from utils.logger import get_logger

logger=get_logger(__name__)

def train_stage1(X,y,test_size=0.35, random_stage=42):
    """
    Docstring for train_stage1
    
    :param X: Input Features 
    :param y: Output Feature- Class {0,1}
    :param test_size: Decide what % of dataset should be training set
    :param random_stage:Set randomness
    """
    logger.info("Splitting train and validation data..")

    X_train,X_val,y_train,y_val=train_test_split(
        X,y,test_size=test_size,stratify=y,random_state=random_stage
    )

    logger.info("Scaling features for the Logistic Regression model..")

    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_val_scaled=scaler.transform(X_val)

    logger.info("Training the model(Logistic Regression- Stage-01)")
    model=LogisticRegression(
        class_weight="balanced",max_iter=1000
    )

    # Training the model
    model.fit(X_train_scaled,y_train)

    # Making model predictions
    logger.info("Making predictions..")
    y_probs=model.predict_proba(X_val_scaled)[:,1]
    y_pred=(y_probs>0.5).astype(int) # rounding off to 0 or 1 with 0.5 as threshold

    recall=recall_score(y_true=y_val,y_pred=y_pred)
    precision=precision_score(y_true=y_val,y_pred=y_pred)

    logger.info(f"Stage-1 recall: {recall:.4f}")
    logger.info(f"Stage-1 precision: {precision:.4f}")


    return {
        "model":model,
        "scaler":scaler,
        
        "X_val":X_val,
        "y_val":y_val,
        "y_probs":y_probs,
        "y_preds":y_pred
    }