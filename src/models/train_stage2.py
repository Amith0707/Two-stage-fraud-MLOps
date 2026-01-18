import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import recall_score,precision_score,accuracy_score
from models.stage1_routing import route_predictions
from utils.logger import get_logger


logger=get_logger(__name__)

def train_stage2(X_train,y_train,train_probs,X_val,y_val,val_probs,
                 low_threshold:float=0.3,high_threshold:float=0.7):
    
    logger.info("Applying routing logic to training data..")
    train_routing=route_predictions(train_probs,low_threshold=low_threshold,high_threshold=high_threshold)
    





