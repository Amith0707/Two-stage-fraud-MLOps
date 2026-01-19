import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score
from utils.logger import get_logger

logger = get_logger(__name__)

def compute_final_predictions(stage1_preds,stage2_preds,uncertain_mask,y_true):
    """
    Merge stage-1 and stage 2 predictions into a single system output
    """

    logger.info("Combining Stage-1 and Stage-2 Predictions..")

    final_preds=stage1_preds.copy()
    # Replacing only those predictions where stage 1 was uncertain
    final_preds[uncertain_mask]=stage2_preds

    # System-level metrics
    recall=recall_score(y_true,final_preds)
    precision = precision_score(y_true, final_preds)
    accuracy = accuracy_score(y_true, final_preds)

    logger.info(f"SYSTEM Recall: {recall:.4f}")
    logger.info(f"SYSTEM Precision: {precision:.4f}")
    logger.info(f"SYSTEM Accuracy: {accuracy:.4f}")

    return {"final_preds":final_preds,"recall":recall,"precision":precision,"accuracy":accuracy}    