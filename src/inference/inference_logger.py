import json
from datetime import datetime,timezone
from utils.logger import get_logger

logger = get_logger("inference")
log=get_logger(__name__)
def log_inference(features, prediction, probability, stage):
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": prediction,
        "probability": probability,
        "stage": stage
    }
    log.info("Logged the transaction successfully..")
    logger.info(json.dumps(log_entry))