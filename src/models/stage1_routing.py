import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

def route_predictions(probs, low_threshold=0.3, high_threshold=0.7):
    """
    Apply uncertainty-based routing to probabilities.
    """

    probs = np.asarray(probs)

    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be < high_threshold")

    uncertain_mask = (probs > low_threshold) & (probs < high_threshold)
    confident_mask = ~uncertain_mask
    escalation_rate = uncertain_mask.mean()

    logger.info(f"LOW={low_threshold} | HIGH={high_threshold}")
    logger.info(f"Escalation rate: {escalation_rate:.4f}")
    logger.info(f"Confident samples: {confident_mask.sum()}")
    logger.info(f"Uncertain samples: {uncertain_mask.sum()}")

    return {
        "uncertain_mask": uncertain_mask,
        "confident_mask": confident_mask,
        "escalation_rate": escalation_rate
    }