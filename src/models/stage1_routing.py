import numpy as np
from utils.logger import get_logger

logger=get_logger(__name__)

def route_predictions(probs,low_threshold:float=0.3,high_threshold:float=0.7):
    """
    Route predictions based on uncertainity

    Paramters:
    ---------------
    :param probs: array like stage 1 predicted probabilites for positive class.
    :param low_threshold: Lower bound for uncertainity by Logiscitc Regression
    :type low_threshold: float
    :param high_threshold: Upper bound for uncertainity by Logiscitc Regression
    :type high_threshold: float

    Returns
    --------------
    dict
        Dictionary containing masks and statistics
    """

    probs=np.asarray(probs)

    if low_threshold >= high_threshold:
        raise ValueError("low threshold can't be greater than or equal to high threshold..")
    

    logger.info("Applying routing logic..")
    logger.info(f"LOW={low_threshold} | HIGH={high_threshold}")

    # Masks are boolean arrays here..
    uncertain_mask=(probs>low_threshold) & (probs<high_threshold)
    confident_mask= ~uncertain_mask

    escalation_rate=uncertain_mask.mean()

    logger.info(f"Escalation Rate:{escalation_rate:.4f}")
    logger.info(f"Confident Samples: {confident_mask.sum()}")
    logger.info(f"Uncertain Samples: {uncertain_mask.sum()}")

    return{
        "uncertain_mask":uncertain_mask,
        "confident_mask":confident_mask,
        "escalation_rate":escalation_rate
    }