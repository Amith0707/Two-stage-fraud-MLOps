"""
Feature preprocessing module.
This script is only to split features and target 
"""

import os
import pandas as pd
from utils.logger import get_logger

logger=get_logger(__name__)

def split_features(df:pd.DataFrame):
    """
    Docstring for split_features
    
    :param df: A dataframe
    :type df: pd.DataFrame

    Output: None as we just split the data into X and y
    """

    logger.info("Splitting features and target")

    X=df.drop(columns=["id","Class"])
    y=df["Class"]

    logger.info(f"Features Shape:{X.shape}")
    logger.info(f"Target Shape:{X.shape}")

    return X,y