"""
This file is to insert the transaction data 
passed during inference into a new postgres table
and not into the existing DB from where our dataset
for model exists.
"""
import json
from datetime import datetime,timezone
from sqlalchemy import text
from src.data.load_from_db import get_engine
from utils.logger import get_logger

logger=get_logger(__name__)

def insert_transaction(features:dict,prediction:str,probability:float,stage:str):

    engine=get_engine()
    insert_query=text("""
        INSERT INTO inference_logs(
                      timestamp,
                      features,
                      prediction,
                      probability,
                      stage
                      )
                      VALUES(
                      :timestamp,
                      :features,
                      :prediction,
                      :probability,
                      :stage
                      )
        """)
    payload={
        "timestamp": datetime.now(timezone.utc),
        "features": json.dumps(features),
        "prediction": prediction,
        "probability": probability,
        "stage": stage
    }
    try:
        with engine.begin() as conn:
            conn.execute(insert_query, payload)
            print("="*50)
        logger.info("Inference transaction inserted into Postgres sucessfully")

    except Exception as e:
        print("="*50)
        logger.error(f"Failed to insert inference transaction: {e}")
