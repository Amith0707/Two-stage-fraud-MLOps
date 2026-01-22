"""
Docstring for data.load_from_db
Utility for loading data from Postgres
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from utils.logger import get_logger

logger=get_logger(__name__)


def get_engine():
    """
    Creating and return a SQLAlchemy engine using env variables
    """
    load_dotenv()

    user=os.getenv("POSTGRES_USER")
    password=os.getenv("POSTGRES_PASSWORD")
    db=os.getenv("POSTGRES_DB")
    host=os.getenv("POSTGRES_HOST","localhost")
    port=os.getenv("POSTGRES_PORT","5432")

    if not all([user,password,db]):
        raise RuntimeError("Database environment variables are not set..")
    
    db_url=f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(db_url)

def load_dataframe(query:str)->pd.DataFrame:
    """
    Load data from PostgreSQL using a SQL Query
    """
    logger.info("Connecting to Postgres..")
    engine=get_engine()

    logger.info("Executing Query")
    df=pd.read_sql(query,engine) # interesting
    logger.info(f"Loaded Dataframe with shape: {df.shape}")
    return df