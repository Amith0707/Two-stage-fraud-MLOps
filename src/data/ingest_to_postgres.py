"""
Docstring for src.data.ingest_to_postgres

This is just a one-time thing to manually simulate real-world experience
where data is pulled from DB's
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger=get_logger(__name__)

DB_URL=(
    f"postgresql://{os.getenv("POSTGRES_USER")}:"
    f"{os.getenv("POSTGRES_PASSWORD")}@"
    f"localhost:5432/{os.getenv('POSTGRES_DB')}"
)

def ingest(csv_path:str,table_name:str):
    logger.info("Reading CSV")
    df=pd.read_csv(csv_path)

    engine=create_engine(DB_URL)

    logger.info(f"Writing {len(df)} rows to Postgres table '{table_name}'")
    df.to_sql(table_name,engine,if_exists="replace",index=False)

    logger.info("Ingestion Completed")

if __name__=="__main__":
    ingest("data/raw/creditcard_2023.csv","transactions")
