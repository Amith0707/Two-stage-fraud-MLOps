"""
This file is made to allow 
`src\db\insert_transaction.py`
file to insert the transaction data point into inference_logs table 
which is created here
"""
from sqlalchemy import text
from src.data.load_from_db import get_engine

engine = get_engine()

create_table_query = text("""
CREATE TABLE IF NOT EXISTS inference_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,
    prediction TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    stage TEXT NOT NULL
);
""")

with engine.begin() as conn:
    conn.execute(create_table_query)

print("inference_logs table created successfully")