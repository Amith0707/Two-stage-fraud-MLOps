"""
Docstring for main-

This file is the main controller of our entire project.
All sub-modules of our project is executed from this main file.

"""

import os
import pandas as pd
from dotenv import load_dotenv
from src.data.load_from_db import load_dataframe
from utils.logger import get_logger

logger=get_logger(__name__)

df=load_dataframe("SELECT * from transactions")
logger.info("Pulled the data from postgres from main file..")