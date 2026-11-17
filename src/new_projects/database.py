
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

db = SQLDatabase(
    engine,
    sample_rows_in_table_info=2
)

# print("Database connected:", db.dialect)
# print("Available tables:", db.get_usable_table_names())
