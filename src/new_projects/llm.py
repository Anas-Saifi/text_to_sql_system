from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os


model = ChatOpenAI(api_key = os.environ["HIVE_API_KEY"], model = "deepseek-ai/deepseek-v4.1-flash-1017", base_url = "https://api-cdn.thehive.ai/api/v3", temperature = 0)