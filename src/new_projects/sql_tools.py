from langchain_community.agent_toolkits import SQLDatabaseToolkit

from .database import db
from .llm import model 

toolkit = SQLDatabaseToolkit(
    db=db,
    llm = model
)

tools = toolkit.get_tools()

# for tool in tools:
#     print(tool.name, "-", tool.description)