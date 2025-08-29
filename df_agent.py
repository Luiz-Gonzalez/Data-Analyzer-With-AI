from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain.agents.agent_types import AgentType

df = pd.read_csv("df_rent.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(model="gpt-4.1-mini"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.invoke("Quantas linhas existem no conjunto de dados?")
