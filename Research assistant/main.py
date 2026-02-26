from smolagents import CodeAgent
from tools import *
from llm import llm

agent = CodeAgent(
    tools = [retriever_tool],
    model= llm,
    max_steps = 8, #Reasoning steps
    verbosity_level= 2 #To show detailed agent reasoning
    
)


agent.run("What are the key aspects of an Agentic AI System")