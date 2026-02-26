import json
from smolagents import OpenAIServerModel
import os

filename = "config.json"
with open(filename,"r") as file :
    config = json.load(file)

api_key = config["API_KEY"]
base_url = config["OPENAI_BASE_URL"]

#This is a callable class which means we can use it as a function.
llm = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base=base_url,
    api_key=api_key
)




