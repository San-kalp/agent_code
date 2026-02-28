from langchain_openai import ChatOpenAI
import json

filename = "config.json"

with open(filename,"r") as f :
    config = json.load(f)

api_key = config["API_KEY"]
base_url = config["OPENAI_BASE_URL"]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature = 0, 
    api_key=api_key, 
    base_url=base_url
)