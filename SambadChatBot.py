from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import tool
from langchain import LLMChain
from transformers import pipeline

from GenerateEmbedding import VectorDBDataSource


import requests
import json
import os

from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
MONGODB_ATLAS_URI = os.environ.get("MONGODB_ATLAS_URI", default=None)
MONGODB_DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME", default=None)
MONGODB_COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION_NAME", default=None)


# Replace with your actual API key

# Define your APIs as LangChain tools
def get_weather(city: str) -> str:
    """Useful for getting the current weather in a city."""
    try:
        response = requests.get(f"https://api.weatherapi.com/v1/current.json?key=YOUR_WEATHER_API_KEY&q={city}") # Replace with your weather api key.
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        weather_data = response.json()
        return f"The current temperature in {city} is {weather_data['current']['temp_c']}°C."
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except KeyError:
        return "Could not parse weather information from the API."

def get_studyMaterial(topic: str) -> str:
    """Useful for getting the Study material of a subject or course."""
    try:


        vector_db = VectorDBDataSource(OPENAI_API_KEY, MONGODB_ATLAS_URI, MONGODB_DATABASE_NAME, MONGODB_COLLECTION_NAME).get_vector_db()
        results = vector_db.similarity_search(topic)
        # Extract relevant document content

        result_string = "\n".join([doc.page_content for doc in results])
        return result_string

    except Exception as e:
        return f"Error fetching Study Material for : {topic}"


# Initialize the LLM and tools


model = ChatOpenAI()


# Construct the prompt
prefix = """Answer the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!

Question: {input}
{agent_scratchpad}"""



classifier = pipeline("zero-shot-classification")
candidate_labels = ["weather", "study material"]

def decide_api(user_input):
    """Decides which API to call using zero-shot classification."""
    result = classifier(user_input, candidate_labels)
    predicted_label = result['labels'][0]

    if predicted_label == "weather":
        # Extract city from input
        words = user_input.split()
        city = words[-1] # very simple city extraction. Improve this.
        return get_weather(city)
    elif predicted_label == "study material":

        return get_studyMaterial(user_input)
    else:
        return "I'm not sure which API to call for that."

# Example usage

user_input1 = "give  the  information about PRASANNA"
response1 = decide_api(user_input1)
print(response1)
