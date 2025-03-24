import asyncio

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, StrOutputParser
from langchain_core.tools import tool
from transformers import pipeline
from googletrans import Translator
import speech_recognition as sr
import os
import io
from pydub import AudioSegment
from GenerateEmbedding import VectorDBDataSource
import requests
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
WEATHER_API_KEY = os.environ.get("YOUR_WEATHER_API_KEY") # Replace with your weather API key
MONGODB_ATLAS_URI= os.environ.get("MONGODB_ATLAS_URI", default=None)
MONGODB_DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME", default=None)
MONGODB_COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION_NAME", default=None)

# --- Initialize Components ---
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
translator = Translator()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")



vector_db = VectorDBDataSource(OPENAI_API_KEY, MONGODB_ATLAS_URI, MONGODB_DATABASE_NAME, MONGODB_COLLECTION_NAME).get_vector_db()

# --- Tools ---

@tool
def get_weather(city: str) -> str:
    """Useful for getting the current weather in a city."""
    try:
        response = requests.get(f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}")
        response.raise_for_status()
        weather_data = response.json()
        return f"The current temperature in {city} is {weather_data['current']['temp_c']}°C."
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except KeyError:
        return "Could not parse weather information from the API."

@tool
def search_knowledge_base(query: str) -> str:
    """Useful for answering questions about Study material. Use this to find information from the knowledge base."""
    try:
        docs = vector_db.similarity_search(query)
        if not docs:
            return "No relevant information found in the knowledge base."
        context = "\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based on the provided context.

            Context:
            {context}

            Question: {question}
            Answer:
            """
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return llm_chain.invoke({"context": context, "question": query})['text']
    except Exception as e:
        return f"Error searching knowledge base: {e}"

tools = [
    get_weather,
    search_knowledge_base,
]

# --- Conversational Agent Setup ---

def create_conversational_agent(user_id):
    """Creates a conversational agent with memory for a specific user."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix="User", ai_prefix="Assistant")

    # Construct the prompt for the agent



    system_message_content = "You are a helpful and friendly chatbot. You will try to answer the user's questions based on the provided context and your knowledge. If you need to use a tool, please use the format: `{{tool_code}}` where tool_code is the name of the tool and the input is the argument to the tool."
    prompt_template = """{system_message}

    Conversation:
    {chat_history}
    User: {input}
    Agent: {agent_scratchpad}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "input", "agent_scratchpad"],
        partial_variables={"system_message": system_message_content},
    )


    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
    return agent_executor


# --- Main Chatbot Logic ---

user_sessions = {}

async def get_user_agent(user_id):
    """Retrieves or creates a conversational agent for a given user."""
    if user_id not in user_sessions:
        user_sessions[user_id] = create_conversational_agent(user_id)
    return user_sessions[user_id]

async def process_voice_input(audio_data):
    """Converts audio to text using Speech Recognition."""
    try:
        r = sr.Recognizer()
        audio_stream = io.BytesIO(audio_data)
        audio = sr.AudioFile(audio_stream)
        with audio as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

async def chatbot_response(user_id, user_input_raw, voice_input=None):
    """Main function to handle user input and generate a response."""
    try:
        user_agent = await get_user_agent(user_id)
        detected_language = 'en'
        english_input = user_input_raw

        if voice_input:
            user_input_text = await process_voice_input(voice_input)
            if "Could not understand audio." in user_input_text or "Could not request results" in user_input_text:
                return {"response": user_input_text, "language": "en"}
            user_input = user_input_text
        else:
            user_input = user_input_raw

        # Detect language and translate to English if necessary
        try:
            detected_language = await translator.detect(user_input)
            if detected_language.lang != 'en':
                translates_english_input = await translator.translate(user_input, dest='en')
                english_input = translates_english_input.text
                print(f"Translated input ({detected_language.lang}): {english_input}")
        except Exception as e:
            print(f"Language detection/translation error: {e}")

        # Initialize agent_scratchpad as an empty list
        agent_scratchpad = []

        # Decide whether to use a tool or general conversation
        classification_result = classifier(english_input, ["weather", "knowledge", "general conversation"])
        predicted_label = classification_result['labels'][0]
        print(f"Predicted Label: {predicted_label}")

        if predicted_label == "weather":
            words = english_input.split()
            city = words[-1] if words else "London"
            response_text = await user_agent.ainvoke(f"What is the weather in {city}?", agent_scratchpad=agent_scratchpad)
        elif predicted_label == "knowledge":
            response_text = await user_agent.ainvoke(english_input, agent_scratchpad=agent_scratchpad)
        elif predicted_label == "general conversation":
            response_text = await user_agent.ainvoke(english_input, agent_scratchpad=agent_scratchpad)
        else:
            response_text = "I'm not sure how to respond to that."

        # Translate the response back to the user's language
        translated_response = response_text
        if detected_language.lang != 'en' and "Error" not in response_text:
            try:
                translated_response = await translator.translate(response_text, dest=detected_language)
            except Exception as e:
                print(f"Translation back to {detected_language} error: {e}")

        return {"response": translated_response.text, "language": detected_language}

    except Exception as e:
        return {"response": f"An error occurred: {e}", "language": "en"}

# --- Example of Invocation from Chatbot Client (Conceptual) ---

async def invoke_chatbot_from_ui(user_id, message, audio_data=None):
    """Simulates invoking the chatbot from a UI or mobile app."""
    response = await chatbot_response(user_id, message, voice_input=audio_data)
    print(f"UI/Mobile App (User: {user_id}): {message}")
    print(f"Chatbot Response: {response['response']}")
    print(f"Detected Language: {response['language']}")
    print("-" * 20)
    return response

async def main():
    user1 = "user123"
    user2 = "another_user"
    audio_data_user1 = None  # Replace with actual audio data for testing voice

    # User 1 Interaction
    await invoke_chatbot_from_ui(user1, "Hello, how are you?")
    await invoke_chatbot_from_ui(user1, "Please provide the information about the Organisational  Behavior.")

    # User 2 Interaction (French)
    await invoke_chatbot_from_ui(user2, "Bonjour, comment ça va?")
    await invoke_chatbot_from_ui(user2, "Quel temps fait-il à Paris?")

if __name__ == "__main__":
    asyncio.run(main())
    # Example with voice input (commented out - needs audio data)
    # with open("sample.wav", "rb") as f:
    #     audio_data = f.read()
    #     invoke_chatbot_from_ui(user1, None, audio_data)