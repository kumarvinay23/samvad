from langchain.chat_models import ChatOpenAI
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage

from dotenv import load_dotenv

load_dotenv()



# Initialize LLM
llm = ChatOpenAI(temperature=0)



# Define a sample tool (replace with your actual tools)
def get_current_time(topic: str) -> str:
    """Returns the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(
        name="Current Time",
        func=get_current_time,
        description="Useful for getting the current time.",
    ),
]

# Create memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create prompt for the agent
system_message = SystemMessage(content="You are a helpful assistant.")
prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = ConversationalChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    prompt=prompt,
    verbose=True, # set to true to see the agent's internal reasoning
)

# Create the agent executor
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True
)

# Example conversation
user_input1 = "What time is it?"
response1 = agent_chain.invoke(input=user_input1)
print(response1)

user_input2 = "Thanks, that's helpful."
response2 = agent_chain.invoke(input=user_input2)
print(response2)

user_input3 = "What did I just ask you?"
response3 = agent_chain.invoke(input=user_input3)
print(response3)