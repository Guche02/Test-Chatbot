from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import trim_messages, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Sequence

import uuid
import sqlite3

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
    """The user is asking about how to contact the author of the paper. 

    Continue the conversation based on the following information.
    If some steps are already done as shown in history, do not repeat them.
     
    1) Ask the user to provide the necessary information about themselves such as: Name, Email, Phone Number.
    2) let them choose which author they want to contact.
     The choices are:
        * Author 1 - looza@gmail.com
        * Author 2 - john@gmail.com
        * Author 3 - jojobailey@gmail.com
    
    3) Ask them to provide the message they want to send to the author.
    4) Finally, ask them to confirm if they want to send the message.

    5) provide the response in the following format based on the history:

    User's info:
    Name:
    Email:
    Phone:

    Author's Info:
    Email:
    
    Message: 

    Status:
    Message sent to the author successfully.

     """
        ), MessagesPlaceholder(variable_name="messages"),
    ]
)

class State(TypedDict):
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer: str

def contact_step(state: State) -> dict:
    trimmed_messages = trimmer.invoke(state["messages"])
    print("Trimmed messages: ", trimmed_messages)
    filled_prompt = prompt.invoke(
       {
          "messages": trimmed_messages,
       }, 
    )  
    print("-----------------------------------------------------In contact step, Filled prompt: ", filled_prompt)
    response = llm.invoke(filled_prompt)
    return {"messages": response}

def graph_builder():
    graph = StateGraph(State)
    graph.add_edge(START, "contact")
    graph.add_node("contact", contact_step)
    
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    return graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

def contact_chain(state: State):
    print("---------------------------------------------------------In contact_chain, received from main.py :", state)

    question = state["question"]
    print("---------------------------------------------------------In contact_chain, After filtering the question from state :", question)
    input_messages = [HumanMessage(question)]

    graph = graph_builder()
    response = graph.invoke({"messages": input_messages}, config)
    return {"answer" : response["messages"][-1].content}

if __name__ == "__main__":
    question = """user: How do I contact the author? """
    answer = contact_chain(question)
    print(f"Answer: {answer}")