from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import trim_messages, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Sequence
from model import Person
from langchain_core.tools import tool

import uuid
import sqlite3

from typing import NotRequired

class State(TypedDict):
    question: NotRequired[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer: NotRequired[str]

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

extraction_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "Ask the user to provide the value.",
        ),
        ("human", "{messages}"),
    ]
)

structured_llm = llm.with_structured_output(schema=Person)

@tool
def extract_info(messages: list):
    """
    Extracts structured user information from the latest message in the conversation.

    This tool performs the following steps:
    1. Trims the message history using a token-aware trimmer to focus on the most recent relevant content.
    2. Fills a prompt template with the user's latest message to prepare it for structured extraction.
    3. Uses a language model to extract structured information such as name, email, phone number, and appointment date.
    4. Returns the extracted information wrapped in a messages format compatible with the conversation state.

    Returns:
        dict: A dictionary with a single key "messages" containing the extracted response from the model.
    """

    trimmed_messages = trimmer.invoke(messages)
    print("--------------------------In extract info, Trimmed messages: ", trimmed_messages)
    filled_prompt = extraction_prompt_template.invoke(
        {
            "messages": trimmed_messages[-1].content,
        }
    )
    print("-----------------------------------------------------In extract_info, Filled prompt: ", filled_prompt)
    response = structured_llm.invoke(filled_prompt)
    return {"message": response}

tools = [extract_info]

main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an intelligent reasoning-and-acting (ReAct) agent designed to help users contact the author of a paper.

        You analyze the user's query step-by-step and determine **which tools should be used**, and **which steps do not require tools**. 
        Use your reasoning ability to decide which step to continue and **when to invoke a tool**.

        Follow this structured process carefully. Avoid repeating any steps that are already completed.

        ---

        ### Goal: Help the user contact the author of the paper.

        ### Reasoning and Action Flow:

        1. **Ask the user** to provide the following:
        - Name
        - Email
        - Phone Number

        2. STRICTLY **Use the extract_info tool** to extract this information from the user's message. Do not generate the response yourself.

        4. **Ask the user to choose** which author they want to contact from the following options:
        - Author 1 – looza@gmail.com
        - Author 2 – john@gmail.com
        - Author 3 – jojobailey@gmail.com

        5. **Ask the user** to provide the message they want to send to the selected author.

        6. **Ask for confirmation** before sending the message.

        7. **Once confirmed**, summarize the conversation and finalize the action.

        ---

        ### Final Output Format:
        Use this format at the end when all information is collected and confirmed:

        User's Info:
        Name:
        Email:
        Phone:

        Author's Info:
        Email:

        Message:

        Status:
        Message sent to the author successfully.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm_with_tools = llm.bind_tools(tools)

def contact_step(state: State) -> dict:
    trimmed_messages = trimmer.invoke(state["messages"])
    print("Trimmed messages: ", trimmed_messages)

    filled_prompt = main_prompt.invoke({"messages": trimmed_messages})
    print("-----------------------------------------------------In contact step, Filled prompt: ", filled_prompt)

    response = llm_with_tools.invoke(filled_prompt)
    print("-----------------------------------------------------In contact step, Response: ", response)

    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Tool selected: {tool_call['name']} with arguments {tool_call['args']}")
        tool_result = extract_info.invoke(state)

        return {"messages": [*state["messages"], response, tool_result["response"]]}

    return {"messages": [*state["messages"], response]}

def graph_builder():
    graph = StateGraph(State)
    graph.add_edge(START, "contact")
    graph.add_node("contact", contact_step)
    
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    return graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "a222"}}

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