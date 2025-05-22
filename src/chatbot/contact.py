from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import trim_messages, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Sequence
from langchain_core.tools import tool
import sqlite3
from typing import NotRequired
from src.chatbot.model import Person

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
            "Extract the following information from the message and return it as JSON with these keys: "
            "`name`, `email`, `phone`, `appointment_date`. "
            "If not available, leave them null.\n"
            "Message: {messages}"
        ),
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
    return {"messages": response}

tools = [extract_info]

main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an intelligent reasoning-and-acting (ReAct) agent designed to help users contact the author of a paper.

        You analyze the user's query step-by-step and use extract_info tool to extract the user's info. 
        Use your reasoning ability to decide which step to continue.

        Follow this structured process carefully. Avoid repeating any steps that are already completed.

        ---

        ### Goal: Help the user contact the author of the paper.

        ### Reasoning and Action Flow:

        1. **Ask the user** to provide the following:
        - Name
        - Email
        - Phone Number
        - Date of Appointment 

        2. STRICTLY **Use the extract_info tool from available tools** to extract this information from the user's message. Do not generate the response yourself.

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
        print("-----------------------------------------------------In contact step, Tool result: ", tool_result)   
        return {"messages": tool_result}
    
    return {"messages": [*state["messages"], response]}

def graph_builder():
    graph = StateGraph(State)
    graph.add_edge(START, "contact")
    graph.add_node("contact", contact_step)
    
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    return graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "a342"}}

def contact_chain(state: State):
    print("---------------------------------------------------------In contact_chain, received from main.py :", state)
    question = state["question"]
    print("---------------------------------------------------------In contact_chain, After filtering the question from state :", question)
    input_messages = [HumanMessage(question)]

    graph = graph_builder()
    response = graph.invoke({"messages": input_messages}, config)
    return {"answer" : response["messages"][-1].content}

if __name__ == "__main__":
    test_state: State = {
        "question": "Hi, my name is Anjali. You can reach me at anjali@example.com or call me on 9812345678. I want to schedule a meeting for next Friday.",
    }
    answer = contact_chain(test_state)
    print(f"Answer: {answer}")

    # test_messages = [
    #     HumanMessage(content="Hi, my name is Anjali. You can reach me at anjali@example.com or call me on 9812345678. I want to schedule a meeting for next Friday.")
    # ]
    # # Test the extract_info tool directly
    # result = extract_info.invoke({"messages": test_messages})
    # print("Extracted Info:\n", result)