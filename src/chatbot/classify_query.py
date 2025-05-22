from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

prompt = PromptTemplate.from_template(
    """Given the user question, classify into one of the following categories:
     * Document Retrieval 
     * Contact 

     Return only the category name.
     Do not return any other text or explanation.

     Given Question: {question}
     """
)

class State(TypedDict):
    question: str
    category: str

def classify_step(state: State) -> dict:
    chain = prompt | llm | StrOutputParser()
    category = chain.invoke({"question": state["question"]})
    return {"category": category}

def graph_builder():
    graph = StateGraph(State)
    graph.add_edge(START, "classify")
    graph.add_node("classify", classify_step)
    return graph.compile()

def classify_chain(question: str):
    graph = graph_builder()
    response = graph.invoke(question)
    print("Category:", response["category"])
    return response

if __name__ == "__main__":
    response = classify_chain("How do I contact the author?")
    print(response)
