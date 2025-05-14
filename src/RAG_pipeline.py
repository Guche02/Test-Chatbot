from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from embed_data import get_vector_store

vector_store = get_vector_store()

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    # print("---------------------------------------------------------In retrieve, received state from rag_chain:", state)
    retrieved_docs = vector_store.similarity_search(state["question"])
    # print("Retrieved documents: ", retrieved_docs)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    # print("Prompt: ", messages)
    response = llm.invoke(messages)
    return {"answer": response.content}

def build_rag_pipeline():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

def rag_chain(state: State) -> str:
    pipeline = build_rag_pipeline()
    # print("---------------------------------------------------------In rag_chain, received from main.py :", state)
    response = pipeline.invoke({"question": state["question"]})
    return response

if __name__ == "__main__":
    question = {"question": "What is a transformer?"}
    answer = rag_chain(question)
    print(f"Answer: {answer}")