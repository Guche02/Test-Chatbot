from typing_extensions import TypedDict, List, Literal
from langgraph.graph import StateGraph, START
from langchain_core.documents import Document
from langchain_core.messages import trim_messages
from classify_query import classify_chain
from RAG_pipeline import rag_chain
from contact import contact_chain
import sys
import os
import uuid
import sqlite3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class State(TypedDict, total=False):   # allow partial state during classification
    question: str
    category: Literal["Document Retrieval", "Contact"]
    context: List[Document]
    answer: str

def router(state: State) -> str:
    if state["category"] == "Document Retrieval":
        return "retrieve"
    elif state["category"] == "Contact":
        return "contact"
    else:
        raise ValueError(f"Unknown category: {state['category']}")

def build_main_pipeline():
    builder = StateGraph(State)

    builder.add_node("classify", classify_chain)
    builder.add_node("rag", rag_chain)
    builder.add_node("contact", contact_chain)

    builder.add_conditional_edges("classify", router, {
        "retrieve": "rag",
        "contact": "contact"
    })

    builder.add_edge(START, "classify")

    return builder.compile()

def main_chain(question: str) -> str:
    pipeline = build_main_pipeline()

    input_state = {
        "question": question,
    }

    response = pipeline.invoke(input_state)
    return response["answer"]

if __name__ == "__main__":
    question = "My name is looza. My email is looza@gmail.com, my phone number is 1234567890."
    answer = main_chain(question)
    print(f"\nAnswer:\n{answer}")
