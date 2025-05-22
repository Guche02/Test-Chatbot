from typing_extensions import TypedDict, List, Literal
from langgraph.graph import StateGraph, START
from langchain_core.documents import Document
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))   # This adds the src directory to the Python module search path.

from src.chatbot.classify_query import classify_chain
from src.chatbot.RAG_pipeline import rag_chain
from src.chatbot.contact import contact_chain

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

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

@app.get("/")
def index():
    return {"title": "Test Chatbot", "message": "Ask me anything! Or book an appointment!"}

@app.post("/chat", response_model=AnswerResponse)   # response_model ensures that the response follows the AnswerResponse schema
def chat(question_request: QuestionRequest):
    state = {"question": question_request.question}
    pipeline = build_main_pipeline()
    response = pipeline.invoke(state)
    return AnswerResponse(answer=response["answer"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)