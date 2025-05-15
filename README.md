This project implements an intelligent chatbot capable of answering questions about the research paper _"Attention is All You Need"_ or assisting users in contacting the authors. Built using **[LangGraph](https://github.com/langchain-ai/langgraph)**, it intelligently classifies user intent and responds accordingly.

---

## Features

###  Intent Classification
The chatbot uses an LLM to determine the user's intent:
- **Question about the paper**
- **Request to contact an author**

Based on the intent, it dynamically routes the conversation through appropriate logic paths.

### Question Answering via RAG
When the user asks a question about the paper:
- A **Retrieval-Augmented Generation (RAG)** pipeline retrieves relevant content chunks from the "Attention is All You Need" PDF.
- The LLM then generates a concise and relevant answer based on the retrieved context.

### Contacting Authors
If the user wants to contact an author:
- The chatbot initiates a conversation to collect details such as the user's name, email, and message.
- The user is prompted to select a specific author from the paper to whom they would like to send the message.

### Memory Management
- The chatbot uses **SQLite-backed checkpoints** provided by **LangGraph** to preserve memory across conversation turns.
- This ensures context retention and smoother multi-turn dialogues.
