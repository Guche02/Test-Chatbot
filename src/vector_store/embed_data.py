from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
load_dotenv()
from src.vector_store.load_data import load_data

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

pc = Pinecone(api_key=api_key)
index_name = "new-index"

if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating it...")
    pc.create_index(
        name=index_name,
        dimension=768,  # Change this to match your embedding model
        metric="cosine",
        spec=ServerlessSpec(
        cloud="aws",
       region="us-east-1"
    ))
else:
    print(f"Index '{index_name}' already exists.")
    
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

def embed_data(data):
    """
    Function to embed data into a vector store.
    Args:
        data (List[Document]): List of documents to be embedded.
    """

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(data)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)
    print(f"Indexed {len(all_splits)} chunks into the vector store.")

if __name__ == "__main__":
    data = load_data(pdf_folder="D:\\AI Stuff\\ML-April-2025\\projects\\langchain\\data")
    embed_data(data)