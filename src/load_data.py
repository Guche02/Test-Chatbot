from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import bs4
import os

def load_data(url_path=None, pdf_folder=None):
    """
    Load data from a URL and/or a local PDF path.
    
    Args:
        url_path (str, optional): The URL to load data from.
        pdf_path (str, optional): The local path of the PDF to load data from.
        
    Returns:
        list: Combined list of documents loaded from provided sources.
    """
    data = []

    # Load data from URL if provided
    if url_path:
        web_loader = WebBaseLoader(
            web_paths=(url_path,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        data.extend(web_loader.load())

    # Load data from PDF if provided
    if pdf_folder and os.path.isdir(pdf_folder):
        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, filename)
                pdf_loader = PyPDFLoader(pdf_path)
                data.extend(pdf_loader.load())

    return data