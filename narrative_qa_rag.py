import os
import pandas as pd
import openai
import uuid
from tqdm import tqdm  

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

openai.api_key = os.getenv("OPENAI_API_KEY")
DATASET_PATH = '/Users/kremerr/Documents/GitHub/RARR/archive/narrative_qa/summaries.csv'
PERSIST_PATH = '/Users/kremerr/Documents/GitHub/RARR/vecdb_versions/narrative_qa_vecdb'


def create_vecdb():

    # Initialize chromadb client, embedding model, and text splitter
    miniLM_ef = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

    vecdb = Chroma(persist_directory=PERSIST_PATH, 
                   embedding_function=miniLM_ef,
                   collection_name='narrative_qa_collection'
                )
    print('Chromadb initiated...')
    
    # Load summaries.csv
    df = pd.read_csv(DATASET_PATH)
    df.pop('summary_tokenized')

    loader = DataFrameLoader(data_frame=df, page_content_column='summary')
    docs = loader.load()

    # Split documents
    splits = text_splitter.split_documents(docs)
    
    print("Documents loaded and split...")

    texts = []
    ids = []
    metadatas = []
    for doc in splits:
        texts.append(doc.page_content)
        ids.append(str(uuid.uuid4()))
        metadatas.append(doc.metadata) 
    print("Adding texts to vector store...")
    vecdb.add_texts(ids=ids, texts=texts, metadatas=metadatas)
    print("Documents added.")

def generate_queries():
    queries = []
    # Complete implementation
    return queries

def retrieve_evidence(query, n=1):
    evidence = []
    
    return evidence


def main():
    create_vecdb()



if __name__ == "__main__":
    main()
