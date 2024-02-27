import os
import pandas as pd
import openai
from tqdm import tqdm  

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

openai.api_key = os.getenv("OPENAI_API_KEY")
DATASET_PATH = '/Users/kremerr/Documents/GitHub/RARR/archive/narrative_qa/summaries.csv'
PERSIST_PATH = '/Users/kremerr/Documents/GitHub/RARR/vecdb_versions/narrative_qa_vecdb'


def create_vecdb():

    # Initialize the vector store, embedding model, and text splitter
    embedding = OpenAIEmbeddings()
    vecdb = Chroma(persist_directory=PERSIST_PATH, 
                   collection_name='narrative_qa_collection',
                   embedding_function=embedding)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

    # Load summaries.csv
    df = pd.read_csv(DATASET_PATH)
    df.pop('summary_tokenized')
    df.pop('set')

    loader = DataFrameLoader(data_frame=df, page_content_column='summary')
    docs = loader.load()

    # Split documents
    splits = text_splitter.split_documents(docs)
    
    # Add documents to vector store
    vecdb.add_documents(splits)
    vecdb.persist()

    print(vecdb._client.get_collection('narrative_qa_collection').peek())
    return vecdb


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
