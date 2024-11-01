import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

class Embeddings:

    def get_available_embeddings():
        return ['small', 'large']

    def get_embeddings(select_embedding):
        if select_embedding.lower() == 'small':
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            except Exception as e:
                raise Exception(f"Cannot load small embedding model. Error: {e}")
        
        elif select_embedding.lower() == 'large':
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            except Exception as e:
                raise Exception(f"Cannot load large embedding model. Error: {e}")
        else:
            raise ValueError("Unsupported embedding. Choose 'small' or 'large'.")
        
        return embeddings