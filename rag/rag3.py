import os
import pickle
import time
from typing import List

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


def load_faiss_index(index_path: str) -> faiss.Index:
    return faiss.read_index(index_path)


def load_chunks(pkl_path: str) -> List[str]:
    with open(pkl_path, "rb") as file:
        return pickle.load(file)


def process_query(query: str, index: faiss.Index, chunks: List[str], model: SentenceTransformer, top_k: int = 5) -> \
List[str]:
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


def generate_response_api(query: str, context: str, api_url: str, model: str, api_key: str) -> str:
    messages = [
        {"role": "system", "content": f"Use the following context to answer the user's question: {context}"},
        {"role": "user", "content": query}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"


class RAGPipeline:
    def __init__(self, index_path: str, chunks_path: str, embedding_model: str,
                 api_url: str, llm_model: str, api_key: str):
        self.index = load_faiss_index(index_path)
        self.chunks = load_chunks(chunks_path)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.api_url = api_url
        self.llm_model = llm_model
        self.api_key = api_key

    def process(self, query: str, top_k: int = 5) -> str:
        relevant_chunks = process_query(query, self.index, self.chunks, self.embedding_model, top_k)
        context = "\n".join(relevant_chunks)
        print(f"context: {context}")
        response = generate_response_api(query, context, self.api_url, self.llm_model, self.api_key)
        return response


if __name__ == "__main__":
    INDEX_PATH = "../embeddings/newsqa-data-v0.index"
    CHUNKS_PATH = "../embeddings/newsqa-data-v0.index.pkl"
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    API_URL = "https://api.together.xyz/v1/chat/completions"
    LLM_MODEL1 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLM_MODEL2 = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

    API_KEY = os.environ.get('TOGETHER_API_KEY')

    if not API_KEY:
        raise ValueError("API Key not set, usage export TOGETHER_API_KEY=API_KEY")

    rag_pipeline1 = RAGPipeline(INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL, API_URL, LLM_MODEL1, API_KEY)
    rag_pipeline2 = RAGPipeline(INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL, API_URL, LLM_MODEL2, API_KEY)

    rags = [rag_pipeline1, rag_pipeline2]

    user_query = "How many years old was the businessman?"

    for rag in rags:
        print(rag.process(user_query))
        # for throttling 429
        time.sleep(5)
