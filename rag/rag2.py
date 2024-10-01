import os
import requests
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_chunks(pkl_path):
    with open(pkl_path, "rb") as file:
        return pickle.load(file)

def process_query(query, index, chunks, model, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_response(query, context):
    url = "https://api.together.xyz/v1/chat/completions"
    
    messages = [
        {"role": "system", "content": "You are an expert on C and C++ programming languages. Use the following context to answer the user's question: " + context},
        {"role": "user", "content": query}
    ]
    
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
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
        "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

def main(query):
    index = load_faiss_index("embeddings/wikipedia_c_cpp.index")
    chunks = load_chunks("embeddings/chunks.pkl")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    relevant_chunks = process_query(query, index, chunks, model)
    context = "\n".join(relevant_chunks)
    print(f"retrieved chunks: {context}")
    print(f"-"* 100)
    
    response = generate_response(query, context)

    return response

if __name__ == "__main__":
    # enter your query here
    # usage - export TOGETHER_API_KEY='<your key>'
    user_query = """
        What are the differences between C and C++?
    """
    answer = main(user_query.strip())
    print(answer)