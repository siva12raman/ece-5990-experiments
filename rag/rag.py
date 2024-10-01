from openai import OpenAI

#insert api key here 
client = OpenAI(api_key='')
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path):
    """
    Load the FAISS index from the specified path.
    """
    return faiss.read_index(index_path)

def load_chunks(pkl_path):
    """
    Load the text chunks (context) from the specified pickle file.
    """
    with open(pkl_path, "rb") as file:
        return pickle.load(file)

def process_query(query, index, chunks, model, top_k=5):
    """
    Given a query, retrieve the top_k most relevant chunks using the FAISS index.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_response(query, relevant_chunks):
    """
    Use OpenAI GPT model to generate a response based on the relevant context.
    """
    context = "\n".join(relevant_chunks)
    messages = [
        {"role": "system", "content": "You are an expert on C and C++ programming languages."},
        {"role": "user", "content": f"Based on the following information about C and C++ programming languages, please answer the query:\n\nContext:\n{context}\n\nQuery: {query}"}
    ]
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=150,
    temperature=0.7)

    return response.choices[0].message.content.strip()

def main(query):
    index = load_faiss_index("embeddings/wikipedia_c_cpp.index")
    chunks = load_chunks("embeddings/chunks.pkl")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    relevant_chunks = process_query(query, index, chunks, model)
    response = generate_response(query, relevant_chunks)

    return response

# Example query
if __name__ == "__main__":
    user_query = "What are the differences between C and C++?"
    answer = main(user_query)
    print(answer)