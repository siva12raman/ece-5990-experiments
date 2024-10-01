from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load the pre-trained model
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# Read the content of a file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Generate sentence embeddings
def generate_embeddings(model, text_chunks):
    return model.encode(text_chunks)

# Split content into chunks by line
def split_content_by_line(content):
    return content.split('\n')

# Create FAISS index and add embeddings
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(np.array(embeddings).astype('float32'))
    return index

# Save FAISS index to disk
def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)

# Save data using pickle
def save_pickle(data, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

# Main function to execute the process
def main():
    model = load_model()

    # Load and prepare data
    c_language_content = read_file('../example-banks/c_language.txt')
    c_plus_plus_content = read_file('../example-banks/c_plus_plus_language.txt')

    # Split content into chunks
    c_chunks = split_content_by_line(c_language_content)
    cpp_chunks = split_content_by_line(c_plus_plus_content)
    
    # Combine chunks from both files
    all_chunks = c_chunks + cpp_chunks

    # Generate embeddings
    embeddings = generate_embeddings(model, all_chunks)

    # Create and save FAISS index
    index = create_faiss_index(embeddings)
    save_faiss_index(index, "wikipedia_c_cpp.index")

    # Save the chunks for future reference
    save_pickle(all_chunks, "chunks.pkl")

# Run the main function
if __name__ == "__main__":
    main()