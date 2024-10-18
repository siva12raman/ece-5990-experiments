"""
Generate embedding for the news_qa dataset
File source - newsqa-data-v1.csv
The story contents are nested under the story directory
"""
import json
import pickle

import pandas as pd;
import enum
import faiss;
import numpy as np;
from sentence_transformers import SentenceTransformer

from util.time_it import time_it

class FileType(enum.Enum):
    TEXT = 1
    JSON = 2
    CSV = 3


class DataSource:

    def __init__(self):
        self.mapping = dict();

    def store(self, key, value):
        self.mapping[key] = value;

    def fetch(self, key):
        if key in self.mapping:
            return self.mapping[key];
        else:
            raise IOError('Key not found in mapping');


class DataLoader:
    def __init__(self, mode=FileType.CSV, file_path=None, key=None):
        if file_path is None:
            raise ValueError("Invalid file path");

        if key is None:
            raise ValueError(f"Key: {key} not found in mapping");

        self.mode = mode;
        self.file_path = file_path;
        self.source = DataSource();
        self.key = key;
        self.load_data();

    def load_data(self):
        if self.mode == FileType.CSV:
            df = pd.read_csv(self.file_path).to_csv(self.file_path, index=False)
            self.source.store(self.key, df);

        elif self.mode == FileType.JSON:
            # Load JSON data
            with open(self.file_path, 'r') as f:
                json_data = json.load(f)

            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            else:
                df = json_data

            self.source.store(self.key, df)

    def get_data(self):
        # todo: code smell - hardcoded attribute
        return self.source.fetch(self.key)[self.key].tolist();


class Embedder:
    def __init__(self, model_name, data_loader, index_path):
        self.embeddings = None
        self.model = SentenceTransformer(model_name);
        self.data_loader = data_loader;
        self.index_path = index_path;
        self.chunks = None

    def _split_chunks(self):
        self.chunks = self.data_loader.get_data();
        with open(self.index_path + ".pkl", "wb") as file:
            pickle.dump(self.chunks, file)

    def generate_embedding(self):
        self._split_chunks();
        self.embeddings = self.model.encode(self.chunks);

    def save_index(self):
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance index
        index.add(np.array(self.embeddings).astype('float32'))
        faiss.write_index(index, self.index_path);


def main():
    print(f"Starting embedder chain")

    dl = DataLoader(FileType.JSON, '../example-banks/newsqa-data-v0.json',
                    'news_content');
    el = Embedder('all-MiniLM-L6-v2', dl, 'newsqa-data-v0.index');
    el.generate_embedding();
    el.save_index();
    print(f"Finished embedder chain")


if __name__ == '__main__':
    with time_it():
        main()
