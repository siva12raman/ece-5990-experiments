import pandas as pd
import json


def map_additional_data(id_value):
    print(f"Fetching story with id: {id_value}")
    # ./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story -> 42d01e187213e86f5fe617fe32e716ff7fa3afc4.story
    id_value = id_value.strip().split('/')[-1];

    with open(f"../../example-banks/stories/{id_value}", "r") as f:
        return f.read();

def csv_to_json(csv_file_path, json_file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Convert DataFrame to a dictionary with records format (list of dictionaries)
    df['news_content'] = df['ls story_id'].apply(map_additional_data)

    data = df.to_dict(orient='records')

    # Write the JSON data to a file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"CSV file '{csv_file_path}' has been converted to JSON and saved as '{json_file_path}'.")


if __name__ == "__main__":
    csv_file = '../../example-banks/newsqa-data-v0.csv'
    json_file = '../../example-banks/newsqa-data-v0.json'
    csv_to_json(csv_file, json_file)