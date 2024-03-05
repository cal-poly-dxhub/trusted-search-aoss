import json
import os

def split_json(input_file, output_dir, chunk_size):
    with open(input_file, 'r') as file:
        data = json.load(file)

    total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)

    for i in range(total_chunks):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        chunk_data = data[start_index:end_index]

        output_file = os.path.join(output_dir, f'{input_file}.output_{i + 1}.json')
        with open(output_file, 'w') as output_file:
            json.dump(chunk_data, output_file, indent=2)

if __name__ == "__main__":
    input_file = '2023_full.json'  # Replace with your input file name
    output_directory = 'out'  # Replace with your desired output directory
    chunk_size = 3  # Replace with your desired chunk size

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    split_json(input_file, output_directory, chunk_size)