import pg8000
import numpy as np
import joblib
import time
import io
from concurrent.futures import ThreadPoolExecutor
import sys
import os
from config import *

def load_tensors(table_name):
    # Connect to the PostgreSQL database

    connection = pg8000.connect(database=DATABASE_NAME, user=USER, password=PASSWORD, host=HOST, port=5432)
    cursor = connection.cursor()

    # Execute a query to fetch the tensors from the specified table
    query = f"SELECT tablenames, cellvalues, embeddings FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    # Close the database connection
    cursor.close()
    connection.close()

    # Extract tensors and convert them to NumPy arrays
    tensors = [row[2] for row in rows]
    cell_values = [row[1] for row in rows]
    table_names = [row[0] for row in rows]

    return table_names, cell_values, tensors

def main(start, end, params):
    start_time = time.time()

    # Define the names of tables containing tensors
    table_names = [f"re_tables_{str((i) // 10 + 1).zfill(4)}_{i % 10 + 1}" for i in range(start, end)]
    print(f"Processing tables. Start = {start}, End = {end}")

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(load_tensors, table_names)

    # Separate the results
    sub_table_names, cell_values, embeddings_list = zip(*results)

    print(f"Tables names: {len(sub_table_names)}")
    print(f"cell values length: {len(cell_values)}")

    # Concatenate the loaded tensors into a single embeddings array
    embeddings_array = np.vstack(embeddings_list)
    
    print(f'Embeddings array shape: {embeddings_array.shape}')

    output_dir = f"tables_{start}_to_{end}"
    os.mkdir(output_dir)
    #Create joblib file to save the embedding_array as a file
    joblib.dump(embeddings_array, f'{output_dir}/embeddings_array.joblib')
    joblib.dump(cell_values, f'{output_dir}/cell_values.joblib')
    joblib.dump(sub_table_names, f'{output_dir}/sub_table_names.joblib')

    end_time = time.time()

    runtime = end_time - start_time
    print(f"Creating arrays is finished at. Runtime: {round(runtime, 2)}s")

if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    main(start, end)


