import os
import json
import sqlite3

# File paths
correct_file = 'correct-merged.json'
incorrect_file = 'incorrect_merged.json'
failed_queries_file = 'failed_queries.json'

# Load queries from files
def load_queries(file_path):
    with open(file_path, 'r') as file:
        queries = json.load(file)
    return queries

# Determine the database path based on db_id
def get_db_path(db_id):
    if db_id is None:
        return None

    database_folders = ['database', 'test_database']
    for folder in database_folders:
        db_path = os.path.join(folder, db_id, f'{db_id}.sqlite')
        if os.path.exists(db_path):
            return db_path
    return None

# Fetch sample values for columns in the database
def fetch_sample_values(db_id):
    sample_values = {}
    db_path = get_db_path(db_id)
    if db_path is None:
        return sample_values

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for column in columns:
                column_name = column[1]
                cursor.execute(f"SELECT {column_name} FROM {table_name} LIMIT 1")
                sample_value = cursor.fetchone()
                if sample_value:
                    sample_values[column_name] = sample_value[0]
    except Exception as e:
        print(f"Error fetching sample values: {e}")
    finally:
        conn.close()

    return sample_values

# Preprocess the query to replace 'value' with sample values or appropriate defaults
def preprocess_query(query, sample_values):
    query = query.replace("limit value", "LIMIT 1")
    query = query.replace("> = value", ">= 2")  # Default value for counts or numbers
    query = query.replace("< = value", "<= 2")  # Default value for counts or numbers
    query = query.replace("! = value", "!= 'sample'")
    query = query.replace("!= value", "!= 'sample'")
    query = query.replace("HAVING count ( * ) > = value", "HAVING count(*) >= 2")
    query = query.replace("ORDER BY count ( * ) desc limit value", "ORDER BY count(*) DESC LIMIT 1")
    query = query.replace("= value", "= 'sample'")  # Default value for text
    query = query.replace("LIKE value", "LIKE '%sample%'")

    for column, sample_value in sample_values.items():
        if isinstance(sample_value, str):
            query = query.replace("value", f"'{sample_value}'")
        else:
            query = query.replace("value", str(sample_value))
    return query

# Run a single query and return the error if it fails
def run_query(query_info):
    db_id = query_info.get('db_id')
    query = query_info.get('query')

    if db_id is None or query is None:
        return query_info, "Missing db_id or query"

    db_path = get_db_path(db_id)
    if db_path is None:
        return query_info, f"Database not found for db_id: {db_id}"

    sample_values = fetch_sample_values(db_id)
    query = preprocess_query(query, sample_values)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()
        return None, None  # No error
    except Exception as e:
        return query_info, str(e)

# Main function to run queries and write failed ones to a file
def main():
    correct_queries = load_queries(correct_file)
    incorrect_queries = load_queries(incorrect_file)
    all_queries = correct_queries + incorrect_queries

    failed_queries = []
    successful_queries = []

    for query_info in all_queries:
        _, error = run_query(query_info)
        if error:
            query_info['error'] = error
            failed_queries.append(query_info)
        else:
            successful_queries.append(query_info)

    with open(failed_queries_file, 'w') as file:
        json.dump(failed_queries, file, indent=4)

    with open('correct_preprocessed.json', 'w') as file:
        json.dump([q for q in successful_queries if q in correct_queries], file, indent=4)

    with open('incorrect_preprocessed.json', 'w') as file:
        json.dump([q for q in successful_queries if q in incorrect_queries], file, indent=4)

    print(f"Completed. Failed queries written to {failed_queries_file}")

if __name__ == "__main__":
    main()
