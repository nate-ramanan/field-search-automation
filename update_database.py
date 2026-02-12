import pandas as pd
import psycopg2
from psycopg2 import sql
from ConnectionPool import pool
from configparser import ConfigParser
import os

def connect_db(config):
    try:
        conn = pool.getconn()
        print("Database connection successful.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def update_existing_data(conn, df, table_name, id_column, columns_to_update):
    cur = conn.cursor()
    conn.autocommit = False

    missing_cols = [col for col in columns_to_update if col not in df.columns]
    if missing_cols:
        print(f"Error: The following columns are missing in the loaded Excel file: {', '.join(missing_cols)}")
        print("Please ensure your Excel file contains all the necessary columns for update.")
        cur.close()
        return

    update_set_clauses = [
        sql.SQL("{} = %s").format(sql.Identifier(col))
        for col in columns_to_update
    ]

    update_query = sql.SQL("""
        UPDATE {table}
        SET {update_clauses}
        WHERE {id_col} = %s;
    """).format(
        table=sql.Identifier(table_name),
        update_clauses=sql.SQL(', ').join(update_set_clauses),
        id_col=sql.Identifier(id_column)
    )

    data_for_update = []
    for index, row in df.iterrows():
        row_values = [row[col] for col in columns_to_update]
        row_values.append(row[id_column])
        data_for_update.append(tuple(row_values))

    try:
        print(f"Attempting to update {len(data_for_update)} rows in '{table_name}' where '{id_column}' matches existing records...")
        cur.executemany(update_query, data_for_update)
        conn.commit()
        print(f"Update operation for {len(data_for_update)} records completed. "
              "Only rows with matching 'field_search_id' were updated.")
    except psycopg2.Error as e:
        print(f"Error during update operation: {e}")
        conn.rollback()
    finally:
        cur.close()
        pool.putconn(conn)

def insert_data(conn, df, table_name, columns_to_insert):
    """
    Inserts data from a pandas DataFrame into an SQL table.
    """
    cur = conn.cursor()
    conn.autocommit = False

    # Check for missing columns
    missing_cols = [col for col in columns_to_insert if col not in df.columns]
    if missing_cols:
        print(f"Error: The following columns are missing in the DataFrame: {', '.join(missing_cols)}")
        print("Please ensure your DataFrame contains all the necessary columns.")
        cur.close()
        return

    # Create the correct INSERT query with column names and value placeholders
    columns = sql.SQL(', ').join(map(sql.Identifier, columns_to_insert))
    values = sql.SQL(', ').join(sql.Placeholder() * len(columns_to_insert))

    insert_query = sql.SQL("""
        INSERT INTO {table} ({columns})
        VALUES ({values})
    """).format(
        table=sql.Identifier(table_name),
        columns=columns,
        values=values
    )

    # Prepare data for insertion
    data_for_insert = [tuple(row) for row in df[columns_to_insert].itertuples(index=False)]

    try:
        print(f"Attempting to insert {len(data_for_insert)} rows into '{table_name}'...")
        cur.executemany(insert_query, data_for_insert)
        conn.commit()
        print(f"Insert operation for {len(data_for_insert)} records completed successfully.")
    except Exception as e:
        print(f"Error during insert operation: {e}")
        conn.rollback()
    finally:
        cur.close()
        pool.putconn(conn)

def update_latest_existing_data_with_df(prefix, table_name, columns_to_update, id_column):
    excel_files = [f for f in os.listdir('.') if f.startswith(f'{prefix}') and f.endswith('.xlsx')]
    if not excel_files:
        print(f"No '{prefix}*.xlsx' files found in the current directory.")
        exit()

    excel_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_excel_file = excel_files[0]

    print(f"Found the latest Excel file for {prefix}: {latest_excel_file}")
    
    user_choice = input(f"Do you want to use '{latest_excel_file}' for database update? (yes/no or enter filename): ").strip().lower()

    if user_choice == 'yes':
        excel_path = latest_excel_file
    elif os.path.exists(user_choice) and user_choice.endswith('.xlsx'):
        excel_path = user_choice
    else:
        print("Invalid choice or file not found. Exiting.")
        exit()

    try:
        df = pd.read_excel(excel_path)
        print(f"Successfully loaded data from {excel_path}.")
        print(df.head())
    except Exception as e:
        print(f"Error loading Excel file '{excel_path}': {e}")
        exit()

    if not df.empty:
        conn = connect_db(config)
        if conn:
            update_existing_data(
                conn, 
                df, 
                table_name, 
                id_column,
                columns_to_update=columns_to_update
            )
            print("Update complete.")
            '''
            pool.putconn(conn)
            conn.close()
            print("Database connection closed.")
            '''
    else:
        print("Loaded DataFrame is empty. No data to update in the database.")

def insert_latest_existing_data_with_df(prefix, table_name, columns_to_insert):
    excel_files = [f for f in os.listdir('.') if f.startswith(f'{prefix}') and f.endswith('.xlsx')]
    if not excel_files:
        print(f"No '{prefix}*.xlsx' files found in the current directory.")
        exit()

    excel_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_excel_file = excel_files[0]

    print(f"Found the latest Excel file for {prefix}: {latest_excel_file}")
    
    user_choice = input(f"Do you want to use '{latest_excel_file}' for database insert? (yes/no or enter filename): ").strip().lower()

    if user_choice == 'yes':
        excel_path = latest_excel_file
    elif os.path.exists(user_choice) and user_choice.endswith('.xlsx'):
        excel_path = user_choice
    else:
        print("Invalid choice or file not found. Exiting.")
        exit()

    try:
        df = pd.read_excel(excel_path)
        print(f"Successfully loaded data from {excel_path}.")
        print(df.head())
    except Exception as e:
        print(f"Error loading Excel file '{excel_path}': {e}")
        exit()

    if not df.empty:
        conn = connect_db(config)
        if conn:
            insert_data(
                conn,
                df,
                table_name,
                columns_to_insert=columns_to_insert
            )
            print("Insert complete.")
    else:
        print("Loaded DataFrame is empty. No data to update in the database.")

if __name__ == '__main__':
    file = './config.ini'
    config = ConfigParser()
    config.read(file)

    table_name_1 = "new_google_earth"
    id_column_1 = "field_search_id"
    columns_to_update_1 = [
        'predicted_field_type',
        'predicted_sport',
        'predicted_large_field_type',
        'predicted_large_sport',
        'detected_sport'
    ]

    table_name_2 = "nge_object"
id_column_2 = "nge_object_id"
columns_to_update_2 = [
    'predicted_field_type',
    'predicted_sport',
    'predicted_field_type_probability',
    'predicted_sport_type_probability',
    'predicted_large_field_type',
    'predicted_large_sport',
    'large_predicted_field_type_probability',
    'large_predicted_sport_type_probability',
    'detected_sport',
    'detect_confidence'
]

    #new_google_earth table update
'''
        'predicted_field_type',
        'predicted_sport',
        'predicted_field_type_probability',
        'predicted_sport_type_probability',
        'predicted_large_field_type',
        'predicted_large_sport',
        'large_predicted_field_type_probability',
        'large_predicted_sport_type_probability',
        'detected_sport',
    '''
'''
    update_latest_existing_data_with_df(
        prefix='combined_results_', 
        table_name=table_name_1,
        id_column=id_column_1,
        columns_to_update=columns_to_update_1
    )
    '''
    

    #nge_object table update
'''
    insert_latest_existing_data_with_df(
        prefix='combined_results_',
        table_name=table_name_2,
        columns_to_insert=columns_to_insert_2
    )
    '''
    
update_latest_existing_data_with_df(
        prefix='combined_results_', 
        table_name=table_name_2,
        id_column=id_column_2,
        columns_to_update=columns_to_update_2
)
    