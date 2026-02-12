import psycopg2
import pandas as pd
from configparser import ConfigParser
from ConnectionPool import pool

def save_field_data(df):

    conn = None
    cur = None

    try:
        conn = pool.getconn()
        
        cur = conn.cursor() #creates a cursor in order to interact with the database
        
        
        for index, row in df.iterrows():
            field_search_id = row['field_search_id']
            predicted_field_type = row['predicted_field_type']
            predicted_sport_id = row['predicted_sport_id']
            predicted_field_type_probability = row['predicted_field_type_probability']
            predicted_sport_type_probability = row['predicted_sport_type_probability']
            
            
            query = "UPDATE public.new_google_earth SET predicted_field_type=%s, predicted_sport_id = %s, predicted_field_type_probability = %s, predicted_sport_type_probability = %s WHERE field_search_id = %s"
            values = (predicted_field_type,predicted_sport_id,predicted_field_type_probability,predicted_sport_type_probability,field_search_id)
            cur.execute(query,values)
            
            conn.commit()
                
    except Exception as error:
        print(error)
    

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            pool.putconn(conn)

    
