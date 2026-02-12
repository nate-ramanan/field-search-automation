import psycopg2
import pandas as pd
from ConnectionPool import pool
from configparser import ConfigParser

def get_field_data():
    
    #get all the variables from the config file
    file ='./config.ini'
    config = ConfigParser()
    config.read(file)
    
    batch_size = int(config['database']['batch_size'])
    
    conn = None
    cur = None


    try:
        #connect to the Database
        conn = pool.getconn()
        cur = conn.cursor() #creates a cursor in order to interact with the Database
    
      
        offset = 0
        total_df = pd.DataFrame(columns=['searched_sport_id', 'field_name', 'field_map','field_id','gps_location'])
        
        while True:
            cur.execute("""
                SELECT search_sport_type, field_name, gearth_link, field_search_id,gps_location FROM public.new_google_earth nge 
                ORDER BY nge.field_search_id 
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
            
            rows = cur.fetchall() # Fetch the rows from the query result

            if not rows:
                break  # No more rows, exit the loop
            
            # Create a DataFrame from the query result
            df = pd.DataFrame(rows, columns=['searched_sport_id', 'field_name', 'field_map', 'field_id','gps_location'])
            
            offset += batch_size

            total_df = pd.concat([total_df,df], ignore_index=True)
            

    except Exception as error:
        print(error)
    

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            pool.putconn(conn)

    return total_df