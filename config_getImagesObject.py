import psycopg2
import pandas as pd
from configparser import ConfigParser
from ConnectionPool import pool

def get_field_data(pool):
    
    #get all the variables from the config file
    file ='./config.ini'
    config = ConfigParser()
    config.read(file)
    
    batch_size = int(config['database']['batch_size'])
    
    conn = None
    cur = None


    try:
        #connect to the database
        conn = pool.getconn()
        cur = conn.cursor() #creates a cursor in order to interact with the database
    
      
        offset = 0
        total_df = pd.DataFrame(columns=['searched_sport_id', 'field_name', 'field_map','field_id','sport_name','nge_object_id'])
        
        while True:
            cur.execute("""
                SELECT * FROM (
                SELECT nge.search_sport_type as searched_sport_id, nge.field_name, obj.adjusted_gps_location as field_map, nge.field_search_id as field_id, obj.detected_sport as sport_name, obj.nge_object_id
                FROM public.new_google_earth nge
                JOIN nge_object obj ON obj.field_search_id = nge.field_search_id
                UNION
                SELECT nge.search_sport_type as searched_sport_id, nge.field_name, nge.gearth_link as field_map, nge.field_search_id as field_id, 'orig' as sport_name, null as nge_object_id
                FROM public.new_google_earth nge
                ) AS combined
                SELECT
                    nge.search_sport_type,
                    nge.field_name,
                    obj.image_url as field_map,
                    nge.field_search_id as field_id,
                    obj.sport_name,
                    obj.nge_object_id
                FROM public.new_google_earth nge
                INNER JOIN public.nge_object obj
                    ON obj.field_search_id = nge.field_search_id
                WHERE obj.nge_object_id IS NOT NULL
                  AND obj.sport_name IN ('Baseball', 'Basketball', 'Golf', 'Soccer', 'Stadium', 'Tennis')
                ORDER BY nge.field_search_id, obj.nge_object_id
                LIMIT %s OFFSET %s
                """, (batch_size, offset))

            
            rows = cur.fetchall() # Fetch the rows from the query result

            if not rows:
                break  # No more rows, exit the loop
            
            # Create a DataFrame from the query result
            df = pd.DataFrame(rows, columns=['searched_sport_id', 'field_name', 'field_map', 'field_id','sport_name','nge_object_id'])
            
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