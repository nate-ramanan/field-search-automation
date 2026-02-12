import psycopg2
import pandas as pd
from ConnectionPool import pool
from configparser import ConfigParser

def get_field_data():
    
    conn = None
    cur = None


    try:
        #connect to the database
        conn = pool.getconn()
        
        cur = conn.cursor() #creates a cursor in order to interact with the database
    
      
        offset = 0
        total_df = pd.DataFrame(columns=['sport_type_id', 'field_name', 'gps_location','field_id'])
        

        cur.execute("""select sport_type_id, field_name, 
            split_part(split_part(field_map,'@',2),',',1) || ',' ||
            split_part(split_part(field_map,'@',2),',',2) as gps_location,field_id
            from field f
            join facility fa on fa.facility_id = f.facility_id
            join address a on a.address_id = fa.address_id
            where sport_type_id=79 and (a.city ilike 'danville' or a.city ilike 'san mateo')
            and a.gps_location ilike '%,%' and field_map is not null;     """)

        rows = cur.fetchall() # Fetch the rows from the query result
        # Create a DataFrame from the query result
        df = pd.DataFrame(rows, columns=['sport_type_id', 'field_name', 'gps_location','field_id'])
        total_df = pd.concat([total_df,df], ignore_index=True)
            

    except Exception as error:
        print(error)
    

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            pool.putconn(conn)

    return total_df


