import requests
import xml.etree.ElementTree as ET
import psycopg2
import pandas as pd
from configparser import ConfigParser
import os
import json
import numpy as np
import pyproj
from pyproj import Transformer
import collections.abc
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

KEY = 'AIzaSyC5cT2KgRuUuz51GQ71DvY8gB_VN8O8EtE'
file = r'c:\Users\owner\Documents\Gameplay_FieldAutomation_full_v3\Gameplay_FieldAutomation_full_v3\config.ini'
config = ConfigParser()
config.read(file)
print(config.sections())                      
print(dict(config.items('model_paths')))      

#configure the object detection model
obd_model_path = config.get('model_paths','obd_model')
obd_model = YOLO(obd_model_path, verbose=False)
display_names = {
    0: 'Expressway-Service-area', 1: 'Expressway-toll-station', 2: 'airplane',
    3: 'airport', 4: 'Baseball', 5: 'Basketball',
    6: 'bridge', 7: 'chimney', 8: 'dam', 9: 'Golf',
    10: 'Soccer', 11: 'harbor', 12: 'overpass',
    13: 'ship', 14: 'Stadium', 15: 'storagetank',
    16: 'Tennis', 17: 'trainstation', 18: 'vehicle',
    19: 'windmill'
}

def getImage(lat, lon, key, zoom, width, height):
    """Generates the URL for a small satellite image."""
    if lat != 'Error' and lon != 'Error':
        url = "https://maps.googleapis.com/maps/api/staticmap?key={}&center={},{}&zoom={}&size={}x{}&maptype=satellite".format(
            key,
            lat,
            lon,
            zoom,
            width,
            height
        )
        return url
    else:
        return 'Error'

def get_satellite_image_array(gps_location, zoom_level=18, size=(800, 850)):
    """
    Uses the getImage function to get a URL and then fetches the image
    data, returning it as a NumPy array.
    """
    lat = gps_location['latitude']
    lon = gps_location['longitude']
    
    image_url = getImage(lat, lon, KEY, zoom_level, size[0], size[1])
    
    if image_url == 'Error':
        print("Error: Could not generate image URL.")
        return None
        
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        # Explicitly convert the image to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img)
        return img_array
    except requests.exceptions.RequestException as e:
        print(f"Error fetching satellite image from URL: {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_fields(query: str):
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/xml?key={KEY}&query={query}"
    print(f"Fetching data from URL: {url}")
    response = requests.get(url)
    print(f"Received response with status code: {response.status_code}")
    return response.text

def object_detection_based_modification_by_class(img_array, obd_model, gps_location, target_class_ids, zoom_level):
    '''
    Function to modify gps_location by concentrating on the average center of fields detected 
    by class ID. This version returns a list of GPS locations, one for each field type.
    '''
    try:
        results = obd_model.predict(source=img_array, save=False, conf=0.45, iou=0.65, stream=False)
    except Exception as e:
        print(f"Error during object detection: {e}")
        return []

    gps_location['latitude'] = float(gps_location['latitude'])
    gps_location['longitude'] = float(gps_location['longitude'])
    
    grouped_boxes = collections.defaultdict(list)
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls)
            if class_id in target_class_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                grouped_boxes[class_id].append((x1, y1, x2, y2))
    
    new_gps_locations = []
    
    for class_id, boxes in grouped_boxes.items():
        mean_x1 = np.mean([b[0] for b in boxes])
        mean_y1 = np.mean([b[1] for b in boxes])
        mean_x2 = np.mean([b[2] for b in boxes])
        mean_y2 = np.mean([b[3] for b in boxes])

        height, width, _ = img_array.shape
        
        original_center = (width // 2, height // 2)
        new_center = ((mean_x1 + mean_x2) // 2, (mean_y1 + mean_y2) // 2)
        offset_x = new_center[0] - original_center[0]
        offset_y = new_center[1] - original_center[1]

        transformer_4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        center_lon, center_lat = float(gps_location['longitude']), float(gps_location['latitude'])
        center_x_3857, center_y_3857 = transformer_4326_to_3857.transform(center_lon, center_lat)

        earth_circumference = 40075016.686
        ground_resolution = earth_circumference * np.cos(np.radians(center_lat)) / (256 * (2 ** zoom_level))
        
        meters_offset_x = offset_x * ground_resolution
        meters_offset_y = -offset_y * ground_resolution 

        new_x_3857 = center_x_3857 + meters_offset_x
        new_y_3857 = center_y_3857 + meters_offset_y

        transformer_3857_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        new_lon, new_lat = transformer_3857_to_4326.transform(new_x_3857, new_y_3857)
        
        new_gps_locations.append({'latitude': new_lat, 'longitude': new_lon, 'class_id': class_id})
    
    if not new_gps_locations:
        print("No target objects detected. Returning an empty list.")

    return new_gps_locations

def parse_fields(xml_data: str, search_sport_type: int):
    print("Parsing XML data")
    root = ET.fromstring(xml_data)
    fields = []
    for result in root.findall('result'):
        address = result.find('formatted_address').text
        address_parts = address.split(',')

        if len(address_parts) >= 3:
            street = address_parts[0].strip()
            city = address_parts[-3].strip()
            state_postal = address_parts[-2].strip().split()
            state = state_postal[0]
            postal_code = state_postal[1] if len(state_postal) > 1 else None
        else:
            street, city, state, postal_code = None, None, None, None
        
        lat = result.find('.//location/lat').text
        lng = result.find('.//location/lng').text
        
        field = {
            'field_name': result.find('name').text,
            'formatted_address': address,
            'postal_code': postal_code,
            'street': street,
            'city': city,
            'state': state,
            'original_gps_location': {'latitude': float(lat), 'longitude': float(lng)},
            'gplace_id': result.find('place_id').text,
            'search_sport_type': search_sport_type,
            'gearth_link': f"https://earth.google.com/web/@{lat},{lng},4.1972381a,15000d",
            'modified_fields': []
        }
        fields.append(field)
    return fields

def save_field_data(fields):
    file = './config.ini'
    config = ConfigParser()
    print(f"Reading config file: {file}")
    config.read(file)
    
    if 'database' not in config:
        print(f"Section 'database' not found in {file}")
        return
    
    hostname = config['database']['hostname']
    database = config['database']['database']
    username = config['database']['username']
    password = config['database']['password']
    port_id = config['database']['port_id']
    conn = None
    cur = None

    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=password,
            port=port_id
        )
        cur = conn.cursor()
        
        all_values = []
        for field in fields:
            if not field['modified_fields']:
                # If no fields were detected, save the original location
                all_values.append(cur.mogrify("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (
                    field['field_name'],
                    field['formatted_address'],
                    field['postal_code'],
                    field['street'],
                    field['city'],
                    field['state'],
                    f"{field['original_gps_location']['latitude']},{field['original_gps_location']['longitude']}",
                    field['gplace_id'],
                    field['search_sport_type'],
                    field['gearth_link']
                )).decode('utf-8'))
            else:
                for modified_field in field['modified_fields']:
                    class_id = modified_field['class_id']
                    field_type_name = display_names.get(class_id, f'Unknown Field Type {class_id}')
                    
                    new_gps_str = f"{modified_field['latitude']},{modified_field['longitude']}"
                    
                    new_field_name = f"{field['field_name']} - type:{field_type_name}"
                    
                    all_values.append(cur.mogrify("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (
                        new_field_name,
                        field['formatted_address'],
                        field['postal_code'],
                        field['street'],
                        field['city'],
                        field['state'],
                        new_gps_str,
                        field['gplace_id'],
                        class_id,
                        f"https://earth.google.com/web/@{new_gps_str},4.1972381a,15000d"
                    )).decode('utf-8'))
        
        if all_values:
            query = """
            INSERT INTO public.new_google_earth (
                field_name, formatted_address, postal_code, street, city, state, gps_location, gplace_id, search_sport_type, gearth_link
            ) VALUES %s
            """
            final_query = query % ",".join(all_values)
            print(f"Executing bulk insert for {len(all_values)} fields")
            cur.execute(final_query)
            conn.commit()
        else:
            print("No fields to save.")
                
    except Exception as error:
        print("Error during database operation")
        print(error)
    
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def fetch_cities(state_code):
    file = 'config.ini'
    config = ConfigParser()
    config.read(file)

    if 'database' not in config:
        print(f"Section 'database' not found in {file}")
        return []

    hostname = config['database']['hostname']
    database = config['database']['database']
    username = config['database']['username']
    password = config['database']['password']
    port_id = config['database']['port_id']

    conn = None
    cur = None
    cities = []

    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=password,
            port=port_id
        )
        cur = conn.cursor()

        query = """
        SELECT DISTINCT city FROM public.postal_code pc
        JOIN state_province sp ON sp.state_id = pc.state_id
        WHERE sp.state_code = %s
        """
        cur.execute(query, (state_code,))
        cities = [row[0] for row in cur.fetchall()]
        print(f"Cities found: {cities}")

    except Exception as error:
        print("Error fetching cities from database")
        print(error)

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return cities

def fetch_zip_codes(city, state_code):
    file = 'config.ini'
    config = ConfigParser()
    config.read(file)

    if 'database' not in config:
        print(f"Section 'database' not found in {file}")
        return []

    hostname = config['database']['hostname']
    database = config['database']['database']
    username = config['database']['username']
    password = config['database']['password']
    port_id = config['database']['port_id']

    conn = None
    cur = None
    zip_codes = []

    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=password,
            port=port_id
        )
        cur = conn.cursor()

        query = """
        SELECT postal_code FROM public.postal_code pc
        JOIN state_province sp ON sp.state_id = pc.state_id
        WHERE UPPER(pc.city) = %s AND sp.state_code = %s
        AND postal_code NOT IN (
            SELECT zip_code FROM maps_api_log WHERE city = %s AND state = %s
        )
        """
        cur.execute(query, (city.upper().strip(), state_code, city.upper().strip(), state_code))
        zip_codes = [row[0] for row in cur.fetchall()]
        print(f"Zip codes found: {zip_codes}")

    except Exception as error:
        print("Error fetching zip codes from database")
        print(error)

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return zip_codes

def fetch_schools(zip_code):
    file = 'config.ini'
    config = ConfigParser()
    config.read(file)

    if 'database' not in config:
        print(f"Section 'database' not found in {file}")
        return []

    hostname = config['database']['hostname']
    database = config['database']['database']
    username = config['database']['username']
    password = config['database']['password']
    port_id = config['database']['port_id']

    conn = None
    cur = None

    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=password,
            port=port_id
        )
        cur = conn.cursor()

        query = """
        SELECT * FROM public.schools
        WHERE zip_code = %s;
        """
        cur.execute(query, (zip_code,))
        schools_results = cur.fetchall()

    except Exception as error:
        print("Error fetching schools from database")
        print(error)

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
    
    if schools_results is None:
        print(f"No schools with {zip_code}")

    return schools_results

def log_processed(state, city, zip_code):
    file = 'config.ini'
    config = ConfigParser()
    config.read(file)

    if 'database' not in config:
        print(f"Section 'database' not found in {file}")
        return
    
    hostname = config['database']['hostname']
    database = config['database']['database']
    username = config['database']['username']
    password = config['database']['password']
    port_id = config['database']['port_id']
    conn = None
    cur = None

    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=password,
            port=port_id
        )
        cur = conn.cursor()
        
        query = """
        INSERT INTO maps_api_log (state, city, zip_code)
        VALUES (%s, %s, %s)
        """
        cur.execute(query, (state, city, zip_code))
        
        conn.commit()
                
    except Exception as error:
        print("Error during logging operation")
        print(error)
    
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def delete_duplicates():
    file = 'config.ini'
    config = ConfigParser()
    print(f"Reading config file: {file}")
    config.read(file)
    
    if 'database' not in config:
        print(f"Section 'database' not found in {file}")
        return
    
    hostname = config['database']['hostname']
    database = config['database']['database']
    username = config['database']['username']
    password = config['database']['password']
    port_id = config['database']['port_id']
    conn = None
    cur = None

    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=password,
            port=port_id
        )
        cur = conn.cursor()
        
        print("Deleting duplicates based on gplace_id and search_sport_type")
        delete_query_1 = """
        DELETE FROM public.new_google_earth
        WHERE ctid IN (
            SELECT ctid
            FROM (
                SELECT ctid,
                       ROW_NUMBER() OVER(PARTITION BY gplace_id, search_sport_type ORDER BY ctid) AS rn
                FROM public.new_google_earth
            ) t WHERE t.rn > 1
        );
        """
        cur.execute(delete_query_1)
        
        # print("Deleting duplicates based on field_name (partial match)")
        # delete_query_2 = """
        # DELETE FROM public.new_google_earth
        # WHERE field_search_id IN (
        #     SELECT c.field_search_id
        #     FROM public.new_google_earth p
        #     JOIN public.new_google_earth c
        #     ON trim(split_part(p.field_name, '-', 1)) = trim(split_part(c.field_name, '-', 1))
        #     AND p.field_name != c.field_name
        #     AND p.search_sport_type = c.search_sport_type
        #     AND position('-' in c.field_name) <= 0
        # );
        # """
        # cur.execute(delete_query_2)
        
        conn.commit()
                
    except Exception as error:
        print("Error during database operation")
        print(error)
    
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    country = "United States"
    state_code = input("Enter state code: ")
    city_to_add = input("Enter city to add: ")
    city_to_add = city_to_add.upper()
    sport_types = {
        "baseball": 8,
        "basketball": 9,
        "soccer": 79,
        "tennis": 87,
        "volleyball": 90
    }
    
    cities_list = fetch_cities(state_code)
    print(f"Cities in {state_code}, {country}: {cities_list}")

    for city in cities_list:
        if city != city_to_add:
            continue
        zip_codes_list = fetch_zip_codes(city, state_code)
        print(f"Zip codes in {city}: {zip_codes_list}")

        for zip_code in zip_codes_list:
            all_fields = []
            schools_data = fetch_schools(zip_code=zip_code)
            schools_locations = []
            for school in schools_data:
                gps = school[3]
                lat, lon = gps.split(",")
                school_dict = {
                    'field_name': school[0],
                    'formatted_address': None,
                    'postal_code': None,
                    'street': None,
                    'city': school[5],
                    'state': school[1],
                    'original_gps_location': {'latitude': float(lat), 'longitude': float(lon)},
                    'gplace_id': None,
                    'search_sport_type': 100,
                    'gearth_link': school[11],
                    'modified_fields': []
                }
                schools_locations.append(school_dict)
            for school in schools_locations:
                print(f"Processing school: {school['field_name']} at {school['original_gps_location']}")
                
                img_array = get_satellite_image_array(school['original_gps_location'], size=(800, 850)) # For recentering
                if img_array is not None:
                    all_target_class_ids = [4, 5, 9, 10, 14, 16]
                    modified_locations = object_detection_based_modification_by_class(
                        img_array, obd_model, school['original_gps_location'], all_target_class_ids, zoom_level=18
                    )
                    school['modified_fields'] = modified_locations
                else:
                    print("Image array is None")
                print('\n')
            all_fields.extend(schools_locations)

            for sport, sport_type_id in sport_types.items():
                print(f"Fetching fields for {sport} in {city}, {state_code} for zip code {zip_code}")
                
                query = f"sports fields and facilities for {sport} in {zip_code}, {state_code}"
                xml_data = get_fields(query)
                fields_from_search = parse_fields(xml_data, sport_type_id)
                
                for field in fields_from_search:
                    print(f"Processing field: {field['field_name']} at {field['original_gps_location']}")
                    
                    img_array = get_satellite_image_array(field['original_gps_location']) # For recentering
                    if img_array is not None:
                        all_target_class_ids = [4, 5, 9, 10, 14, 16]
                        modified_locations = object_detection_based_modification_by_class(
                            img_array, obd_model, field['original_gps_location'], all_target_class_ids, zoom_level=18
                        )
                        field['modified_fields'] = modified_locations
                all_fields.extend(fields_from_search)

            if all_fields:
                save_field_data(all_fields)
                log_processed(state_code, city, zip_code)
                
    delete_duplicates()
    df = get_field_data()
    nge_object = []
    all_target_class_ids = [4, 5, 9, 10, 14, 16]
    for index, row in df.iterrows():
        print(f"Processing field: {row['field_name']} at {row['original_gps_location']}")

        img_array = get_satellite_image_array(row['original_gps_location'])  # For recentering
        if img_array is not None:
            modified_locations = object_detection_based_modification_by_class(
                row['field_id'],img_array, obd_model, row['gps_location'], all_target_class_ids, zoom_level=18)
        nge_object.extend(modified_locations)
    save_object_data(nge_object)
    print("Data input process completed")
