import cv2
import requests
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
from config_getImages import get_field_data
from ConnectionPool import pool

# Set up the config file
KEY = 'AIzaSyC5cT2KgRuUuz51GQ71DvY8gB_VN8O8EtE'
file = r'C:\Users\adrgu\field-search-automation\config.ini'
config = ConfigParser()
config.read(file)
print("Loaded Config Sections:", config.sections())                      
print("Model Paths Config:", dict(config.items('model_paths')))      

# Configure the object detection model
obd_model_path = config.get('model_paths', 'obd_model')
obd_model = YOLO(obd_model_path, verbose=False)

# Configure the display names for the object detection model
display_names = {
    0: 'Expressway-Service-area', 1: 'Expressway-toll-station', 2: 'airplane',
    3: 'airport', 4: 'Baseball', 5: 'Basketball',
    6: 'bridge', 7: 'chimney', 8: 'dam', 9: 'Golf',
    10: 'Soccer', 11: 'harbor', 12: 'overpass',
    13: 'ship', 14: 'Stadium', 15: 'storagetank',
    16: 'Tennis', 17: 'trainstation', 18: 'vehicle',
    19: 'windmill'
}

# Mapping internal workflow IDs to OpenStreetMap/Overpass standard tags
SPORT_TAGS = {
    8: "baseball",
    9: "basketball",
    79: "soccer",
    87: "tennis",
    90: "volleyball"
}

def getImage(lat, lon, key, zoom, width, height):
    """Generates the URL for a small satellite image via Google Static Maps."""
    if lat != 'Error' and lon != 'Error':
        url = "https://maps.googleapis.com/maps/api/staticmap?key={}&center={},{}&zoom={}&size={}x{}&maptype=satellite".format(
            key, lat, lon, zoom, width, height
        )
        return url
    else:
        return 'Error'

def get_satellite_image_array(gps_location, zoom_level=18, size=(800, 850)):
    """Fetches image raw bytes using coordinates and turns it into a NumPy Array."""
    lat = gps_location['latitude']
    lon = gps_location['longitude']
    image_url = getImage(lat, lon, KEY, zoom_level, size[0], size[1])
    if image_url == 'Error':
        print("Error: Could not generate image URL.")
        return None
    try:
        response = requests.get(image_url, headers={"User-Agent": "SportsFacilityFinder/1.0"})
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error fetching/processing satellite image: {e}")
        return None

def fetch_zip_bbox_via_photon(zip_code, state_code):
    """Uses Photon API Geocoder to extract a clean bounding box for the ZIP area."""
    url = "https://photon.komoot.io/api/"
    params = {"q": f"{zip_code}, {state_code}, United States", "limit": 1}
    try:
        print(f"Resolving boundary box via Photon for ZIP: {zip_code}")
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
        if data.get("features"):
            feat = data["features"][0]
            extent = feat.get("properties", {}).get("extent")
            if extent and len(extent) == 4:
                # Photon order: [minLon, minLat, maxLon, maxLat]
                # Overpass expects: (minLat, minLon, maxLat, maxLon)
                return (extent[1], extent[0], extent[3], extent[2])
    except Exception as e:
        print(f"Photon bounding geocoding failed: {e}")
    return None

def get_fields_from_overpass(bbox, sport_name):
    """Executes structured Qlever/Overpass interpreter queries over a localized area."""
    endpoint = "https://overpass-api.de/api/interpreter"
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Overpass Query targeting both pitches and active sport markers
    query = f"""
    [out:json][timeout:30];
    (
      nwr["leisure"="pitch"]["sport"="{sport_name}"]({min_lat},{min_lon},{max_lat},{max_lon});
      nwr["leisure"="stadium"]["sport"="{sport_name}"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    try:
        print(f"Querying Overpass for sport: {sport_name}")
        response = requests.post(endpoint, data={"data": query}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Overpass API evaluation failed for {sport_name}: {e}")
        return {"elements": []}

def parse_osm_elements(osm_data, search_sport_type, city, state, postal_code):
    """Normalizes unstructured OpenStreetMap Nodes, Ways, and Relations into the schema."""
    fields = []
    elements = osm_data.get("elements", [])
    for el in elements:
        tags = el.get("tags", {})
        
        # Calculate coordinate tracking based on element structure types
        if "center" in el:
            lat = el["center"]["lat"]
            lon = el["center"]["lon"]
        elif "lat" in el and "lon" in el:
            lat = el["lat"]
            lon = el["lon"]
        else:
            continue # Incomplete spatial mapping metadata
            
        el_type = el.get("type", "node")
        el_id = el.get("id", 0)
        osm_pseudo_id = f"osm/{el_type}/{el_id}"
        
        # Standard fallback string generations if data properties are empty
        raw_name = tags.get("name", tags.get("description", f"OSM {tags.get('leisure', 'Facility')} Location"))
        street = tags.get("addr:street", "")
        house_num = tags.get("addr:housenumber", "")
        formatted_address = f"{house_num} {street}, {city}, {state} {postal_code}".strip(", ")
        
        field = {
            'field_name': raw_name,
            'formatted_address': formatted_address,
            'postal_code': postal_code,
            'street': f"{house_num} {street}".strip(),
            'city': city,
            'state': state,
            'original_gps_location': {'latitude': float(lat), 'longitude': float(lon)},
            'gplace_id': osm_pseudo_id, # Safely maps to the unique constraint column key
            'search_sport_type': search_sport_type,
            'gearth_link': f"https://earth.google.com/web/@{lat},{lng},4.1972381a,15000d" if 'lng' in locals() else f"https://earth.google.com/web/@{lat},{lon},4.1972381a,15000d",
            'modified_fields': []
        }
        fields.append(field)
    return fields

def object_detection_based_modification_by_class(field_search_id, img_array, obd_model, original_gps_location, target_class_ids, zoom_level):
    try:
        results = obd_model.predict(source=img_array, save=False, conf=0.45, iou=0.65, stream=False)
    except Exception as e:
        print(f"Error during object detection: {e}")
        return []

    new_gps_locations = []
    if results and results[0].boxes is not None:
        height, width, _ = img_array.shape
        original_center = (width // 2, height // 2)
        for box in results[0].boxes:
            class_id = int(box.cls)
            if class_id in target_class_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                new_coordinates, ground_resolutions = recenter_image(x1, x2, y1, y2, original_center, original_gps_location, zoom_level)
                
                box_description = f"box: (x1, y1) = ({x1},{y1}), (x2, y2) = ({x2},{y2}), dim = {x2 - x1}x{y2 - y1}"
                image_description = f"image: size = {width}x{height}"
                ground_res_desc = f"old_res = {ground_resolutions['old_ground_resolution']:.4f}, new_res = {ground_resolutions['new_ground_resolution']:.4f}"
                description = f"{box_description} {image_description} {ground_res_desc} new_gps: {new_coordinates['latitude']},{new_coordinates['longitude']}"
                
                final_gps_location = {'field_search_id': field_search_id, 'class_id': class_id, 'description': description}
                final_gps_location.update(new_coordinates)            
                new_gps_locations.append(final_gps_location)        
    return new_gps_locations

def recenter_image(x1, x2, y1, y2, original_center, original_gps_location, zoom_level):
    new_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    offset_x = (new_center[0] - original_center[0])
    offset_y = (new_center[1] - original_center[1])

    earth_radius_meters = 6378137
    earth_circumference_meters = 40075017

    transformer_4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    center_x_3857, center_y_3857 = transformer_4326_to_3857.transform(original_gps_location['longitude'], original_gps_location['latitude'])

    old_ground_resolution = earth_circumference_meters * np.cos(np.radians(original_gps_location['latitude'])) / (256 * (2 ** zoom_level))
    ground_resolution = (np.cos(original_gps_location['latitude'] * np.pi/180) * 2 * np.pi * earth_radius_meters) / (256 * 2 ** zoom_level)

    meters_offset_x = offset_x * ground_resolution
    meters_offset_y = -offset_y * ground_resolution
    new_x_3857 = center_x_3857 + meters_offset_x
    new_y_3857 = center_y_3857 + meters_offset_y

    transformer_3857_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    new_lon, new_lat = transformer_3857_to_4326.transform(new_x_3857, new_y_3857)
    return {'latitude': float(new_lat), 'longitude': float(new_lon)}, {'old_ground_resolution': old_ground_resolution, 'new_ground_resolution': ground_resolution}

def save_object_data(fields):
    if 'database' not in config: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        all_values = []
        for field in fields:
            class_id = field['class_id']
            field_type_name = display_names.get(class_id, f'Unknown Field Type {class_id}')
            new_gps_str = f"{field['latitude']},{field['longitude']}"
            all_values.append(cur.mogrify("(%s,%s,%s,%s)", (
                field['field_search_id'], field_type_name,
                f"https://earth.google.com/web/@{new_gps_str},4.1972381a,15000d", field['description'])
            ).decode('utf-8'))
        if all_values:
            query = "INSERT INTO public.nge_object (field_search_id, sport_name, image_url, description) VALUES %s"
            cur.execute(query % ",".join(all_values))
            conn.commit()
            print(f"Successfully bulk inserted {len(all_values)} object analysis entries.")
    except Exception as error:
        print("Error during database save_object_data execution:", error)
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)

def save_field_data(fields):
    if 'database' not in config: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        all_values = []
        for field in fields:
            if not field['modified_fields']:
                all_values.append(cur.mogrify("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (
                    field['field_name'], '', field['formatted_address'], field['postal_code'],
                    field['street'], field['city'], field['state'],
                    f"{field['original_gps_location']['latitude']},{field['original_gps_location']['longitude']}",
                    field['gplace_id'], field['search_sport_type'], field['gearth_link'], '', ''
                )).decode('utf-8'))
            else:
                for modified_field in field['modified_fields']:
                    class_id = modified_field['class_id']
                    field_type_name = display_names.get(class_id, f'Unknown Field Type {class_id}')
                    new_gps_str = f"{modified_field['latitude']},{modified_field['longitude']}"
                    all_values.append(cur.mogrify("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (
                        field['field_name'], field_type_name, field['formatted_address'], field['postal_code'],
                        field['street'], field['city'], field['state'],
                        f"{field['original_gps_location']['latitude']},{field['original_gps_location']['longitude']}",
                        field['gplace_id'], class_id, field['gearth_link'],
                        f"https://earth.google.com/web/@{new_gps_str},4.1972381a,15000d", new_gps_str
                    )).decode('utf-8'))
        if all_values:
            query = """
            INSERT INTO public.new_google_earth (
                field_name, object_sport, formatted_address, postal_code, street, city, state, gps_location, gplace_id, search_sport_type, gearth_link, object_gearth_link, object_gps_location
            ) VALUES %s
            """
            cur.execute(query % ",".join(all_values))
            conn.commit()
            print(f"Successfully bulk inserted {len(all_values)} new_google_earth rows.")
    except Exception as error:
        print("Error during database save_field_data execution:", error)
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)

def fetch_cities(state_code):
    if 'database' not in config: return []
    conn, cur, cities = None, None, []
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        query = """
        SELECT DISTINCT city FROM public.postal_code pc
        JOIN state_province sp ON sp.state_id = pc.state_id
        WHERE sp.state_code = %s
        """
        cur.execute(query, (state_code,))
        cities = [row[0] for row in cur.fetchall()]
    except Exception as error:
        print("Error fetching cities:", error)
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)
    return cities

def fetch_zip_codes(city, state_code):
    if 'database' not in config: return []
    conn, cur, zip_codes = None, None, []
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        query = """
<<<<<<< HEAD
        SELECT pc.postal_code FROM public.postal_code pc
        JOIN state_province sp ON sp.state_id = pc.state_id
        WHERE UPPER(pc.city) = %s AND sp.state_code = %s
          AND NOT EXISTS (
              SELECT 1 FROM maps_api_log mal
              WHERE mal.city = %s AND mal.state = %s AND mal.zip_code = pc.postal_code
=======
        SELECT pc.postal_code
        FROM public.postal_code pc
        JOIN state_province sp ON sp.state_id = pc.state_id
        WHERE UPPER(pc.city) = %s
          AND sp.state_code = %s
          AND NOT EXISTS (
              SELECT 1
              FROM maps_api_log mal
              WHERE mal.city = %s
                AND mal.state = %s
                AND mal.zip_code = pc.postal_code
>>>>>>> b8f9beb550708ef0a931fbe050aaa8db686b3d19
          )
        """
        cur.execute(query, (city.upper().strip(), state_code, city.upper().strip(), state_code))
        zip_codes = [row[0] for row in cur.fetchall()]
    except Exception as error:
        print("Error fetching zip codes:", error)
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)
    return zip_codes

def log_processed(state, city, zip_code):
    if 'database' not in config: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        query = "INSERT INTO maps_api_log (state, city, zip_code) VALUES (%s, %s, %s)"
        cur.execute(query, (state, city, zip_code))
        conn.commit()
    except Exception as error:
        print("Error updating execution logs:", error)
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)

def delete_duplicates():
    if 'database' not in config: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
<<<<<<< HEAD
        delete_query = """
        WITH ranked AS (
            SELECT field_search_id, ROW_NUMBER() OVER (
                PARTITION BY COALESCE(gplace_id, ''), COALESCE(field_name, ''), COALESCE(gps_location, '')
                ORDER BY field_search_id
            ) AS rn FROM public.new_google_earth
        ),
        duplicate_ids AS (SELECT field_search_id FROM ranked WHERE rn > 1),
        deleted_objects AS (
            DELETE FROM public.nge_object WHERE field_search_id IN (SELECT field_search_id FROM duplicate_ids)
            RETURNING field_search_id
        )
        DELETE FROM public.new_google_earth WHERE field_search_id IN (SELECT field_search_id FROM duplicate_ids);
=======

        print("Deleting duplicates by (gplace_id, field_name, gps_location)")
        delete_query = """
        WITH ranked AS (
    SELECT
        field_search_id,
        ROW_NUMBER() OVER (
            PARTITION BY
                COALESCE(gplace_id, ''),
                COALESCE(field_name, ''),
                COALESCE(gps_location, '')
            ORDER BY field_search_id
        ) AS rn
    FROM public.new_google_earth
),
duplicate_ids AS (
    SELECT field_search_id FROM ranked WHERE rn > 1
),
deleted_objects AS (
    DELETE FROM public.nge_object
    WHERE field_search_id IN (SELECT field_search_id FROM duplicate_ids)
    RETURNING field_search_id
)
DELETE FROM public.new_google_earth
WHERE field_search_id IN (SELECT field_search_id FROM duplicate_ids);
>>>>>>> b8f9beb550708ef0a931fbe050aaa8db686b3d19
        """
        cur.execute(delete_query)
        rows_deleted = cur.rowcount
        conn.commit()
<<<<<<< HEAD
        print(f"Cleaned up duplicate rows successfully.")
=======
        print(f"Deleted {rows_deleted} duplicate rows.")

>>>>>>> b8f9beb550708ef0a931fbe050aaa8db686b3d19
    except Exception as error:
        print("Error during database duplicate cleaning:", error)
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)


if __name__ == "__main__":
    state_code = input("Enter state code (e.g. WA): ").strip().upper()
    city_to_add = input("Enter city to add (e.g. SEATTLE): ").strip().upper()
    single_zip = input("Enter zip to check (e.g. 98115): ").strip()
    
    cities_list = fetch_cities(state_code)
    
    for city in cities_list:
        if city.upper() != city_to_add:
            continue
        zip_codes_list = fetch_zip_codes(city, state_code)

        for zip_code in zip_codes_list:
            if zip_code != single_zip:
                continue
                
            # Step 1: Query location bounding parameters using Photon API geocoding
            bbox = fetch_zip_bbox_via_photon(zip_code, state_code)
            if not bbox:
                print(f"Could not compute bounding boundaries for ZIP code: {zip_code}. Skipping.")
                continue
                
            all_fields = []
<<<<<<< HEAD
            
            # Step 2: Loop through each sport category and fetch records via Overpass
            for sport_id, sport_name in SPORT_TAGS.items():
                print(f"Fetching fields for sport tag: '{sport_name}' inside zip code {zip_code}")
                osm_json_response = get_fields_from_overpass(bbox, sport_name)
                fields_from_search = parse_osm_elements(osm_json_response, sport_id, city, state_code, zip_code)
                
                for field in fields_from_search:
=======
            schools_data = fetch_schools(zip_code=zip_code)
            schools_locations = []
            for school in schools_data:
                gps = school[3]
                if not gps or ',' not in gps:
                    continue  # skip bad / missing GPS
                lat, lon = gps.split(",")
                try:
                    lat_f, lon_f = float(lat), float(lon)
                except ValueError:
                    continue

                school_dict = {
                    'field_name': school[0],
                    'formatted_address': None,
                    'postal_code': None,
                    'street': None,
                    'city': school[5],
                    'state': school[1],
                    'original_gps_location': {'latitude': lat_f, 'longitude': lon_f},
                    'gplace_id': None,
                    'search_sport_type': 100,
                    'gearth_link': school[11],
                    'modified_fields': []
                }
                schools_locations.append(school_dict)

            seen_gplace_ids = set()
                
            for sport, sport_type_id in sport_types.items():
                print(f"Fetching fields for {sport} in {city}, {state_code} for zip code {zip_code}")
                
                query = f"sports fields and facilities for {sport} in {zip_code}, {state_code}"
                xml_data = get_fields(query)
                fields_from_search = parse_fields(xml_data, sport_type_id)
                
                for field in fields_from_search:
                    gplace_id = field.get('gplace_id')
                    if gplace_id and gplace_id in seen_gplace_ids:
                        print(f"Skipping duplicate facility: {field['field_name']}")
                        continue
                    if gplace_id:
                        seen_gplace_ids.add(gplace_id)
>>>>>>> b8f9beb550708ef0a931fbe050aaa8db686b3d19
                    all_fields.append(field)

            if all_fields:
                save_field_data(all_fields)
                log_processed(state_code, city, zip_code)
                
    delete_duplicates()
    
    # Run spatial recentering via the local neural networks
    df = get_field_data()
    print(df)
    nge_object = []
    all_target_class_ids = [4, 5, 9, 10, 14, 16]
    for index, row in df.iterrows():
        print(f"Processing field: {row['field_name']} at {row['gps_location']}")
        lat, lon = row['gps_location'].split(",")
        gps_loc = {'latitude': float(lat), 'longitude': float(lon)}
        img_array = get_satellite_image_array(gps_loc)
        if img_array is not None:
            modified_locations = object_detection_based_modification_by_class(
                row['field_id'], img_array, obd_model, gps_loc, all_target_class_ids, zoom_level=18
            )
            nge_object.extend(modified_locations)
            
    save_object_data(nge_object)
<<<<<<< HEAD
    print("Data input process completed utilizing OpenStreetMap data engines successfully.")
=======
    print("Data input process completed")
>>>>>>> b8f9beb550708ef0a931fbe050aaa8db686b3d19
