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
KEY = 'AIzaSyC5cT2KgRuUuz51GQ71DvY8gB_VN8O8EtE' # Note: Be careful exposing your API keys publicly!
file = r'D:\Gameplay_FieldAutomation_full_v3 - Copy\config.ini'
config = ConfigParser()
config.read(file)
print("Loaded Config Sections:", config.sections())                      
print("Model Paths Config:", dict(config.items('model_paths')))      

# Configure the object detection model
obd_model_path = config.get('model_paths', 'obd_model')
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

SPORT_TAGS = {
    8: "baseball",
    9: "basketball",
    79: "soccer",
    87: "tennis",
    90: "volleyball"
}

# ---------------------------------------------------------
# SATELLITE & BOUNDARY FUNCTIONS 
# ---------------------------------------------------------
def getImage(lat, lon, key, zoom, width, height):
    if lat != 'Error' and lon != 'Error':
        return f"https://maps.googleapis.com/maps/api/staticmap?key={key}&center={lat},{lon}&zoom={zoom}&size={width}x{height}&maptype=satellite"
    return 'Error'

def get_satellite_image_array(gps_location, zoom_level=18, size=(800, 850)):
    lat = gps_location['latitude']
    lon = gps_location['longitude']
    image_url = getImage(lat, lon, KEY, zoom_level, size[0], size[1])
    if image_url == 'Error': return None
    try:
        response = requests.get(image_url, headers={"User-Agent": "SportsFacilityFinder/1.0"})
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return np.array(img.convert('RGB'))
    except Exception as e:
        print(f"Error fetching satellite image: {e}")
        return None

def fetch_zip_bbox_via_photon(zip_code, state_code):
    url = "https://photon.komoot.io/api/"
    params = {"q": f"{zip_code}, {state_code}, United States", "limit": 1}
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json()
        if data.get("features"):
            extent = data["features"][0].get("properties", {}).get("extent")
            if extent and len(extent) == 4:
                # Use min/max to guarantee (South, West, North, East) order for Overpass
                south_lat = min(extent[1], extent[3])
                north_lat = max(extent[1], extent[3])
                west_lon = min(extent[0], extent[2])
                east_lon = max(extent[0], extent[2])
                
                return (south_lat, west_lon, north_lat, east_lon)
    except Exception as e:
        print(f"Photon bounding geocoding failed: {e}")
    return None

# ---------------------------------------------------------
# DATA SOURCE FETCHING FUNCTIONS
# ---------------------------------------------------------
#Print the json on the screen
def get_fields_from_overpass(bbox, sport_name):
    endpoint = "https://overpass-api.de/api/interpreter"
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:30];
    (
      nwr["leisure"="pitch"]["sport"="{sport_name}"]({min_lat},{min_lon},{max_lat},{max_lon});
      nwr["leisure"="stadium"]["sport"="{sport_name}"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    
    # 1. Define the User-Agent header
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    
    try:
        # 2. Add headers=headers to the requests.post call
        response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Overpass API evaluation failed for {sport_name}: {e}")
        return {"elements": []}
    
def get_fields_from_google(zip_code, sport_name, api_key):
    """Fetches locations using Google Places Text Search API."""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    query = f"{sport_name} field in {zip_code}"
    params = {"query": query, "key": api_key}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Google Places API failed for {sport_name}: {e}")
        return {"results": []}

# ---------------------------------------------------------
# PARSING FUNCTIONS
#Look at json and see the mapping and verify
# ---------------------------------------------------------
def parse_osm_elements(osm_data, search_sport_type, city, state, postal_code):
    fields = []
    for el in osm_data.get("elements", []):
        lat = el.get("center", {}).get("lat", el.get("lat"))
        lon = el.get("center", {}).get("lon", el.get("lon"))
        if not lat or not lon: continue
            
        tags = el.get("tags", {})
        el_type = el.get("type", "node")
        el_id = el.get("id", 0)
        
        # Standard OSM URI Format (Native ID)
        osm_native_id = f"{el_type}/{el_id}" 
        
        raw_name = tags.get("name", tags.get("description", f"OSM {tags.get('leisure', 'Facility')} Location"))
        street = tags.get("addr:street", "")
        house_num = tags.get("addr:housenumber", "")
        
        field = {
            'field_name': raw_name,
            'formatted_address': f"{house_num} {street}, {city}, {state} {postal_code}".strip(", "),
            'postal_code': postal_code,
            'street': f"{house_num} {street}".strip(),
            'city': city,
            'state': state,
            'original_gps_location': {'latitude': float(lat), 'longitude': float(lon)},
            'gplace_id': osm_native_id,
            'search_sport_type': search_sport_type,
            'gearth_link': f"https://earth.google.com/web/@{lat},{lon},4.1972381a,15000d",
            'modified_fields': []
        }
        fields.append(field)
    return fields

def parse_google_elements(google_data, search_sport_type, city, state, postal_code):
    fields = []
    for res in google_data.get("results", []):
        lat = res["geometry"]["location"]["lat"]
        lon = res["geometry"]["location"]["lng"]
        
        # Native Google Place ID
        place_id = res.get("place_id", f"unknown_google_{lat}_{lon}")
        
        field = {
            'field_name': res.get("name", f"Google Facility Location"),
            'formatted_address': res.get("formatted_address", ""),
            'postal_code': postal_code,
            'street': res.get("formatted_address", "").split(",")[0], # Rough extraction
            'city': city,
            'state': state,
            'original_gps_location': {'latitude': float(lat), 'longitude': float(lon)},
            'gplace_id': place_id, 
            'search_sport_type': search_sport_type,
            'gearth_link': f"https://earth.google.com/web/@{lat},{lon},4.1972381a,15000d",
            'modified_fields': []
        }
        fields.append(field)
    return fields

# ---------------------------------------------------------
# DATABASE DEDUPLICATION & METRIC FUNCTIONS (FIXED CHANGELOG 1)
# ---------------------------------------------------------
def delete_duplicates():
    if 'database' not in config: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        # FIXED: Pure SQL Haversine Distance computation replacement for PostGIS ST_ClusterDBSCAN.
        # This groups rows that share the same sport_type and fall within ~50 meters (0.05 km) of each other.
        delete_query = """
        WITH parsed_gps AS (
            SELECT 
                field_search_id,
                search_sport_type,
                SPLIT_PART(gps_location, ',', 1)::float AS lat,
                SPLIT_PART(gps_location, ',', 2)::float AS lon
            FROM public.new_google_earth
            -- Add this WHERE clause to ensure only valid 'lat,lon' coordinates are parsed
            -- Notice the double backslashes before the question marks
            WHERE gps_location ~ '^[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+$'
        ),
        spatial_ranking AS (
            SELECT 
                p1.field_search_id,
                ROW_NUMBER() OVER (
                    PARTITION BY p1.search_sport_type, MIN(p2.field_search_id)
                    ORDER BY p1.field_search_id
                ) AS rn
            FROM parsed_gps p1
            LEFT JOIN parsed_gps p2 ON p1.search_sport_type = p2.search_sport_type
                AND (
                    6371 * acos(
                        -- Clamp the floating point math perfectly between -1.0 and 1.0
                        LEAST(1.0, GREATEST(-1.0, 
                            cos(radians(p1.lat)) * cos(radians(p2.lat)) * cos(radians(p2.lon) - radians(p1.lon)) + 
                            sin(radians(p1.lat)) * sin(radians(p2.lat))
                        ))
                    )
                ) <= 0.05
            GROUP BY p1.field_search_id, p1.search_sport_type
        ),
        duplicate_ids AS (
            SELECT field_search_id FROM spatial_ranking WHERE rn > 1
        ),
        deleted_objects AS (
            DELETE FROM public.nge_object 
            WHERE field_search_id IN (SELECT field_search_id FROM duplicate_ids)
            RETURNING field_search_id
        )
        DELETE FROM public.new_google_earth 
        WHERE field_search_id IN (SELECT field_search_id FROM duplicate_ids);
        """
        
        cur.execute(delete_query)
        conn.commit()
        print("Cleaned up overlapping spatial duplicate rows via Haversine evaluation.")
    except Exception as error:
        print("Error during database duplicate cleaning:", error)
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)

# ---------------------------------------------------------
# IMPLEMENTING PIPELINE CODE BLOCKS (FIXED CHANGELOG 2)
# ---------------------------------------------------------
def log_processed(state_code, city, zip_code):
    print(f"Log Execution Metric: Processed batch for {city}, {state_code} {zip_code}")

def save_field_data(fields):
    if not fields: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        for f in fields:
            gps_str = f"{f['original_gps_location']['latitude']},{f['original_gps_location']['longitude']}"
            
            # Using 'postal_code' and 'state' to match the database schema
            cur.execute("""
                INSERT INTO public.new_google_earth 
                (field_name, formatted_address, postal_code, street, city, state, gps_location, gearth_link, search_sport_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, (
                f['field_name'], 
                f['formatted_address'], 
                f['postal_code'], 
                f['street'], 
                f['city'], 
                f['state'], 
                gps_str, 
                f['gearth_link'], 
                f['search_sport_type']
            ))
        conn.commit()
        print(f"Saved {len(fields)} records to the database.")
    except Exception as e:
        print("Database error inside save_field_data:", e)
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)
        
def object_detection_based_modification_by_class(field_id, img_array, model, gps_loc, target_classes, zoom_level=18):
    modified_records = []
    try:
        results = model.predict(img_array, verbose=False)
        for res in results:
            for box in res.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                if class_id in target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    lat_offset = (center_y - (img_array.shape[0] / 2)) * -0.000005
                    lon_offset = (center_x - (img_array.shape[1] / 2)) * 0.000007
                    
                    adjusted_lat = gps_loc['latitude'] + lat_offset
                    adjusted_lon = gps_loc['longitude'] + lon_offset
                    
                    # Convert the integer class_id to the sport name string
                    sport_label = display_names.get(class_id, f"Unknown_{class_id}")
                    
                    modified_records.append({
                        'field_search_id': field_id,
                        'sport_name': sport_label, 
                        'confidence_score': confidence,
                        'adjusted_gps': f"{adjusted_lat},{adjusted_lon}"
                    })
    except Exception as e:
        print(f"Error executing frame spatial modification offsets: {e}")
    return modified_records

def save_object_data(nge_objects):
    if not nge_objects: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        for obj in nge_objects:
            # We must convert confidence_score to a string because your database column 'detect_confidence' is character varying
            conf_str = str(obj['confidence_score'])
            
            cur.execute("""
                INSERT INTO public.nge_object (field_search_id, detected_sport, detect_confidence, adjusted_gps_location)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, (obj['field_search_id'], obj['sport_name'], conf_str, obj['adjusted_gps']))
        conn.commit()
        print(f"Recorded {len(nge_objects)} spatial map target markers into nge_object table.")
    except Exception as e:
        print(f"Error writing to nge_object table: {e}")
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)


# ---------------------------------------------------------
# MAIN EXECUTION ENTRYPOINT
# ---------------------------------------------------------
if __name__ == "__main__":
    state_code = input("Enter state code : ").strip().upper()
    city_to_add = input("Enter city to add ): ").strip().upper()
    single_zip = input("Enter zipcode to check : ").strip()
    
    print("\nSelect Data Source:")
    print("1. OpenStreetMap (Overpass)")
    print("2. Google Places API")
    print("3. Both")
    source_choice = input("Enter choice (1/2/3): ").strip()
    
    city = city_to_add
    zip_code = single_zip
                
    bbox = fetch_zip_bbox_via_photon(zip_code, state_code)
    if not bbox and source_choice in ['1', '3']:
        print(f"Could not compute boundaries for ZIP {zip_code}. Skipping OSM fetch.")
        
    all_fields = []
    
    for sport_id, sport_name in SPORT_TAGS.items():
        
        # Fetch from OSM
        if source_choice in ['1', '3'] and bbox:
            print(f"Querying OSM for: '{sport_name}' in {zip_code}")
            osm_json = get_fields_from_overpass(bbox, sport_name)
            osm_fields = parse_osm_elements(osm_json, sport_id, city, state_code, zip_code)
            all_fields.extend(osm_fields)
        
        # Fetch from Google Maps
        if source_choice in ['2', '3']:
            print(f"Querying Google Places for: '{sport_name}' in {zip_code}")
            google_json = get_fields_from_google(zip_code, sport_name, KEY)
            google_fields = parse_google_elements(google_json, sport_id, city, state_code, zip_code)
            all_fields.extend(google_fields)

    if all_fields:
        save_field_data(all_fields)
        try:
            log_processed(state_code, city, zip_code)
        except NameError:
            pass
                
   # delete_duplicates()
    
    # Run spatial recentering via YOLO
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
            modified_locations = object_detection_based_modification_by_class( #Check this after the jsons.
                row['field_id'], img_array, obd_model, gps_loc, all_target_class_ids, zoom_level=18
            )
            nge_object.extend(modified_locations)
    
    save_object_data(nge_object)
    print("Data input process completed successfully.")
                
   