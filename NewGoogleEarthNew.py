import cv2
import requests
import psycopg2
import pandas as pd
from configparser import ConfigParser
import os
import json
import numpy as np
import pyproj
import re
from pyproj import Transformer
import collections.abc
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from ConnectionPool import pool

# Set up the config file
KEY = 'AIzaSyC5cT2KgRuUuz51GQ71DvY8gB_VN8O8EtE' # Note: Be careful exposing your API keys publicly!
file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
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

def get_fields_from_google(zip_code, sport_name, api_key):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    query = f"{sport_name} field in {zip_code}"
    params = {"query": query, "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"results": []}

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
    import time  # Imported locally to avoid external dependency issues
    
    # List of reliable public Overpass API mirrors
    endpoints = [
        "https://overpass-api.de/api/interpreter",          # Main Server (often overloaded)
        "https://overpass.kumi.systems/api/interpreter",    # Kumi Systems (highly reliable backup)
        "https://overpass.openstreetmap.ru/api/interpreter",# Russian Mirror
        "https://overpass.nchc.org.tw/api/interpreter"      # Taiwanese Mirror
    ]
    
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:30];
    (
      nwr["leisure"="pitch"]["sport"="{sport_name}"]({min_lat},{min_lon},{max_lat},{max_lon});
      nwr["leisure"="stadium"]["sport"="{sport_name}"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    
    for endpoint in endpoints:
        # Try each server up to 2 times before falling back to the next mirror
        for attempt in range(2):
            try:
                # Lowering timeout to 15s so it fails over faster instead of hanging
                response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=15)
                
                # Handle 429 (Rate Limited) with exponential backoff
                if response.status_code == 429:
                    print(f"⚠️ Mirror {endpoint} returned 429 (Rate Limited). Retrying in 3s...")
                    time.sleep(3)
                    continue
                    
                # Handle 504 or other server issues by breaking out to try the next mirror
                if response.status_code >= 500:
                    print(f"⚠️ Mirror {endpoint} returned {response.status_code} (Server Error). Swapping mirrors...")
                    break  
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Connection failed for mirror {endpoint} (Attempt {attempt + 1}/2): {e}")
                time.sleep(1)
                
    print(f"All Overpass API mirrors failed or timed out for sport '{sport_name}'.")
    return {"elements": []}
# ---------------------------------------------------------
# PARSING FUNCTIONS
#Look at json and see the mapping and verify
# ---------------------------------------------------------
def extract_clean_address(tags):
    """
    Extracts and builds a clean string address from OSM tags,
    mirroring the structured fallback logic in the Qlever pipeline.
    """
    housenumber = tags.get('addr:housenumber', '').strip()
    street = tags.get('addr:street', '').strip()
    city = tags.get('addr:city', '').strip()
    postcode = tags.get('addr:postcode', '').strip()
    state = tags.get('addr:state', '').strip()
    
    # Check if a combined address tag already exists
    full_addr = tags.get('addr:full', '')
    if full_addr:
        return full_addr.strip()
        
    # Construct address dynamically based on available items
    parts = []
    if housenumber and street:
        parts.append(f"{housenumber} {street}")
    elif street:
        parts.append(street)
        
    if city:
        parts.append(city)
    if state:
        parts.append(state)
    if postcode:
        parts.append(postcode)
        
    return ", ".join(parts) if parts else None
def get_base_name(name):
    """""
    Strips trailing parenthesis sports lists for raw facility name
    """
    if not name:
        return ""
    return re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
import re  # Ensure this is imported at the very top of your script

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two GPS coordinates in meters.
    """
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371000  # Returns distance in meters


def group_incoming_fields(fields):
    """
    Groups incoming fields in memory based on:
    - Distance less than 120 meters apart AND same facility name.
    - OR identical street address AND same facility name.
    - OR exact match of facility name AND full formatted address.
    """
    grouped = []
    for f in fields:
        lat = f['original_gps_location']['latitude']
        lon = f['original_gps_location']['longitude']
        street = (f['street'] or '').strip().lower()
        addr = (f['formatted_address'] or '').strip().lower()
        base_name = get_base_name(f['field_name']).lower()
        
        match = None
        for g in grouped:
            g_lat = g['original_gps_location']['latitude']
            g_lon = g['original_gps_location']['longitude']
            g_street = (g['street'] or '').strip().lower()
            g_addr = (g['formatted_address'] or '').strip().lower()
            g_base_name = get_base_name(g['field_name']).lower()
            
            # Calculate physical distance
            dist = haversine(lat, lon, g_lat, g_lon)
            
            # Match based on spatial proximity OR exact street address OR exact full address matching
            same_facility = (
                (dist < 120.0 and base_name == g_base_name and base_name != "") or
                (street == g_street and base_name == g_base_name and street != "" and base_name != "") or
                (addr == g_addr and base_name == g_base_name and addr != "" and base_name != "")
            )
            
            if same_facility:
                match = g
                break
                
        sport_name = SPORT_TAGS.get(f['search_sport_type'], "Field").title()
        if match:
            if 'sports' not in match:
                match['sports'] = {SPORT_TAGS.get(match['search_sport_type'], "Field").title()}
            match['sports'].add(sport_name)
            match['coords_list'].append((lat, lon))
            # match['number_of_fields'] += 1
        else:
            f_copy = f.copy()
            f_copy['sports'] = {sport_name}
            f_copy['coords_list'] = [(lat, lon)]
            # f_copy['number_of_fields'] = 1
            grouped.append(f_copy)
            
    # Calculate the centralized centroid coordinate for each grouped facility
    for g in grouped:
        lats = [c[0] for c in g['coords_list']]
        lons = [c[1] for c in g['coords_list']]
        g['original_gps_location']['latitude'] = sum(lats) / len(lats)
        g['original_gps_location']['longitude'] = sum(lons) / len(lons)
        
    return grouped
def construct_facility_name(tags, default_sport_name="Field"):
    """
    Implements the advanced name fallback architecture from qlever_v_6.
    Replaces generic 'osm pitch' strings with descriptive parent or feature labels.
    """
    # 1. Primary choice: Explicitly tagged object name
    if tags.get('name'):
        return tags.get('name').strip()
        
    # 2. Secondary choice: Parent container name (if it was pulled or nested)
    if tags.get('container_name'):
        container = tags.get('container_name').strip()
        sport = tags.get('sport', default_sport_name).replace('_', ' ').title()
        return f"{container} ({sport} Facility)"
        
    # 3. Tertiary choice: Build a descriptive asset label using auxiliary tags
    operator = tags.get('operator', '').strip()
    surface = tags.get('surface', '').strip().replace('_', ' ').title()
    sport = tags.get('sport', default_sport_name).replace('_', ' ').title()
    leisure = tags.get('leisure', '').strip().replace('_', ' ').title()
    
    name_parts = []
    if operator:
        name_parts.append(operator)
    elif surface:
        name_parts.append(f"OSM {surface}")
    else:
        name_parts.append("OSM")
        
    name_parts.append(sport)
    
    if leisure:
        name_parts.append(leisure)
    else:
        name_parts.append("Facility")
        
    return " ".join(name_parts)
# ---------------------------------------------------------
# PARSING FUNCTIONS (WITH ADVANCED QLEVER NAME & ADDRESS HANDLING)
# ---------------------------------------------------------
# ---------------------------------------------------------
# PARSING & REVERSE GEOCODING ENRICHMENT FUNCTIONS
# ---------------------------------------------------------
# ---------------------------------------------------------
# PARSING & REVERSE GEOCODING ENRICHMENT FUNCTIONS
# ---------------------------------------------------------
def get_nearest_street_via_osm(lat, lon, radius=150):
    """
    Queries Overpass to find the closest named public road
    within a specified radius of the coordinates.
    """
    endpoint = "https://overpass-api.de/api/interpreter"
    # Filters for real drivable roads, avoiding unnamed paths/sidewalks
    query = f"""
    [out:json][timeout:15];
    (
      way["highway"~"residential|tertiary|secondary|primary|unclassified|service"]["name"](around:{radius},{lat},{lon});
    );
    out tags;
    """
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    try:
        response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            if elements:
                # Returns the first/closest matching road name found
                return elements[0].get("tags", {}).get("name", "").strip()
    except Exception as e:
        print(f"Failed to find nearest street via Overpass spatial scan: {e}")
    return None
def get_nearest_address_via_osm(lat, lon, radius=200):
    """
    Queries Overpass to find the closest object that has an explicit 
    house number and street name tagged near the coordinates.
    """
    endpoint = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:15];
    nwr["addr:housenumber"]["addr:street"](around:{radius},{lat},{lon});
    out center;
    """
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    try:
        response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            if elements:
                # Extract the tags from the closest matching addressed feature
                tags = elements[0].get("tags", {})
                return tags.get("addr:housenumber", "").strip(), tags.get("addr:street", "").strip()
    except Exception as e:
        print(f"Failed to find nearest address via Overpass spatial scan: {e}")
    return None, None

def enrich_location_via_photon(lat, lon, default_city, default_state, default_zip, sport_label):
    """
    Queries Photon reverse geocoding to resolve the real-world 
    campus facility name (e.g. Park or School) and full street address.
    """
    url = "https://photon.komoot.io/reverse"
    params = {"lat": lat, "lon": lon, "lang": "en"}
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    
    # Defaults in case the external lookup falls back
    name = f"OSM {sport_label} Facility"
    address = f"{default_city}, {default_state} {default_zip}"
    street = ""
    city = default_city
    state = default_state
    postcode = default_zip
    
    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            features = data.get("features", [])
            if features:
                props = features[0].get("properties", {})
                
                photon_name = props.get("name", "").strip()
                photon_street = props.get("street", "").strip()
                photon_house = props.get("housenumber", "").strip()
                photon_city = props.get("city", props.get("town", default_city)).strip()
                photon_state = props.get("state", default_state).strip()
                photon_postcode = props.get("postcode", default_zip).strip()
                
                if photon_name and photon_name != photon_street:
                    name = f"{photon_name} ({sport_label} Field)"
                elif photon_street:
                    name = f"{photon_street} {sport_label} Field"
                if not photon_street:
                    print(f"Address details missing for '{photon_name}' at {lat},{lon}. Scanning for closest address...")
                    nearest_house, nearest_road = get_nearest_address_via_osm(lat, lon)
                    if nearest_road:
                        photon_street = nearest_road
                        if nearest_house:
                            photon_house = nearest_house
                    
                if photon_house and photon_street:
                    street_line = f"{photon_house} {photon_street}"
                else:
                    street_line = photon_street if photon_street else photon_name
                    
                # FIX: Fall back to the facility name if a formal street name is missing
                street = street_line if street_line else "Unnamed Road"
                    
                if street_line:
                    address = f"{street_line}, {photon_city}, {photon_state} {photon_postcode}"
                else:
                    address = f"{photon_city}, {photon_state} {photon_postcode}"
                    
                city = photon_city
                state = photon_state
                postcode = photon_postcode
    except Exception as e:
        print(f"Photon reverse enrichment failed for ({lat}, {lon}): {e}")
        
    return name, address, street, city, state, postcode
def get_osm_parent_name(lat, lon, radius=80):
    """
    Queries Overpass to find a named parent facility (like a park, school, or campus)
    surrounding or immediately adjacent to the unnamed pitch coordinates.
    """
    endpoint = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:15];
    (
      nwr["leisure"="park"]["name"](around:{radius},{lat},{lon});
      nwr["amenity"~"school|university|college"]["name"](around:{radius},{lat},{lon});
      nwr["leisure"="sports_centre"]["name"](around:{radius},{lat},{lon});
      nwr["landuse"~"recreation_ground|education"]["name"](around:{radius},{lat},{lon});
    );
    out tags;
    """
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    try:
        response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            if elements:
                # Return the name of the closest or first matching parent feature found
                return elements[0].get("tags", {}).get("name", "").strip()
    except Exception as e:
        print(f"Failed to fetch OSM parent container name: {e}")
    return None

def parse_osm_elements(osm_data, search_sport_type, city, state, postal_code):
    """
    Parses OpenStreetMap elements and automatically enriches empty records
    using live coordinate-based parent lookups and reverse geocoding.
    """
    fields = []
    sport_str = SPORT_TAGS.get(search_sport_type, "Field").title()

    for el in osm_data.get("elements", []):
        lat = el.get("center", {}).get("lat", el.get("lat"))
        lon = el.get("center", {}).get("lon", el.get("lon"))
        if not lat or not lon: 
            continue
            
        tags = el.get("tags", {})
        osm_native_id = f"{el.get('type', 'node')}/{el.get('id', 0)}" 
        
        has_native_name = tags.get("name") is not None
        has_native_address = tags.get("addr:street") is not None
        
        # --- 1. RESOLVE FACILITY NAME ---
        if has_native_name:
            raw_name = tags.get("name").strip()
        else:
            # Look up surrounding named Park/School polygons in OSM
            print(f"Pitch missing name at {lat},{lon}. Searching OSM parent boundaries...")
            parent_name = get_osm_parent_name(lat, lon)
            if parent_name:
                raw_name = f"{parent_name} ({sport_str} Field)"
            else:
                raw_name = None  # Let the address-based naming engine handle it if nothing is found
        
        # --- 2. RESOLVE ADDRESS & BACKUP NAME ---
        if has_native_address:
            street_field = tags.get('addr:street', '').strip()
            housenumber = tags.get('addr:housenumber', '').strip()
            if housenumber:
                street_field = f"{housenumber} {street_field}"
            city_field = tags.get('addr:city', city).strip()
            state_field = tags.get('addr:state', state).strip()
            postcode_field = tags.get('addr:postcode', postal_code).strip()
            formatted_address = tags.get('addr:full', f"{street_field}, {city_field}, {state_field} {postcode_field}").strip()
            
            # If parent lookup failed, construct name using the verified street
            if not raw_name:
                raw_name = f"{street_field} {sport_str} Field"
        else:
            # Fallback to Photon reverse-geocoding if no native address tags exist
            photon_name, formatted_address, street_field, city_field, state_field, postcode_field = enrich_location_via_photon(
                lat, lon, city, state, postal_code, sport_str
            )
            # Use Photon's guessed name only if our Overpass parent search didn't locate a better boundary
            if not raw_name:
                raw_name = photon_name

        field = {
            'field_name': raw_name,
            'formatted_address': formatted_address,
            'postal_code': postcode_field,
            'street': street_field,
            'city': city_field,
            'state': state_field,
            'original_gps_location': {'latitude': float(lat), 'longitude': float(lon)},
            'gplace_id': osm_native_id,
            'search_sport_type': search_sport_type,
            'gearth_link': f"https://earth.google.com/web/@{lat},{lon},4.1972381a,15000d",
            'modified_fields': []
        }
        fields.append(field)
    return fields

def parse_google_elements(google_data, search_sport_type, city, state, postal_code):
    """
    Parses Google Places API elements using uniform keys matching the database format.
    """
    fields = []
    for res in google_data.get("results", []):
        lat = res["geometry"]["location"]["lat"]
        lon = res["geometry"]["location"]["lng"]
        place_id = res.get("place_id", f"unknown_google_{lat}_{lon}")
        
        formatted_address = res.get("formatted_address", "")
        street_field = formatted_address.split(",")[0].strip() if formatted_address else ""
        
        field = {
            'field_name': res.get("name", "Google Facility Location"),
            'formatted_address': formatted_address if formatted_address else f"{city}, {state} {postal_code}",
            'postal_code': postal_code,
            'street': street_field,
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
def delete_duplicates(target_zip):
    """
    Identifies overlapping cross-source fields inside the same park boundary/zip code 
    using a 80-meter spatial radius and removes the duplicates dynamically.
    """
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        # Validated pure spatial join filtering matching pairs within ~0.08 kilometers
        delete_query = """
        WITH parsed_gps AS (
            SELECT 
                field_search_id,
                search_sport_type,
                SPLIT_PART(gps_location, ',', 1)::float AS lat,
                SPLIT_PART(gps_location, ',', 2)::float AS lon
            FROM public.new_google_earth
            WHERE postal_code = %s
              AND gps_location ~ '^[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+$'
        ),
        duplicate_ids AS (
            SELECT DISTINCT p2.field_search_id
            FROM parsed_gps p1
            JOIN parsed_gps p2 ON p1.search_sport_type = p2.search_sport_type
                AND p1.field_search_id < p2.field_search_id
                AND (
                    6371 * acos(
                        LEAST(1.0, GREATEST(-1.0, 
                            cos(radians(p1.lat)) * cos(radians(p2.lat)) * cos(radians(p2.lon) - radians(p1.lon)) + 
                            sin(radians(p1.lat)) * sin(radians(p2.lat))
                        ))
                    )
                ) <= 0.08
        ),
        deleted_objects AS (
            DELETE FROM public.nge_object 
            WHERE field_search_id IN (SELECT field_search_id FROM duplicate_ids)
        )
        DELETE FROM public.new_google_earth 
        WHERE field_search_id IN (SELECT field_search_id FROM duplicate_ids);
        """
        
        cur.execute(delete_query, (target_zip,))
        conn.commit()
        print(f"🧹 Successfully deduplicated cross-source park variants for ZIP: {target_zip}")
    except Exception as error:
        print("Error executing database duplicate cleaning sequence:", error)
        if conn: conn.rollback()
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)

# ---------------------------------------------------------
# IMPLEMENTING PIPELINE CODE BLOCKS (FIXED)
# ---------------------------------------------------------
# ---------------------------------------------------------
# IMPLEMENTING PIPELINE CODE BLOCKS (FIXED)
# ---------------------------------------------------------
def log_processed(state_code, city, zip_code):
    print(f"Log Execution Metric: Processed batch for {city}, {state_code} {zip_code}")


def save_field_data(fields):
    if not fields: return
    
    # 1. Group incoming API elements in memory first
    grouped_fields = group_incoming_fields(fields)
    target_zip = fields[0]['postal_code']
    
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        # Fetch existing ZIP records once to matching against in memory
        cur.execute("""
            SELECT field_search_id, field_name, gps_location, street, formatted_address 
            FROM public.new_google_earth 
            WHERE postal_code = %s;
        """, (target_zip,))
        
        db_records = []
        for row in cur.fetchall():
            db_records.append({
                'field_search_id': row[0],
                'field_name': row[1],
                'gps_location': row[2],
                'street': row[3],
                'formatted_address': row[4],    # Key map index 4
                # 'number_of_fields': row[5] or 1 # Shifted index to 5
            })
            
        inserted_count = 0
        updated_count = 0
        
        for f in grouped_fields:
            lat = f['original_gps_location']['latitude']
            lon = f['original_gps_location']['longitude']
            gps_str = f"{lat},{lon}"
            street = (f['street'] or '').strip().lower()
            addr = (f['formatted_address'] or '').strip().lower()
            base_name = get_base_name(f['field_name']).lower()
            
            db_match = None
            for db in db_records:
                try:
                    db_lat, db_lon = map(float, db['gps_location'].split(','))
                except ValueError:
                    continue
                    
                db_street = (db['street'] or '').strip().lower()
                db_addr = (db['formatted_address'] or '').strip().lower()
                db_base_name = get_base_name(db['field_name']).lower()
                
                dist = haversine(lat, lon, db_lat, db_lon)
                
                # Added exact database address match checks
                same_facility = (
                    (dist < 200.0 and base_name == db_base_name and base_name != "") or
                    (street == db_street and base_name == db_base_name and street != "" and base_name != "") or
                    (addr == db_addr and base_name == db_base_name and addr != "" and base_name != "")
                )
                
                if same_facility:
                    db_match = db
                    break
            
            # --- RESTORED SQL EXECUTION SEQUENCE ---
            existing_sports = set()
            if db_match:
                match_paren = re.search(r'\(([^)]+)\)\s*$', db_match['field_name'])
                if match_paren:
                    parts = [p.strip().title() for p in match_paren.group(1).split(',')]
                    existing_sports.update(parts)
            
            combined_sports = existing_sports.union(f['sports'])
            sports_list_str = ", ".join(sorted(list(combined_sports)))
            final_field_name = f"{get_base_name(f['field_name'])} ({sports_list_str})"
            # final_num_fields = f['number_of_fields']
            
            if db_match:
                cur.execute("""
                    UPDATE public.new_google_earth SET
                        field_name        = %s,
                        formatted_address = %s,
                        postal_code       = %s,
                        street            = %s,
                        city              = %s,
                        state             = %s,
                        gps_location      = %s,
                        gearth_link       = %s,
                    WHERE field_search_id = %s;
                """, (
                    final_field_name, 
                    f['formatted_address'], 
                    f['postal_code'], 
                    f['street'], 
                    f['city'], 
                    f['state'], 
                    gps_str, 
                    f['gearth_link'], 
                    # final_num_fields,
                    db_match['field_search_id']
                ))
                updated_count += 1
            else:
                cur.execute("""
                    INSERT INTO public.new_google_earth 
                    (field_name, formatted_address, postal_code, street, city, state, gps_location, gearth_link, search_sport_type, gplace_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    final_field_name, 
                    f['formatted_address'], 
                    f['postal_code'], 
                    f['street'], 
                    f['city'], 
                    f['state'], 
                    gps_str, 
                    f['gearth_link'], 
                    f['search_sport_type'], 
                    f['gplace_id']
                    # final_num_fields
                ))
                inserted_count += 1
                
        conn.commit()
        print(f"Cleanly grouped and processed records. (Inserted unique: {inserted_count}, Merged/Updated: {updated_count})")
    except Exception as e:
        print("Database error inside save_field_data:", e)
        if conn:
            conn.rollback()
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


# RESTORED: This function was missing from your script
def save_object_data(nge_objects):
    if not nge_objects: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        # Pull distinct target IDs processed in this batch
        distinct_field_ids = list(set(obj['field_search_id'] for obj in nge_objects))
        
        # 1. Clean out old model records for these fields in nge_object first
        cur.execute("""
            DELETE FROM public.nge_object 
            WHERE field_search_id = ANY(%s);
        """, (distinct_field_ids,))
        
        for obj in nge_objects:
            # Build a helpful description showing the detection confidence
            conf_percent = obj['confidence_score'] * 100
            desc = f"Detected via YOLO with {conf_percent:.1f}% confidence"
            
            # 2. Insert the metadata into nge_object using its correct schema
            cur.execute("""
                INSERT INTO public.nge_object (field_search_id, sport_name, description)
                VALUES (%s, %s, %s);
            """, (obj['field_search_id'], obj['sport_name'], desc))
            
            # 3. Update the high-precision adjusted coordinates in new_google_earth
            cur.execute("""
                UPDATE public.new_google_earth 
                SET gps_location = %s 
                WHERE field_search_id = %s;
            """, (obj['adjusted_gps'], obj['field_search_id']))
            final_num_fields
        conn.commit()
        print(f"🔄 Successfully updated gps_locations in new_google_earth and logged {len(nge_objects)} details in nge_object.")
    except Exception as e:
        print(f"Error writing to database tables: {e}")
        if conn: conn.rollback()
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)
# ---------------------------------------------------------
# MAIN EXECUTION ENTRYPOINT
# ---------------------------------------------------------
# ---------------------------------------------------------
# MAIN EXECUTION ENTRYPOINT (FIXED DATABASE FILTERING)
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
        
        "delete_duplicates(zip_code)"
        
        try:
            log_processed(state_code, city, zip_code)
        except NameError:
            pass
    else:
        print(f"No new fields found via live APIs for ZIP code {zip_code}.")
                
    # Run spatial recentering via YOLO
    # FIXED: Direct database pull targeting only the requested ZIP code
    print(f"Retrieving database records strictly for ZIP code: {zip_code}")
    db_fields = []
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        cur.execute("""
            SELECT field_search_id, field_name, gps_location 
            FROM public.new_google_earth 
            WHERE postal_code = %s;
        """, (zip_code,))
        columns = [desc[0] for desc in cur.description]
        db_fields = [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching filtered fields from database: {e}")
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)

    print(f"Rows matching criteria for YOLO processing: {len(db_fields)}")
    
    nge_object = []
    all_target_class_ids = [4, 5, 9, 10, 14, 16]
    
    for row in db_fields:
        field_id = row['field_search_id']
        print(f"Processing field: {row['field_name']} at {row['gps_location']}")
        
        lat, lon = row['gps_location'].split(",")
        gps_loc = {'latitude': float(lat), 'longitude': float(lon)}
        img_array = get_satellite_image_array(gps_loc)
        
        if img_array is not None:
            modified_locations = object_detection_based_modification_by_class(
                field_id, img_array, obd_model, gps_loc, all_target_class_ids, zoom_level=18
            )
            nge_object.extend(modified_locations)
    
    if nge_object:
        save_object_data(nge_object)
        
    print("Data input process completed successfully.")