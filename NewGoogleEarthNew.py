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

import io
import time
import requests
import numpy as np
from PIL import Image, UnidentifiedImageError

def get_satellite_image_array(gps_loc, max_retries=3):
    """
    Fetches satellite imagery with coordinate validation, retry logic, 
    header validation, and safe PIL decoding to prevent pipeline dropouts.
    """
    # 1. Sanitize and Validate GPS Input
    try:
        lat = float(gps_loc['latitude'])
        lon = float(gps_loc['longitude'])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            print(f"Out-of-bounds GPS coordinates: ({lat}, {lon})")
            return None
    except (KeyError, TypeError, ValueError) as err:
        print(f"Invalid GPS dictionary/format {gps_loc}: {err}")
        return None

    # Replace with your actual API endpoint & key
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=18&size=640x640&maptype=satellite&key={KEY}"
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}

    # 2. Retry Loop with Exponential Delay
    for attempt in range(1, max_retries + 1):
        try:
            # Explicit timeout prevents hanging requests
            response = requests.get(url, headers=headers, timeout=15)

            # Handle Rate Limiting (HTTP 429 or 503)
            if response.status_code in (429, 503):
                delay = attempt * 2
                print(f"Server busy ({response.status_code}). Retrying in {delay}s (Attempt {attempt}/{max_retries})...")
                time.sleep(delay)
                continue

            response.raise_for_status()

            # 3. Verify Response Content Type Before Decoding
            content_type = response.headers.get("Content-Type", "").lower()
            if "image" not in content_type:
                print(f"Received non-image payload ({content_type}): {response.text[:120]}")
                return None

            # 4. Safe Image Decoding
            image_bytes = io.BytesIO(response.content)
            img = Image.open(image_bytes).convert("RGB")
            return np.array(img)

        except requests.exceptions.Timeout:
            print(f"Request timed out for ({lat}, {lon}) [Attempt {attempt}/{max_retries}]")
            time.sleep(1)

        except requests.exceptions.RequestException as req_err:
            print(f"Network error for ({lat}, {lon}): {req_err}")
            time.sleep(1)

        except (UnidentifiedImageError, OSError) as img_err:
            print(f"Failed to decode image payload into NumPy array: {img_err}")
            return None

    print(f"Failed to fetch image for ({lat}, {lon}) after {max_retries} attempts.")
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
    import time
    
    # Active public Overpass API mirrors
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://z.overpass-api.de/api/interpreter"
    ]
    
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:45];
    (
      nwr["leisure"="pitch"]["sport"="{sport_name}"]({min_lat},{min_lon},{max_lat},{max_lon});
      nwr["leisure"="stadium"]["sport"="{sport_name}"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    
    for endpoint in endpoints:
        for attempt in range(2):
            try:
                response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=45)
                
                if response.status_code == 429:
                    print(f"Mirror {endpoint} returned 429 (Rate Limited). Retrying in 3s...")
                    time.sleep(3)
                    continue
                    
                if response.status_code >= 500:
                    print(f"Mirror {endpoint} returned {response.status_code} (Server Error). Swapping mirrors...")
                    break  
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"Connection failed for mirror {endpoint} (Attempt {attempt + 1}/2): {e}")
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

def backfill_missing_objects(zip_code=None):
    """
    Scans for any records in new_google_earth that lack an entry in nge_object.
    Attempts YOLO detection for missing rows, and generates default entries 
    if image fetching fails or YOLO yields 0 detections.
    """
    conn, cur = None, None
    missing_fields = []
    
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        query = """
            SELECT nge.field_search_id, nge.field_name, nge.gps_location, nge.search_sport_type, nge.gearth_link 
            FROM public.new_google_earth nge
            LEFT JOIN public.nge_object obj ON nge.field_search_id = obj.field_search_id
            WHERE obj.nge_object_id IS NULL
        """
        params = []
        if zip_code:
            query += " AND nge.postal_code = %s;"
            params.append(zip_code)
        else:
            query += ";"
            
        cur.execute(query, tuple(params))
        columns = [desc[0] for desc in cur.description]
        missing_fields = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        print(f"Self-Healing Check: Found {len(missing_fields)} records missing from nge_object.")
        
    except Exception as e:
        print(f"Error checking for missing object records: {e}")
        return
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)
        
    if not missing_fields:
        print("Safety check passed: All new_google_earth records have matching nge_object entries.")
        return

    nge_object_records = []
    all_target_class_ids = [4, 5, 9, 10, 14, 16]

    for row in missing_fields:
        field_id = row['field_search_id']
        gps_raw = str(row['gps_location']).strip() if row.get('gps_location') else ""
        sport_type_id = row.get('search_sport_type')
        default_sport_label = SPORT_TAGS.get(sport_type_id, "Sports Facility").title()

        if "," in gps_raw and "http" not in gps_raw:
            try:
                parts = gps_raw.split(",")
                lat, lon = float(parts[0]), float(parts[1])
                gps_loc = {'latitude': lat, 'longitude': lon}
                img_array = get_satellite_image_array(gps_loc)
                
                modified_locations = []
                if img_array is not None:
                    modified_locations = object_detection_based_modification_by_class(
                        field_id, img_array, obd_model, gps_loc, all_target_class_ids, zoom_level=18
                    )
                
                if modified_locations:
                    nge_object_records.extend(modified_locations)
                else:
                    nge_object_records.append({
                        'field_search_id': field_id,
                        'sport_name': default_sport_label,
                        'confidence_score': 0.0,
                        'adjusted_gps': gps_raw
                    })
            except Exception as ex:
                print(f"Error running model for missing field ID {field_id}: {ex}")
                nge_object_records.append({
                    'field_search_id': field_id,
                    'sport_name': default_sport_label,
                    'confidence_score': 0.0,
                    'adjusted_gps': gps_raw if gps_raw else "0.0,0.0"
                })
        else:
            nge_object_records.append({
                'field_search_id': field_id,
                'sport_name': default_sport_label,
                'confidence_score': 0.0,
                'adjusted_gps': "0.0,0.0"
            })

    if nge_object_records:
        save_object_data(nge_object_records)
        print(f"Resolved and saved {len(nge_object_records)} missing nge_object records.")

    nge_object_records = []
    all_target_class_ids = [4, 5, 9, 10, 14, 16]

    for row in missing_fields:
        field_id = row['field_search_id']
        gps_raw = str(row['gps_location']).strip() if row['gps_location'] else ""
        sport_type_id = row.get('search_sport_type')
        default_sport_label = SPORT_TAGS.get(sport_type_id, "Sports Facility").title()

        if "," in gps_raw and "http" not in gps_raw:
            try:
                parts = gps_raw.split(",")
                lat, lon = float(parts[0]), float(parts[1])
                gps_loc = {'latitude': lat, 'longitude': lon}
                img_array = get_satellite_image_array(gps_loc)
                
                modified_locations = []
                if img_array is not None:
                    modified_locations = object_detection_based_modification_by_class(
                        field_id, img_array, obd_model, gps_loc, all_target_class_ids, zoom_level=18
                    )
                
                if modified_locations:
                    nge_object_records.extend(modified_locations)
                else:
                    nge_object_records.append({
                        'field_search_id': field_id,
                        'sport_name': default_sport_label,
                        'confidence_score': 0.0,
                        'adjusted_gps': gps_raw
                    })
            except Exception as ex:
                print(f"Error running model for missing field ID {field_id}: {ex}")
                nge_object_records.append({
                    'field_search_id': field_id,
                    'sport_name': default_sport_label,
                    'confidence_score': 0.0,
                    'adjusted_gps': gps_raw if gps_raw else "0.0,0.0"
                })
        else:
            nge_object_records.append({
                'field_search_id': field_id,
                'sport_name': default_sport_label,
                'confidence_score': 0.0,
                'adjusted_gps': "0.0,0.0"
            })

    if nge_object_records:
        save_object_data(nge_object_records)
        print(f"Resolved and saved {len(nge_object_records)} missing nge_object records.")


def group_incoming_fields(fields):
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
            
            dist = haversine(lat, lon, g_lat, g_lon)
            
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
            match['number_of_fields'] += 1
        else:
            f_copy = f.copy()
            f_copy['sports'] = {sport_name}
            f_copy['coords_list'] = [(lat, lon)]
            f_copy['number_of_fields'] = 1
            grouped.append(f_copy)
            
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
    Guarantees search_sport_type, clean gps_location, and gearth_link generation.
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
            print(f"Pitch missing name at {lat},{lon}. Searching OSM parent boundaries...")
            parent_name = get_osm_parent_name(lat, lon)
            if parent_name:
                raw_name = f"{parent_name} ({sport_str} Field)"
            else:
                raw_name = None
        
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
            
            if not raw_name:
                raw_name = f"{street_field} {sport_str} Field"
        else:
            photon_name, formatted_address, street_field, city_field, state_field, postcode_field = enrich_location_via_photon(
                lat, lon, city, state, postal_code, sport_str
            )
            if not raw_name:
                raw_name = photon_name

        clean_gps = f"{float(lat)},{float(lon)}"
        clean_gearth_link = f"https://earth.google.com/web/@{clean_gps},4.1972381a,15000d"

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
            'gearth_link': clean_gearth_link,
            'modified_fields': []
        }
        fields.append(field)
    return fields

def parse_google_elements(google_data, search_sport_type, city, state, postal_code):
    """
    Parses Google Places API elements using uniform keys matching database schema.
    Guarantees search_sport_type, clean gps_location, and gearth_link generation.
    """
    fields = []
    for res in google_data.get("results", []):
        lat = res["geometry"]["location"]["lat"]
        lon = res["geometry"]["location"]["lng"]
        place_id = res.get("place_id", f"unknown_google_{lat}_{lon}")
        
        formatted_address = res.get("formatted_address", "")
        street_field = formatted_address.split(",")[0].strip() if formatted_address else ""
        
        clean_gps = f"{float(lat)},{float(lon)}"
        clean_gearth_link = f"https://earth.google.com/web/@{clean_gps},4.1972381a,15000d"
        
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
            'gearth_link': clean_gearth_link,
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
    """
    Saves or updates field records in new_google_earth.
    Ensures search_sport_type, gps_location, and gearth_link are explicitly persisted during UPDATEs and INSERTs.
    """
    if not fields: return
    
    grouped_fields = group_incoming_fields(fields)
    target_zip = fields[0]['postal_code']
    
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT field_search_id, field_name, gps_location, street, formatted_address, number_of_fields 
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
                'formatted_address': row[4],
                'number_of_fields': row[5] or 1
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
                
                same_facility = (
                    (dist < 200.0 and base_name == db_base_name and base_name != "") or
                    (street == db_street and base_name == db_base_name and street != "" and base_name != "") or
                    (addr == db_addr and base_name == db_base_name and addr != "" and base_name != "")
                )
                
                if same_facility:
                    db_match = db
                    break
            
            existing_sports = set()
            if db_match:
                match_paren = re.search(r'\(([^)]+)\)\s*$', db_match['field_name'])
                if match_paren:
                    parts = [p.strip().title() for p in match_paren.group(1).split(',')]
                    existing_sports.update(parts)
            
            combined_sports = existing_sports.union(f['sports'])
            sports_list_str = ", ".join(sorted(list(combined_sports)))
            final_field_name = f"{get_base_name(f['field_name'])} ({sports_list_str})"
            final_num_fields = f['number_of_fields']
            
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
                        number_of_fields  = %s,
                        search_sport_type = %s
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
                    final_num_fields,
                    f['search_sport_type'],
                    db_match['field_search_id']
                ))
                updated_count += 1
            else:
                cur.execute("""
                    INSERT INTO public.new_google_earth 
                    (field_name, formatted_address, postal_code, street, city, state, gps_location, gearth_link, search_sport_type, gplace_id, number_of_fields)
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
                    f['gplace_id'],
                    final_num_fields
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


def save_object_data(nge_objects):
    """
    Ensures every record generates an entry in nge_object with a valid
    nge_object_id and non-null sport_name, guaranteeing full coverage for SQL joins.
    """
    if not nge_objects: return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        distinct_field_ids = list(set(obj['field_search_id'] for obj in nge_objects))
        
        cur.execute("""
            DELETE FROM public.nge_object 
            WHERE field_search_id = ANY(%s);
        """, (distinct_field_ids,))
        
        for obj in nge_objects:
            conf = obj.get('confidence_score', 0.0)
            conf_percent = conf * 100
            
            sport_name = obj.get('sport_name') if obj.get('sport_name') else 'Sports Facility'
            
            if conf > 0:
                desc = f"Detected via YOLO with {conf_percent:.1f}% confidence"
            else:
                desc = "Baseline record generated for facility"
            
            cur.execute("""
                INSERT INTO public.nge_object (field_search_id, sport_name, description)
                VALUES (%s, %s, %s);
            """, (obj['field_search_id'], sport_name, desc))
            
            if obj.get('adjusted_gps') and obj['adjusted_gps'] != "0.0,0.0" and "http" not in str(obj['adjusted_gps']):
                cur.execute("""
                    UPDATE public.new_google_earth 
                    SET gps_location = %s 
                    WHERE field_search_id = %s;
                """, (obj['adjusted_gps'], obj['field_search_id']))
            
        conn.commit()
        print(f"Successfully updated database and logged {len(nge_objects)} details in nge_object.")
    except Exception as e:
        print(f"Error writing to database tables: {e}")
        if conn: conn.rollback()
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)
def save_single_object_data(cur, field_search_id, sport_name, conf, status="DETECTED"):
    """
    Saves a single record to public.nge_object with an explicit description
    based on the processing status.
    """
    if status == "FETCH_ERROR":
        desc = "Image fetch failed: Satellite API error or timeout"
    elif status == "ZERO_DETECTIONS":
        desc = "No fields detected by YOLO (0% confidence)"
    else:
        conf_percent = conf * 100 if conf <= 1.0 else conf
        desc = f"Detected via YOLO with {conf_percent:.1f}% confidence"

    cur.execute("""
        INSERT INTO public.nge_object (field_search_id, sport_name, description)
        VALUES (%s, %s, %s);
    """, (field_search_id, sport_name, desc))
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
            # --- CORRECTED: Changed get_fields_from_google to get_field_data ---
            google_json = get_field_data(zip_code, sport_name, KEY)
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
    # Retrieve database records strictly for the target ZIP code
    print(f"Retrieving database records strictly for ZIP code: {zip_code}")
    db_fields = []
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        cur.execute("""
            SELECT field_search_id, field_name, gps_location, search_sport_type, gearth_link 
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
    
    conn = pool.getconn()
    cur = conn.cursor()

    try:
        for row in db_fields:
        # Per-row exception block ensures individual field errors do not abort the loop
            try:
                field_id = row['field_search_id']
                gps_raw = str(row['gps_location']).strip() if row['gps_location'] else ""
                sport_type_id = row.get('search_sport_type')
                default_sport_label = SPORT_TAGS.get(sport_type_id, "Sports Facility").title()

            # Fix links/coordinates stored as URLs
                if "http" in gps_raw:
                    match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', gps_raw) or re.search(r'(-?\d+\.\d+),\s*(-?\d+\.\d+)', gps_raw)
                    if match:
                        clean_lat, clean_lon = match.group(1), match.group(2)
                        clean_gps = f"{clean_lat},{clean_lon}"
                        clean_link = f"https://earth.google.com/web/@{clean_gps},4.1972381a,15000d"
                    
                        cur.execute("""
                            UPDATE public.new_google_earth 
                            SET gps_location = %s, gearth_link = COALESCE(gearth_link, %s)
                            WHERE field_search_id = %s;
                        """, (clean_gps, clean_link, field_id))
                        conn.commit()
                        gps_raw = clean_gps

                if not row.get('gearth_link') and "," in gps_raw:
                    clean_link = f"https://earth.google.com/web/@{gps_raw},4.1972381a,15000d"
                    cur.execute("""
                        UPDATE public.new_google_earth 
                        SET gearth_link = %s 
                        WHERE field_search_id = %s;
                    """, (clean_link, field_id))
                    conn.commit()

                # Process satellite detection
                if "," in gps_raw and "http" not in gps_raw:
                    print(f"Processing field ID {field_id}: '{row['field_name']}' at {gps_raw}")
                    parts = gps_raw.split(",")
                    lat, lon = float(parts[0]), float(parts[1])
                    gps_loc = {'latitude': lat, 'longitude': lon}
                    img_array = get_satellite_image_array(gps_loc)
                
                    modified_locations = []
                    if img_array is not None:
                        modified_locations = object_detection_based_modification_by_class(
                            field_id, img_array, obd_model, gps_loc, all_target_class_ids, zoom_level=18
                        )
                
                    if modified_locations:
                        nge_object.extend(modified_locations)
                    else:
                        print(f"ℹYOLO found 0 objects for '{row['field_name']}'. Creating default object entry...")
                        nge_object.append({
                            'field_search_id': field_id,
                            'sport_name': default_sport_label,
                            'confidence_score': 0.0,
                            'adjusted_gps': gps_raw
                        })
                else:
                    nge_object.append({
                        'field_search_id': field_id,
                        'sport_name': default_sport_label,
                        'confidence_score': 0.0,
                        'adjusted_gps': "0.0,0.0"
                    })
            except Exception as row_error:
                print(f"Exception on field ID {row.get('field_search_id')}: {row_error}. Generating fallback object entry...")
                field_id = row.get('field_search_id')
                if field_id:
                    sport_type_id = row.get('search_sport_type')
                    default_sport_label = SPORT_TAGS.get(sport_type_id, "Sports Facility").title()
                    gps_raw = str(row.get('gps_location', '')).strip()
                    nge_object.append({
                        'field_search_id': field_id,
                        'sport_name': default_sport_label,
                        'confidence_score': 0.0,
                        'adjusted_gps': gps_raw if gps_raw else "0.0,0.0"
                    })

    finally:
        cur.close()
        pool.putconn(conn)
    
    if nge_object:
        save_object_data(nge_object)
    print("\nRunning database integrity check...")
    backfill_missing_objects()
    print("Data input process completed successfully.")