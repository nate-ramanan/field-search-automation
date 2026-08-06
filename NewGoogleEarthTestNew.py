import cv2
import time
import requests
import psycopg2
import pandas as pd
from pathlib import Path
from configparser import ConfigParser, NoSectionError
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
import pyproj
from pyproj import Transformer
import collections.abc
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from ConnectionPool import pool

# ─────────────────────────────────────────────────────────
# CONFIG & MODEL SETUP
# ─────────────────────────────────────────────────────────
KEY = 'AIzaSyC5cT2KgRuUuz51GQ71DvY8gB_VN8O8EtE'

config_path = Path(__file__).resolve().parent / 'config.ini'
config = ConfigParser()
config.read(config_path)

if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
if not config.has_section('model_paths'):
    raise NoSectionError('model_paths')

print("Loaded Config Sections:", config.sections())
print("Model Paths Config:", dict(config.items('model_paths')))

obd_model_path = config.get('model_paths', 'obd_model')
obd_model = YOLO(obd_model_path, verbose=False)

display_names = {
    0: 'Expressway-Service-area', 1: 'Expressway-toll-station', 2: 'airplane',
    3: 'airport',  4: 'Baseball',   5: 'Basketball',
    6: 'bridge',   7: 'chimney',    8: 'dam',        9: 'Golf',
    10: 'Soccer',  11: 'harbor',    12: 'overpass',
    13: 'ship',    14: 'Stadium',   15: 'storagetank',
    16: 'Tennis',  17: 'trainstation', 18: 'vehicle', 19: 'windmill'
}

SPORT_TAGS = {
    8:  "baseball",
    9:  "basketball",
    79: "soccer",
    87: "tennis",
    90: "volleyball"
}

SPORT_LABELS = {
    8:  "Baseball",
    9:  "Basketball",
    79: "Soccer",
    87: "Tennis",
    90: "Volleyball",
}

SPORT_SINGULAR = {
    8:  "baseball field",
    9:  "basketball court",
    79: "soccer field",
    87: "tennis court",
    90: "volleyball court",
}

SPORT_ORDER = [8, 9, 79, 87, 90]

GENERIC_NAMES = {
    "osm pitch location", "osm basketball pitch", "osm asphalt basketball pitch",
    "osm concrete basketball pitch", "osm dirt basketball pitch",
    "osm baseball pitch", "osm grass baseball pitch", "osm dirt baseball pitch",
    "osm soccer pitch", "osm tennis pitch", "osm volleyball pitch",
    "osm pitch facility", "osm soccer facility", "osm tennis facility",
    "osm baseball facility", "osm basketball facility",
}

NAMED_PROXIMITY_M    = 120
DB_MATCH_PROXIMITY_M = 200
SQL_DEDUP_KM         = 0.08
GENERIC_PROXIMITY_M  = 30
GPS_CLUSTER_RADIUS_M = 80

# ─────────────────────────────────────────────────────────
# PERFORMANCE: Photon reverse geocoding cache
# Avoids duplicate HTTP calls for the same GPS coordinate
# when it appears in multiple sport queries
# ─────────────────────────────────────────────────────────
_photon_cache = {}


# ─────────────────────────────────────────────────────────
# SATELLITE & BOUNDARY FUNCTIONS
# ─────────────────────────────────────────────────────────

def getImage(lat, lon, key, zoom, width, height):
    if lat != 'Error' and lon != 'Error':
        return (f"https://maps.googleapis.com/maps/api/staticmap"
                f"?key={key}&center={lat},{lon}&zoom={zoom}"
                f"&size={width}x{height}&maptype=satellite")
    return 'Error'


def get_satellite_image_array(gps_location, zoom_level=18, size=(800, 850)):
    lat = gps_location['latitude']
    lon = gps_location['longitude']
    image_url = getImage(lat, lon, KEY, zoom_level, size[0], size[1])
    if image_url == 'Error':
        return None
    try:
        response = requests.get(image_url, headers={"User-Agent": "SportsFacilityFinder/1.0"})
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return np.array(img.convert('RGB'))
    except Exception as e:
        print(f"Error fetching satellite image: {e}")
        return None


def fetch_zip_bbox_via_photon(zip_code, state_code):
    url     = "https://photon.komoot.io/api/"
    params  = {"q": f"{zip_code}, {state_code}, United States", "limit": 1}
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json()
        if data.get("features"):
            extent = data["features"][0].get("properties", {}).get("extent")
            if extent and len(extent) == 4:
                south_lat = min(extent[1], extent[3])
                north_lat = max(extent[1], extent[3])
                west_lon  = min(extent[0], extent[2])
                east_lon  = max(extent[0], extent[2])
                return (south_lat, west_lon, north_lat, east_lon)
    except Exception as e:
        print(f"Photon bounding geocoding failed: {e}")
    return None


# ─────────────────────────────────────────────────────────
# DATA SOURCE FETCHING FUNCTIONS
# ─────────────────────────────────────────────────────────

def get_fields_from_overpass(bbox, sport_name):
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter"
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
        for attempt in range(2):
            try:
                # PERFORMANCE: reduced timeout 15s -> 8s for faster failover
                response = requests.post(endpoint, data={"data": query},
                                         headers=headers, timeout=8)
                if response.status_code == 429:
                    print(f"⚠️ {endpoint} rate-limited. Retrying in 3s...")
                    time.sleep(3)
                    continue
                if response.status_code >= 500:
                    print(f"⚠️ {endpoint} returned {response.status_code}. Swapping mirrors...")
                    break
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"⚠️ {endpoint} (Attempt {attempt+1}/2): {e}")
                time.sleep(1)
    print(f"All Overpass mirrors failed for sport '{sport_name}'.")
    return {"elements": []}


def get_fields_from_google(zip_code, sport_name, api_key):
    url    = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": f"{sport_name} field in {zip_code}", "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Google Places failed for '{sport_name}' in {zip_code}: {e}")
        return {"results": []}


# ─────────────────────────────────────────────────────────
# SHARED HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Returns great-circle distance in metres between two GPS points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6_371_000


def get_base_name(name):
    if not name:
        return ""
    return re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()


def clean_raw_name(raw_name):
    if not raw_name:
        return raw_name
    cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', raw_name).strip()
    return cleaned if cleaned else raw_name


def is_generic(name):
    return get_base_name(name).strip().lower() in GENERIC_NAMES


def normalize_city(city):
    if not city:
        return ""
    return city.strip().title()


def normalize_state(state):
    if not state:
        return ""
    state = state.strip()
    if len(state) == 2:
        return state.upper()
    return state.title()


def extract_sports_from_name(field_name):
    valid_sports = set(SPORT_LABELS.values())
    paren_match  = re.search(r'\(([^)]+)\)\s*$', field_name or "")
    if not paren_match:
        return set()
    found = set()
    for part in paren_match.group(1).split(','):
        part = part.strip().title()
        if part in valid_sports:
            found.add(part)
            continue
        for sport in valid_sports:
            if part.startswith(sport):
                found.add(sport)
                break
    return found


def build_sport_label_string(sport_ids):
    labels = sorted(SPORT_LABELS[sid] for sid in sport_ids if sid in SPORT_LABELS)
    return ", ".join(labels)


def build_sport_summary(sport_counts):
    parts = []
    for sid in SPORT_ORDER:
        count = sport_counts.get(sid, 0)
        if count > 0:
            label = SPORT_SINGULAR[sid]
            parts.append(f"{count} {label}{'s' if count > 1 else ''}")
    return ", ".join(parts)


def compute_sport_counts(sport_ids_or_counts):
    if isinstance(sport_ids_or_counts, set):
        counts = {sid: 1 for sid in sport_ids_or_counts}
    else:
        counts = dict(sport_ids_or_counts)
    result = {
        'baseball_fields':   counts.get(8,  0),
        'basketball_courts': counts.get(9,  0),
        'soccer_fields':     counts.get(79, 0),
        'tennis_courts':     counts.get(87, 0),
        'volleyball_courts': counts.get(90, 0),
    }
    result['total_fields'] = sum(result.values())
    return result


# ─────────────────────────────────────────────────────────
# PARSING & ENRICHMENT FUNCTIONS
# ─────────────────────────────────────────────────────────

def extract_clean_address(tags):
    housenumber = tags.get('addr:housenumber', '').strip()
    street      = tags.get('addr:street', '').strip()
    city        = tags.get('addr:city', '').strip()
    postcode    = tags.get('addr:postcode', '').strip()
    state       = tags.get('addr:state', '').strip()
    full_addr   = tags.get('addr:full', '')
    if full_addr:
        return full_addr.strip()
    parts = []
    if housenumber and street:
        parts.append(f"{housenumber} {street}")
    elif street:
        parts.append(street)
    if city:     parts.append(city)
    if state:    parts.append(state)
    if postcode: parts.append(postcode)
    return ", ".join(parts) if parts else None


def construct_facility_name(tags, default_sport_name="Field"):
    if tags.get('name'):
        return tags.get('name').strip()
    if tags.get('container_name'):
        container = tags.get('container_name').strip()
        sport = tags.get('sport', default_sport_name).replace('_', ' ').title()
        return f"{container} ({sport} Facility)"
    operator = tags.get('operator', '').strip()
    surface  = tags.get('surface', '').strip().replace('_', ' ').title()
    sport    = tags.get('sport', default_sport_name).replace('_', ' ').title()
    leisure  = tags.get('leisure', '').strip().replace('_', ' ').title()
    name_parts = []
    if operator:   name_parts.append(operator)
    elif surface:  name_parts.append(f"OSM {surface}")
    else:          name_parts.append("OSM")
    name_parts.append(sport)
    name_parts.append(leisure if leisure else "Facility")
    return " ".join(name_parts)


def get_nearest_address_via_osm(lat, lon, radius=200):
    endpoint = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:8];
    nwr["addr:housenumber"]["addr:street"](around:{radius},{lat},{lon});
    out center;
    """
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    try:
        # PERFORMANCE: reduced timeout 15s -> 8s
        response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=8)
        if response.status_code == 200:
            elements = response.json().get("elements", [])
            if elements:
                tags = elements[0].get("tags", {})
                return tags.get("addr:housenumber", "").strip(), tags.get("addr:street", "").strip()
    except Exception as e:
        print(f"Failed to find nearest address via Overpass: {e}")
    return None, None


def get_osm_parent_name(lat, lon, radius=80):
    endpoint = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:8];
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
        # PERFORMANCE: reduced timeout 15s -> 8s
        response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=8)
        if response.status_code == 200:
            elements = response.json().get("elements", [])
            if elements:
                return elements[0].get("tags", {}).get("name", "").strip()
    except Exception as e:
        print(f"Failed to fetch OSM parent container name: {e}")
    return None


def enrich_location_via_photon(lat, lon, default_city, default_state, default_zip, sport_label):
    # PERFORMANCE: cache check — same GPS point queried by multiple sports
    # Round to 4 decimal places (~11m precision) as cache key
    cache_key = (round(lat, 4), round(lon, 4))
    if cache_key in _photon_cache:
        return _photon_cache[cache_key]

    url     = "https://photon.komoot.io/reverse"
    params  = {"lat": lat, "lon": lon, "lang": "en"}
    headers = {"User-Agent": "SportsFacilityFinder/1.0"}
    name     = f"OSM {sport_label} Facility"
    address  = f"{default_city}, {default_state} {default_zip}"
    street   = ""
    city     = default_city
    state    = default_state
    postcode = default_zip
    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)
        if res.status_code == 200:
            features = res.json().get("features", [])
            if features:
                props           = features[0].get("properties", {})
                photon_name     = props.get("name", "").strip()
                photon_street   = props.get("street", "").strip()
                photon_house    = props.get("housenumber", "").strip()
                photon_city     = props.get("city", props.get("town", default_city)).strip()
                photon_state    = props.get("state", default_state).strip()
                photon_postcode = props.get("postcode", default_zip).strip()
                if photon_name and photon_name != photon_street:
                    name = photon_name
                elif photon_street:
                    name = photon_street
                if not photon_street:
                    print(f"Address details missing for '{photon_name}' at {lat},{lon}. Scanning for closest address...")
                    nearest_house, nearest_road = get_nearest_address_via_osm(lat, lon)
                    if nearest_road:
                        photon_street = nearest_road
                        if nearest_house:
                            photon_house = nearest_house
                street_line = (f"{photon_house} {photon_street}".strip()
                               if photon_house and photon_street else
                               photon_street if photon_street else photon_name)
                street   = street_line if street_line else "Unnamed Road"
                address  = (f"{street_line}, {photon_city}, {photon_state} {photon_postcode}"
                            if street_line else
                            f"{photon_city}, {photon_state} {photon_postcode}")
                city     = photon_city
                state    = photon_state
                postcode = photon_postcode
    except Exception as e:
        print(f"Photon reverse enrichment failed for ({lat}, {lon}): {e}")

    result = name, address, street, city, state, postcode
    # PERFORMANCE: store in cache before returning
    _photon_cache[cache_key] = result
    return result


def parse_osm_elements(osm_data, search_sport_type, city, state, postal_code):
    """
    Parses OSM elements with PARALLEL enrichment (3 threads).
    All unnamed pitch lookups (Photon + OSM parent) run 3 at a time
    instead of sequentially — major speed improvement on large ZIPs.
    """
    sport_str = SPORT_TAGS.get(search_sport_type, "Field").title()
    elements  = [
        el for el in osm_data.get("elements", [])
        if el.get("center", {}).get("lat", el.get("lat"))
        and el.get("center", {}).get("lon", el.get("lon"))
    ]

    def process_element(el):
        lat = el.get("center", {}).get("lat", el.get("lat"))
        lon = el.get("center", {}).get("lon", el.get("lon"))
        tags               = el.get("tags", {})
        osm_native_id      = f"{el.get('type', 'node')}/{el.get('id', 0)}"
        has_native_name    = tags.get("name") is not None
        has_native_address = tags.get("addr:street") is not None

        # Resolve name
        if has_native_name:
            raw_name = tags.get("name").strip()
        else:
            print(f"Pitch missing name at {lat},{lon}. Searching OSM parent boundaries...")
            parent_name = get_osm_parent_name(lat, lon)
            raw_name    = parent_name if parent_name else None

        # Resolve address
        if has_native_address:
            street_field   = tags.get('addr:street', '').strip()
            housenumber    = tags.get('addr:housenumber', '').strip()
            if housenumber:
                street_field = f"{housenumber} {street_field}"
            city_field     = normalize_city(tags.get('addr:city', city))
            state_field    = normalize_state(tags.get('addr:state', state))
            postcode_field = tags.get('addr:postcode', postal_code).strip()
            formatted_address = tags.get('addr:full',
                f"{street_field}, {city_field}, {state_field} {postcode_field}").strip()
            if not raw_name:
                raw_name = street_field if street_field else None
        else:
            photon_name, formatted_address, street_field, city_field, state_field, postcode_field = \
                enrich_location_via_photon(lat, lon, city, state, postal_code, sport_str)
            city_field  = normalize_city(city_field)
            state_field = normalize_state(state_field)
            if not raw_name:
                raw_name = clean_raw_name(photon_name)

        return {
            'field_name':            raw_name,
            'formatted_address':     formatted_address,
            'postal_code':           postal_code,
            'street':                street_field,
            'city':                  city_field,
            'state':                 state_field,
            'original_gps_location': {'latitude': float(lat), 'longitude': float(lon)},
            'gplace_id':             osm_native_id,
            'search_sport_type':     search_sport_type,
            'gearth_link':           f"https://earth.google.com/web/@{lat},{lon},4.1972381a,15000d",
            'modified_fields':       []
        }

    # PERFORMANCE: process 3 elements at a time in parallel
    fields = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_element, el): el for el in elements}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    fields.append(result)
            except Exception as e:
                print(f"Error processing OSM element: {e}")

    return fields


def parse_google_elements(google_data, search_sport_type, city, state, postal_code):
    fields = []
    for res in google_data.get("results", []):
        lat      = res["geometry"]["location"]["lat"]
        lon      = res["geometry"]["location"]["lng"]
        place_id = res.get("place_id", f"unknown_google_{lat}_{lon}")
        formatted_address = res.get("formatted_address", "")
        street_field = formatted_address.split(",")[0].strip() if formatted_address else ""
        fields.append({
            'field_name':            res.get("name", "Google Facility Location"),
            'formatted_address':     formatted_address or f"{city}, {state} {postal_code}",
            'postal_code':           postal_code,
            'street':                street_field,
            'city':                  normalize_city(city),
            'state':                 normalize_state(state),
            'original_gps_location': {'latitude': float(lat), 'longitude': float(lon)},
            'gplace_id':             place_id,
            'search_sport_type':     search_sport_type,
            'gearth_link':           f"https://earth.google.com/web/@{lat},{lon},4.1972381a,15000d",
            'modified_fields':       []
        })
    return fields


# ─────────────────────────────────────────────────────────
# new_google_earth PIPELINE
# Deduplicates by BASE NAME + ZIP CODE
# ─────────────────────────────────────────────────────────

def group_incoming_fields(fields):
    """
    Stage 1 — in-memory grouping before DB write.
    Groups by: same base name + 120m, same address + 120m, same street + name.
    Generic OSM names only within 30m.
    """
    grouped = []
    for f in fields:
        lat       = f['original_gps_location']['latitude']
        lon       = f['original_gps_location']['longitude']
        street    = (f['street'] or '').strip().lower()
        addr      = str(f['formatted_address'] or '').strip().lower()
        base_name = get_base_name(f['field_name']).lower()
        generic   = is_generic(f['field_name'])
        match = None
        for g in grouped:
            g_lat     = g['original_gps_location']['latitude']
            g_lon     = g['original_gps_location']['longitude']
            g_street  = (g['street'] or '').strip().lower()
            g_addr    = str(g['formatted_address'] or '').strip().lower()
            g_base    = get_base_name(g['field_name']).lower()
            g_generic = is_generic(g['field_name'])
            dist      = haversine(lat, lon, g_lat, g_lon)
            if generic or g_generic:
                if base_name == g_base and dist < GENERIC_PROXIMITY_M:
                    match = g; break
            else:
                name_match   = base_name == g_base and base_name != "" and dist < NAMED_PROXIMITY_M
                addr_match   = addr not in ('', 'nan') and addr == g_addr and dist < NAMED_PROXIMITY_M
                street_match = street == g_street and base_name == g_base and street != "" and base_name != ""
                if name_match or addr_match or street_match:
                    match = g; break
        sport_id = f['search_sport_type']
        if match:
            match['sport_ids'].add(sport_id)
            match['sport_counts'][sport_id] += 1
            match['coords_list'].append((lat, lon))
        else:
            f_copy                 = f.copy()
            f_copy['sport_ids']    = {sport_id}
            f_copy['sport_counts'] = defaultdict(int, {sport_id: 1})
            f_copy['coords_list']  = [(lat, lon)]
            grouped.append(f_copy)
    for g in grouped:
        lats = [c[0] for c in g['coords_list']]
        lons = [c[1] for c in g['coords_list']]
        g['original_gps_location']['latitude']  = sum(lats) / len(lats)
        g['original_gps_location']['longitude'] = sum(lons) / len(lons)
    print(f"  In-memory grouping: {len(fields)} raw records -> {len(grouped)} unique facilities")
    return grouped


def save_field_data(fields):
    """
    Stage 2 — DB upsert with name+ZIP dedup.
    Step A: exact gplace_id lookup.
    Step B: spatial/name/address fallback + cross-ZIP name check.
    Step C: INSERT or UPDATE with combined sport IDs and counts.
    """
    if not fields:
        return
    grouped_fields = group_incoming_fields(fields)
    target_zip     = fields[0]['postal_code']
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT field_search_id, field_name, gps_location,
                   street, formatted_address, gplace_id
            FROM public.new_google_earth
            WHERE postal_code = %s;
        """, (target_zip,))
        db_records   = []
        db_by_gplace = {}
        for row in cur.fetchall():
            rec = {
                'field_search_id':   row[0],
                'field_name':        row[1],
                'gps_location':      row[2],
                'street':            row[3],
                'formatted_address': row[4],
                'gplace_id':         row[5],
            }
            db_records.append(rec)
            if row[5]:
                db_by_gplace[row[5]] = rec

        # Cross-ZIP name lookup — catches same facility under different ZIP
        base_names_incoming = list({
            get_base_name(clean_raw_name(f['field_name']))
            for f in grouped_fields
            if get_base_name(clean_raw_name(f['field_name']))
        })
        if base_names_incoming:
            cur.execute("""
                SELECT field_search_id, field_name, gps_location,
                       street, formatted_address, gplace_id
                FROM public.new_google_earth
                WHERE postal_code != %s
                  AND REGEXP_REPLACE(field_name, '\\s*\\([^)]*\\)\\s*$', '') = ANY(%s);
            """, (target_zip, base_names_incoming))
            for row in cur.fetchall():
                rec = {
                    'field_search_id':   row[0],
                    'field_name':        row[1],
                    'gps_location':      row[2],
                    'street':            row[3],
                    'formatted_address': row[4],
                    'gplace_id':         row[5],
                }
                if row[5] not in db_by_gplace:
                    db_records.append(rec)

        inserted_count = 0
        updated_count  = 0

        for f in grouped_fields:
            lat       = f['original_gps_location']['latitude']
            lon       = f['original_gps_location']['longitude']
            gps_str   = f"{lat},{lon}"
            street    = (f['street'] or '').strip().lower()
            addr      = str(f['formatted_address'] or '').strip().lower()
            base_name = get_base_name(f['field_name']).lower()

            # Step A: exact gplace_id lookup
            db_match = db_by_gplace.get(f['gplace_id'])

            # Step B: spatial/name/address fallback
            if not db_match:
                for db in db_records:
                    try:
                        db_lat, db_lon = map(float, db['gps_location'].split(','))
                    except (ValueError, AttributeError):
                        continue
                    db_street    = (db['street'] or '').strip().lower()
                    db_addr      = str(db['formatted_address'] or '').strip().lower()
                    db_base_name = get_base_name(db['field_name']).lower()
                    dist         = haversine(lat, lon, db_lat, db_lon)
                    name_match   = db_base_name == base_name and base_name != "" and dist < DB_MATCH_PROXIMITY_M
                    addr_match   = addr not in ('', 'nan') and addr == db_addr and dist < DB_MATCH_PROXIMITY_M
                    street_match = street == db_street and base_name == db_base_name and street != "" and base_name != ""
                    if name_match or addr_match or street_match:
                        db_match = db; break

            # Build combined sport IDs (DB existing + incoming batch)
            existing_sport_labels = extract_sports_from_name(
                db_match['field_name'] if db_match else ""
            )
            label_to_id        = {v: k for k, v in SPORT_LABELS.items()}
            existing_sport_ids = {label_to_id[lbl] for lbl in existing_sport_labels if lbl in label_to_id}
            combined_sport_ids = existing_sport_ids.union(f['sport_ids'])

            sports_label     = build_sport_label_string(combined_sport_ids)
            clean_base       = clean_raw_name(f['field_name'])
            final_field_name = f"{clean_base} ({sports_label})" if clean_base else f"Facility ({sports_label})"

            # Counts from combined_sport_ids — never resets on re-runs
            baseball_cnt     = 1 if 8  in combined_sport_ids else 0
            basketball_cnt   = 1 if 9  in combined_sport_ids else 0
            soccer_cnt       = 1 if 79 in combined_sport_ids else 0
            tennis_cnt       = 1 if 87 in combined_sport_ids else 0
            volleyball_cnt   = 1 if 90 in combined_sport_ids else 0
            total_cnt        = baseball_cnt + basketball_cnt + soccer_cnt + tennis_cnt + volleyball_cnt or 1

            city_to_store  = normalize_city(f['city'])
            state_to_store = normalize_state(f['state'])

            print(f"  {'UPDATE' if db_match else 'INSERT'}: {final_field_name}")

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
                        search_sport_type = %s,
                        baseball_fields   = %s,
                        basketball_courts = %s,
                        soccer_fields     = %s,
                        tennis_courts     = %s,
                        volleyball_courts = %s,
                        total_fields      = %s
                    WHERE field_search_id = %s;
                """, (
                    final_field_name, f['formatted_address'], f['postal_code'],
                    f['street'], city_to_store, state_to_store, gps_str,
                    f['gearth_link'], f['search_sport_type'],
                    baseball_cnt, basketball_cnt, soccer_cnt,
                    tennis_cnt, volleyball_cnt, total_cnt,
                    db_match['field_search_id'],
                ))
                updated_count += 1
            else:
                cur.execute("""
                    INSERT INTO public.new_google_earth
                    (field_name, formatted_address, postal_code, street, city, state,
                     gps_location, gearth_link, search_sport_type, gplace_id,
                     baseball_fields, basketball_courts,
                     soccer_fields, tennis_courts, volleyball_courts, total_fields)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (gplace_id) DO UPDATE SET
                        field_name        = EXCLUDED.field_name,
                        formatted_address = EXCLUDED.formatted_address,
                        postal_code       = EXCLUDED.postal_code,
                        street            = EXCLUDED.street,
                        city              = EXCLUDED.city,
                        state             = EXCLUDED.state,
                        gps_location      = EXCLUDED.gps_location,
                        gearth_link       = EXCLUDED.gearth_link,
                        search_sport_type = EXCLUDED.search_sport_type,
                        baseball_fields   = EXCLUDED.baseball_fields,
                        basketball_courts = EXCLUDED.basketball_courts,
                        soccer_fields     = EXCLUDED.soccer_fields,
                        tennis_courts     = EXCLUDED.tennis_courts,
                        volleyball_courts = EXCLUDED.volleyball_courts,
                        total_fields      = EXCLUDED.total_fields;
                """, (
                    final_field_name, f['formatted_address'], f['postal_code'],
                    f['street'], city_to_store, state_to_store, gps_str,
                    f['gearth_link'], f['search_sport_type'], f['gplace_id'],
                    baseball_cnt, basketball_cnt, soccer_cnt,
                    tennis_cnt, volleyball_cnt, total_cnt,
                ))
                inserted_count += 1

        conn.commit()
        print(f"\nProcessed {len(grouped_fields)} unique facilities "
              f"(Inserted: {inserted_count}, Updated: {updated_count})")
    except Exception as e:
        print("Database error inside save_field_data:", e)
        import traceback
        traceback.print_exc()
        if conn: conn.rollback()
    finally:
        if cur:  cur.close()
        if conn: pool.putconn(conn)


def delete_duplicates(target_zip):
    """Stage 3 — SQL Haversine cleanup within 80m after DB write."""
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT p1.field_search_id AS keep_id,
                   p2.field_search_id AS drop_id,
                   p2.search_sport_type AS drop_sport
            FROM (
                SELECT field_search_id, search_sport_type,
                       SPLIT_PART(gps_location,',',1)::float AS lat,
                       SPLIT_PART(gps_location,',',2)::float AS lon
                FROM public.new_google_earth
                WHERE postal_code = %s
                  AND gps_location ~ '^[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+$'
            ) p1
            JOIN (
                SELECT field_search_id, search_sport_type,
                       SPLIT_PART(gps_location,',',1)::float AS lat,
                       SPLIT_PART(gps_location,',',2)::float AS lon
                FROM public.new_google_earth
                WHERE postal_code = %s
                  AND gps_location ~ '^[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+$'
            ) p2
              ON p1.field_search_id < p2.field_search_id
             AND (6371 * acos(LEAST(1.0, GREATEST(-1.0,
                     cos(radians(p1.lat))*cos(radians(p2.lat))
                     *cos(radians(p2.lon)-radians(p1.lon))
                     +sin(radians(p1.lat))*sin(radians(p2.lat))
                 )))) <= %s;
        """, (target_zip, target_zip, SQL_DEDUP_KM))
        pairs = cur.fetchall()
        if not pairs:
            print(f"No spatial duplicates found for ZIP {target_zip}.")
            return
        keep_to_drop_sports = defaultdict(set)
        drop_ids = set()
        for keep_id, drop_id, drop_sport in pairs:
            keep_to_drop_sports[keep_id].add(drop_sport)
            drop_ids.add(drop_id)
        label_to_id = {v: k for k, v in SPORT_LABELS.items()}
        for keep_id, extra_sport_ids in keep_to_drop_sports.items():
            cur.execute("SELECT field_name FROM public.new_google_earth WHERE field_search_id = %s;", (keep_id,))
            row = cur.fetchone()
            if not row: continue
            existing_ids = {label_to_id[l] for l in extract_sports_from_name(row[0]) if l in label_to_id}
            combined_ids = existing_ids.union(extra_sport_ids)
            new_label    = build_sport_label_string(combined_ids)
            clean_base   = clean_raw_name(row[0])
            new_name     = f"{clean_base} ({new_label})" if clean_base else f"Facility ({new_label})"
            sc           = compute_sport_counts(combined_ids)
            cur.execute("""
                UPDATE public.new_google_earth SET
                    field_name        = %s,
                    baseball_fields   = %s, basketball_courts = %s,
                    soccer_fields     = %s, tennis_courts     = %s,
                    volleyball_courts = %s, total_fields      = %s
                WHERE field_search_id = %s;
            """, (new_name, sc['baseball_fields'], sc['basketball_courts'],
                  sc['soccer_fields'], sc['tennis_courts'],
                  sc['volleyball_courts'], sc['total_fields'], keep_id))
        cur.execute("DELETE FROM public.nge_object WHERE field_search_id = ANY(%s);", (list(drop_ids),))
        cur.execute("DELETE FROM public.new_google_earth WHERE field_search_id = ANY(%s);", (list(drop_ids),))
        conn.commit()
        print(f"Spatial dedup for ZIP {target_zip}: removed {len(drop_ids)} duplicates (80m)")
    except Exception as error:
        print("Error during spatial dedup:", error)
        import traceback; traceback.print_exc()
        if conn: conn.rollback()
    finally:
        if cur:  cur.close()
        if conn: pool.putconn(conn)


def merge_same_name_facilities(target_zip):
    """Stage 4 — merge same-name rows within ZIP regardless of GPS distance."""
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT field_search_id, field_name, search_sport_type
            FROM public.new_google_earth
            WHERE postal_code = %s ORDER BY field_search_id ASC;
        """, (target_zip,))
        rows = cur.fetchall()
        if not rows:
            print(f"No records for ZIP {target_zip}.")
            return
        name_groups = defaultdict(list)
        for fid, fname, stype in rows:
            base = clean_raw_name(fname or "").strip()
            if base:
                name_groups[base].append({'field_search_id': fid, 'field_name': fname, 'sport_type': stype})
        multi_groups = {n: r for n, r in name_groups.items() if len(r) > 1}
        if not multi_groups:
            print(f"No same-name duplicates found for ZIP {target_zip}.")
            return
        merged_count = dropped_total = 0
        label_to_id  = {v: k for k, v in SPORT_LABELS.items()}
        for base_name, records in multi_groups.items():
            records_sorted = sorted(records, key=lambda r: r['field_search_id'])
            keep     = records_sorted[0]
            drop_ids = [r['field_search_id'] for r in records_sorted[1:]]
            all_sport_ids = set()
            for r in records_sorted:
                all_sport_ids.update({label_to_id[l] for l in extract_sports_from_name(r['field_name']) if l in label_to_id})
                if r['sport_type'] in SPORT_LABELS:
                    all_sport_ids.add(r['sport_type'])
            merged_name = f"{base_name} ({build_sport_label_string(all_sport_ids)})"
            sc          = compute_sport_counts(all_sport_ids)
            print(f"  MERGE: {merged_name} (kept ID={keep['field_search_id']}, dropped {len(drop_ids)})")
            cur.execute("""
                UPDATE public.new_google_earth SET
                    field_name        = %s,
                    baseball_fields   = %s, basketball_courts = %s,
                    soccer_fields     = %s, tennis_courts     = %s,
                    volleyball_courts = %s, total_fields      = %s
                WHERE field_search_id = %s;
            """, (merged_name, sc['baseball_fields'], sc['basketball_courts'],
                  sc['soccer_fields'], sc['tennis_courts'],
                  sc['volleyball_courts'], sc['total_fields'], keep['field_search_id']))
            if drop_ids:
                cur.execute("DELETE FROM public.nge_object WHERE field_search_id = ANY(%s);", (drop_ids,))
                cur.execute("DELETE FROM public.new_google_earth WHERE field_search_id = ANY(%s);", (drop_ids,))
            merged_count += 1
            dropped_total += len(drop_ids)
        conn.commit()
        print(f"\nName-based merge for ZIP {target_zip}: merged {merged_count} groups, removed {dropped_total} rows.")
    except Exception as error:
        print("Error during name-based merge:", error)
        import traceback; traceback.print_exc()
        if conn: conn.rollback()
    finally:
        if cur:  cur.close()
        if conn: pool.putconn(conn)


# ─────────────────────────────────────────────────────────
# facility_gps_cluster PIPELINE
# Completely independent — deduplicates by GPS only (80m)
# Uses RAW API fields BEFORE any name+ZIP dedup
# ─────────────────────────────────────────────────────────

def group_by_gps_only(fields):
    """
    Clusters raw API fields purely by GPS distance (80m).
    No name, address or ZIP matching whatsoever.
    """
    clusters = []
    used     = set()
    for i, f in enumerate(fields):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        lat1 = f['original_gps_location']['latitude']
        lon1 = f['original_gps_location']['longitude']
        for j, f2 in enumerate(fields):
            if j in used:
                continue
            lat2 = f2['original_gps_location']['latitude']
            lon2 = f2['original_gps_location']['longitude']
            if haversine(lat1, lon1, lat2, lon2) <= GPS_CLUSTER_RADIUS_M:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)
    print(f"  GPS-only grouping: {len(fields)} raw records -> {len(clusters)} GPS clusters")
    return clusters


def populate_gps_clusters(raw_fields):
    """
    Builds facility_gps_cluster from RAW API fields using GPS-only clustering.
    Independent of new_google_earth name+ZIP logic.
    """
    if not raw_fields:
        print("No raw fields to cluster.")
        return

    clusters    = group_by_gps_only(raw_fields)
    label_to_id = {v: k for k, v in SPORT_LABELS.items()}

    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur  = conn.cursor()
        inserted = 0
        updated  = 0

        for cluster_indices in clusters:
            members = [raw_fields[i] for i in cluster_indices]
            anchor  = members[0]

            # Count actual occurrences per sport (dict not set)
            sport_counts = defaultdict(int)
            for m in members:
                if m['search_sport_type'] in SPORT_LABELS:
                    sport_counts[m['search_sport_type']] += 1

            all_sport_ids = set(sport_counts.keys())
            sports_label  = build_sport_label_string(all_sport_ids)
            sc = {
                'baseball_fields':   sport_counts.get(8,  0),
                'basketball_courts': sport_counts.get(9,  0),
                'soccer_fields':     sport_counts.get(79, 0),
                'tennis_courts':     sport_counts.get(87, 0),
                'volleyball_courts': sport_counts.get(90, 0),
                'total_fields':      sum(sport_counts.values()),
            }

            lats = [m['original_gps_location']['latitude']  for m in members]
            lons = [m['original_gps_location']['longitude'] for m in members]
            centroid_gps = f"{sum(lats)/len(lats):.7f},{sum(lons)/len(lons):.7f}"

            clean_base   = clean_raw_name(anchor['field_name'] or '')
            merged_name  = f"{clean_base} ({sports_label})" if clean_base else f"Facility ({sports_label})"
            source_ids   = ','.join(m['gplace_id'] for m in members if m.get('gplace_id'))

            print(f"  GPS CLUSTER [{len(members)}]: {merged_name} | total={sc['total_fields']}")

            cur.execute("""
                SELECT cluster_id FROM public.facility_gps_cluster
                WHERE gplace_id = %s;
            """, (anchor['gplace_id'],))
            existing = cur.fetchone()

            if existing:
                cur.execute("""
                    UPDATE public.facility_gps_cluster SET
                        field_name        = %s,
                        formatted_address = %s,
                        postal_code       = %s,
                        street            = %s,
                        city              = %s,
                        state             = %s,
                        gps_location      = %s,
                        gearth_link       = %s,
                        search_sport_type = %s,
                        baseball_fields   = %s,
                        basketball_courts = %s,
                        soccer_fields     = %s,
                        tennis_courts     = %s,
                        volleyball_courts = %s,
                        total_fields      = %s,
                        source_field_ids  = %s,
                        cluster_size      = %s,
                        updated_at        = NOW()
                    WHERE cluster_id = %s;
                """, (
                    merged_name, anchor['formatted_address'], anchor['postal_code'],
                    anchor['street'], normalize_city(anchor['city']),
                    normalize_state(anchor['state']), centroid_gps,
                    anchor['gearth_link'], anchor['search_sport_type'],
                    sc['baseball_fields'], sc['basketball_courts'],
                    sc['soccer_fields'],   sc['tennis_courts'],
                    sc['volleyball_courts'], sc['total_fields'],
                    source_ids, len(members), existing[0]
                ))
                updated += 1
            else:
                cur.execute("""
                    INSERT INTO public.facility_gps_cluster
                    (gplace_id, field_name, formatted_address,
                     postal_code, street, city, state,
                     gps_location, gearth_link, search_sport_type,
                     baseball_fields, basketball_courts,
                     soccer_fields, tennis_courts,
                     volleyball_courts, total_fields,
                     source_field_ids, cluster_size)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (gplace_id) DO UPDATE SET
                        field_name        = EXCLUDED.field_name,
                        formatted_address = EXCLUDED.formatted_address,
                        postal_code       = EXCLUDED.postal_code,
                        street            = EXCLUDED.street,
                        city              = EXCLUDED.city,
                        state             = EXCLUDED.state,
                        gps_location      = EXCLUDED.gps_location,
                        gearth_link       = EXCLUDED.gearth_link,
                        search_sport_type = EXCLUDED.search_sport_type,
                        baseball_fields   = EXCLUDED.baseball_fields,
                        basketball_courts = EXCLUDED.basketball_courts,
                        soccer_fields     = EXCLUDED.soccer_fields,
                        tennis_courts     = EXCLUDED.tennis_courts,
                        volleyball_courts = EXCLUDED.volleyball_courts,
                        total_fields      = EXCLUDED.total_fields,
                        source_field_ids  = EXCLUDED.source_field_ids,
                        cluster_size      = EXCLUDED.cluster_size,
                        updated_at        = NOW();
                """, (
                    anchor['gplace_id'], merged_name,
                    anchor['formatted_address'], anchor['postal_code'],
                    anchor['street'], normalize_city(anchor['city']),
                    normalize_state(anchor['state']),
                    centroid_gps, anchor['gearth_link'],
                    anchor['search_sport_type'],
                    sc['baseball_fields'], sc['basketball_courts'],
                    sc['soccer_fields'],   sc['tennis_courts'],
                    sc['volleyball_courts'], sc['total_fields'],
                    source_ids, len(members)
                ))
                inserted += 1

        conn.commit()
        print(f"\nGPS clustering: {len(clusters)} clusters written to facility_gps_cluster "
              f"(Inserted: {inserted}, Updated: {updated}) "
              f"from {len(raw_fields)} raw records.")

    except Exception as e:
        print(f"Error during GPS clustering: {e}")
        import traceback; traceback.print_exc()
        if conn: conn.rollback()
    finally:
        if cur:  cur.close()
        if conn: pool.putconn(conn)


# ─────────────────────────────────────────────────────────
# YOLO DETECTION & OBJECT STORAGE
# ─────────────────────────────────────────────────────────

def object_detection_based_modification_by_class(field_id, img_array, model,
                                                  gps_loc, target_classes, zoom_level=18):
    modified_records = []
    try:
        results = model.predict(img_array, verbose=False)
        for res in results:
            for box in res.boxes:
                class_id   = int(box.cls)
                confidence = float(box.conf)
                if class_id in target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x     = (x1 + x2) / 2
                    center_y     = (y1 + y2) / 2
                    lat_offset   = (center_y - (img_array.shape[0] / 2)) * -0.000005
                    lon_offset   = (center_x - (img_array.shape[1] / 2)) *  0.000007
                    adjusted_lat = gps_loc['latitude']  + lat_offset
                    adjusted_lon = gps_loc['longitude'] + lon_offset
                    sport_label  = display_names.get(class_id, f"Unknown_{class_id}")
                    modified_records.append({
                        'field_search_id': field_id,
                        'sport_name':      sport_label,
                        'confidence_score':confidence,
                        'adjusted_gps':    f"{adjusted_lat},{adjusted_lon}"
                    })
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
    return modified_records


def save_object_data(nge_objects):
    if not nge_objects:
        return
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur  = conn.cursor()
        distinct_field_ids = list(set(obj['field_search_id'] for obj in nge_objects))
        cur.execute("DELETE FROM public.nge_object WHERE field_search_id = ANY(%s);", (distinct_field_ids,))
        for obj in nge_objects:
            conf_percent = obj['confidence_score'] * 100
            desc = f"Detected via YOLO with {conf_percent:.1f}% confidence"
            cur.execute("""
                INSERT INTO public.nge_object (field_search_id, sport_name, description)
                VALUES (%s, %s, %s);
            """, (obj['field_search_id'], obj['sport_name'], desc))
            cur.execute("""
                UPDATE public.new_google_earth SET gps_location = %s
                WHERE field_search_id = %s;
            """, (obj['adjusted_gps'], obj['field_search_id']))
        conn.commit()
        print(f"Saved {len(nge_objects)} YOLO detections to nge_object.")
    except Exception as e:
        print(f"Error writing to nge_object: {e}")
        if conn: conn.rollback()
    finally:
        if cur:  cur.close()
        if conn: pool.putconn(conn)


def log_processed(state_code, city, zip_code):
    print(f"Log Execution Metric: Processed batch for {city}, {state_code} {zip_code}")


# ─────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: User inputs ────────────────────────────────
    state_code = input("Enter state code : ").strip().upper()
    city       = input("Enter city name  : ").strip()
    zip_code   = input("Enter ZIP code   : ").strip()
    city       = normalize_city(city)
    state_code = normalize_state(state_code)

    # ── Step 2: Data source ────────────────────────────────
    print("\nSelect Data Source:")
    print("1. OpenStreetMap (Overpass)")
    print("2. Google Places API")
    print("3. Both")
    source_choice = input("Enter choice (1/2/3): ").strip()

    # ── Step 3: Bounding box ───────────────────────────────
    t_total = time.time()
    t = time.time()
    bbox = fetch_zip_bbox_via_photon(zip_code, state_code)
    print(f"⏱ Bounding box: {time.time()-t:.1f}s")
    if not bbox and source_choice in ['1', '3']:
        print(f"Could not compute bounding box for ZIP {zip_code}. Skipping OSM.")

    # ── Step 4: Query APIs for each sport ──────────────────
    all_fields = []
    for sport_id, sport_name in SPORT_TAGS.items():
        if source_choice in ['1', '3'] and bbox:
            print(f"Querying OSM for: '{sport_name}' in {zip_code}")
            t = time.time()
            osm_json   = get_fields_from_overpass(bbox, sport_name)
            print(f"  ⏱ Overpass fetch: {time.time()-t:.1f}s")
            t = time.time()
            osm_fields = parse_osm_elements(osm_json, sport_id, city, state_code, zip_code)
            print(f"  ⏱ OSM parse+enrich (3 threads): {time.time()-t:.1f}s | {len(osm_fields)} records")
            all_fields.extend(osm_fields)
        if source_choice in ['2', '3']:
            print(f"Querying Google Places for: '{sport_name}' in {zip_code}")
            t = time.time()
            google_json   = get_fields_from_google(zip_code, sport_name, KEY)
            google_fields = parse_google_elements(google_json, sport_id, city, state_code, zip_code)
            print(f"  ⏱ Google fetch+parse: {time.time()-t:.1f}s | {len(google_fields)} records")
            all_fields.extend(google_fields)

    print(f"\nTotal raw API records collected: {len(all_fields)}")

    # ── Step 5a: GPS-only clustering → facility_gps_cluster ─
    print(f"\n--- facility_gps_cluster (GPS-only dedup) ---")
    t = time.time()
    if all_fields:
        populate_gps_clusters(all_fields)
    print(f"⏱ GPS clustering: {time.time()-t:.1f}s")

    # ── Step 5b: Name+ZIP dedup → new_google_earth ─────────
    print(f"\n--- new_google_earth (name+ZIP dedup) ---")
    t = time.time()
    if all_fields:
        save_field_data(all_fields)
        log_processed(state_code, city, zip_code)
    else:
        print(f"No fields found for ZIP {zip_code}.")
    print(f"⏱ save_field_data: {time.time()-t:.1f}s")

    # ── Step 6: SQL spatial cleanup ────────────────────────
    t = time.time()
    delete_duplicates(zip_code)
    print(f"⏱ delete_duplicates: {time.time()-t:.1f}s")

    # ── Step 7: Merge same-name facilities ─────────────────
    t = time.time()
    merge_same_name_facilities(zip_code)
    print(f"⏱ merge_same_name: {time.time()-t:.1f}s")

    # ── Step 8: YOLO satellite detection ───────────────────
    # Set RUN_YOLO = False to skip during testing (saves ~60 min)
    RUN_YOLO = True

    print(f"\nRetrieving DB records for ZIP {zip_code} for YOLO processing...")
    db_fields = []
    try:
        conn = pool.getconn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT field_search_id, field_name, gps_location
            FROM public.new_google_earth
            WHERE postal_code = %s;
        """, (zip_code,))
        columns   = [desc[0] for desc in cur.description]
        db_fields = [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching fields for YOLO: {e}")
    finally:
        if cur:  cur.close()
        if conn: pool.putconn(conn)

    print(f"Rows for YOLO processing: {len(db_fields)}")
    nge_object           = []
    all_target_class_ids = [4, 5, 9, 10, 14, 16]

    t = time.time()
    if RUN_YOLO:
        for row in db_fields:
            field_id  = row['field_search_id']
            print(f"Processing: {row['field_name']} at {row['gps_location']}")
            lat, lon  = row['gps_location'].split(",")
            gps_loc   = {'latitude': float(lat), 'longitude': float(lon)}
            img_array = get_satellite_image_array(gps_loc)
            if img_array is not None:
                detections = object_detection_based_modification_by_class(
                    field_id, img_array, obd_model, gps_loc,
                    all_target_class_ids, zoom_level=18
                )
                nge_object.extend(detections)
    else:
        print("YOLO detection skipped (RUN_YOLO=False).")
    print(f"⏱ YOLO: {time.time()-t:.1f}s")

    if nge_object:
        save_object_data(nge_object)

    print(f"\n⏱ TOTAL RUN TIME: {time.time()-t_total:.1f}s")
    print("Data input process completed successfully.")