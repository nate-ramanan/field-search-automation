"""newgoogleearth2.py — Google Earth facility pipeline with QLever search.

This is a Streamlit re-packaging of ``NewGoogleEarthNew.py``. The ONLY thing
that changes versus the original is *how search results are obtained* and *how
the app is driven*:

* Search: the Overpass + Google Places search of ``NewGoogleEarthNew.py`` is
  fully removed and replaced with the QLever SPARQL (+ NCES CCD schools) search
  ported verbatim from ``qlever_v_6_json.py``.
* UI: the command-line ``input()`` prompts are removed. The app runs directly
  with ``streamlit run newgoogleearth2.py`` and mirrors the QLever frontend.
* Flow: instead of inserting straight into Postgres, the app shows a preview of
  every record that *would* be inserted and waits for explicit confirmation.

Everything after search — grouping, database insertion (``save_field_data``),
YOLO satellite recentering (``save_object_data``), the ``new_google_earth`` /
``nge_object`` schema, filenames, and Google Earth links — is copied byte-for-
byte from ``NewGoogleEarthNew.py`` and behaves identically. ``NewGoogleEarthNew.py``
itself is left untouched.

Run:
    pip install -r requirements.txt
    streamlit run newgoogleearth2.py
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import hashlib
import sqlite3
import logging
import traceback
from copy import deepcopy
from collections import Counter
from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("newgoogleearth2")

# The database connection pool is created at import time by ConnectionPool
# (reads config.ini [database]). Guard the import so a misconfigured config.ini
# surfaces a friendly message in the UI instead of a raw import traceback.
try:
    from ConnectionPool import pool  # noqa: E402
    _POOL_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - environment dependent
    pool = None  # type: ignore
    _POOL_ERROR = f"{type(_e).__name__}: {_e}"

# =============================================================================
# GOOGLE EARTH CONSTANTS  (copied verbatim from NewGoogleEarthNew.py)
# =============================================================================
# Google Maps API key — loaded from the environment, never hardcoded.
# Only needed when the satellite source is set to "Google Static Maps"; the
# default Esri provider needs no key. Set GOOGLE_MAPS_API_KEY in your env / .env.
KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '')
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')

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

# YOLO target classes for the satellite recentering stage (from NewGoogleEarthNew).
ALL_TARGET_CLASS_IDS = [4, 5, 9, 10, 14, 16]

# YOLO model was trained on Google Earth imagery captured at 650 m camera
# altitude and zoom level 18. Keep the satellite fetch (zoom) and the stored
# Google Earth link (camera distance) aligned to that training view.
GEARTH_ZOOM_LEVEL = 18          # Static Maps zoom for the YOLO input image
GEARTH_CAMERA_ALTITUDE_M = 650  # Google Earth web link camera distance (metres)
GEARTH_GROUND_ELEVATION_M = 4.1972381  # ground elevation term in the link


def build_gearth_link(lat: float, lon: float) -> str:
    """Build a Google Earth web link framed at the model's training altitude
    (650 m camera distance) so the linked view matches YOLO's training imagery.
    """
    return (f"https://earth.google.com/web/@{lat},{lon},"
            f"{GEARTH_GROUND_ELEVATION_M}a,{GEARTH_CAMERA_ALTITUDE_M}d")


# =============================================================================
# SATELLITE IMAGERY PROVIDER  (free Esri World Imagery vs paid Google)
# =============================================================================
# "esri"   -> Esri World Imagery XYZ tiles: FREE, no API key, sub-metre in US,
#             served at the same zoom 18 so YOLO's pixel->GPS offsets stay valid.
# "google" -> Google Static Maps: paid, requires KEY (original behaviour).
SATELLITE_PROVIDER_DEFAULT = "esri"
_ACTIVE_SATELLITE_PROVIDER = SATELLITE_PROVIDER_DEFAULT
_ESRI_TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}")
_TILE_SIZE = 256


def set_satellite_provider(provider: str) -> None:
    """Set the active satellite source ('esri' or 'google') process-wide."""
    global _ACTIVE_SATELLITE_PROVIDER
    _ACTIVE_SATELLITE_PROVIDER = provider if provider in ("esri", "google") \
        else SATELLITE_PROVIDER_DEFAULT

# Reasons recorded in nge_object.description when a field cannot be recentered by
# YOLO. Every new_google_earth row must be represented in nge_object; when YOLO
# yields zero usable detections we still insert a row explaining WHY.
# NOTE: nge_object.description is VARCHAR(64) — keep each reason <= 64 chars.
DESCRIPTION_MAX_LEN = 64
YOLO_ZERO_DETECTION_REASON = "YOLO found 0 target objects; kept original GPS"
IMAGE_UNAVAILABLE_REASON = "Satellite image unavailable; kept original GPS"
INVALID_GPS_REASON = "GPS missing or malformed; detection skipped"


@st.cache_resource(show_spinner=False)
def load_object_detection_model():
    """Load the YOLO object-detection model once per Streamlit process.

    Reads the model path from config.ini [model_paths] exactly as
    NewGoogleEarthNew.py does; cached so it is not reloaded on every rerun.
    """
    from ultralytics import YOLO  # imported lazily so the app starts fast
    config = ConfigParser()
    config.read(_CONFIG_FILE)
    obd_model_path = config.get('model_paths', 'obd_model')
    return YOLO(obd_model_path, verbose=False)


# =============================================================================
# QLEVER SEARCH — PORTED VERBATIM FROM qlever_v_6_json.py
# =============================================================================
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
DEFAULT_OVERPASS_URL = OVERPASS_MIRRORS[0]

NOMINATIM_URL = "https://nominatim.openstreetmap.org"

_CONTACT = os.environ.get("CONTACT_EMAIL", "")
if _CONTACT:
    USER_AGENT = f"SportsFacilityFinder/1.0 ({_CONTACT})"
else:
    USER_AGENT = "SportsFacilityFinder/1.0"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

_log_lock = Lock()

# Cap concurrent SPARQL POSTs to the public Qlever endpoint across all worker
# threads. The public instance throttles / rate-limits aggressively, so keep
# this at 1 (fully serial). _QLEVER_MIN_INTERVAL spaces consecutive POSTs.
_QLEVER_SEM = Semaphore(1)
_QLEVER_MIN_INTERVAL = 1.0  # seconds between consecutive Qlever POSTs
_qlever_last_call = [0.0]   # mutable holder guarded by _QLEVER_SEM

CACHE_DB_PATH = "facility_cache.db"
CACHE_TTL_SECONDS = 7 * 24 * 3600

_cache_lock = Lock()


def _init_cache():
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON api_cache(created_at)")


def _cache_key(prefix, *args):
    raw = prefix + "|" + "|".join(str(a) for a in args)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(key, max_age_seconds=None):
    ttl = max_age_seconds if max_age_seconds is not None else CACHE_TTL_SECONDS
    with _cache_lock:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            row = conn.execute(
                "SELECT value, created_at FROM api_cache WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return None
            value, created_at = row
            if time.time() - created_at > ttl:
                return None
            try:
                return json.loads(value)
            except Exception:
                return None


def cache_set(key, value):
    with _cache_lock:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), int(time.time())),
            )


def cache_clear():
    with _cache_lock:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            conn.execute("DELETE FROM api_cache")


def cache_stats():
    if not os.path.exists(CACHE_DB_PATH):
        return 0, 0
    with _cache_lock:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            count = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
    size = os.path.getsize(CACHE_DB_PATH)
    return count, size


_init_cache()

SPORTS_CONFIG = {
    "Soccer / Football": {
        "osm_sports": ["soccer", "football"],
        "keywords": ["soccer", "football", "futbol", "fútbol", "athletic field",
                     "sports field", "multi-purpose", "multipurpose"],
        "exclude": ["swim", "pool", "aqua", "skatepark", "golf", "bowling",
                    "tennis center", "library", "marina", "model airplane"],
        "facility_label": "Soccer Field",
        "label_variants": {
            "soccer_football": "Soccer/Football Field",
            "football_only": "Football Field",
            "multi": "Multi-Purpose Field (Soccer)",
        },
        "section_keywords": ["park", "field", "sports", "recreation"],
        "min_pitch_length_ft": 180,
    },
    "Baseball / Softball": {
        "osm_sports": ["baseball", "softball"],
        "keywords": ["baseball", "softball", "diamond", "little league",
                     "ball field", "ballfield", "tee ball", "t-ball"],
        "exclude": ["swim", "pool", "aqua", "skatepark", "golf", "bowling",
                    "tennis center", "library", "marina"],
        "facility_label": "Baseball Field",
        "label_variants": {
            "softball": "Softball Field",
            "both": "Baseball/Softball Field",
        },
        "section_keywords": ["park", "field", "diamond"],
        "min_pitch_length_ft": 200,
    },
    "Basketball": {
        "osm_sports": ["basketball"],
        "keywords": ["basketball", "gym", "recreation center", "rec center",
                     "community center", "boys & girls", "boys and girls",
                     "sports centre", "ymca"],
        "exclude": ["swim", "pool", "aqua", "skatepark", "golf", "bowling",
                    "marina", "library", "model airplane"],
        "facility_label": "Basketball Court",
        "label_variants": {
            "gym": "Gymnasium Basketball Court",
            "half": "Half Court",
            "full": "Full Court",
        },
        "section_keywords": ["park", "court", "gym", "recreation"],
        "min_pitch_length_ft": 42,
    },
    "Tennis": {
        "osm_sports": ["tennis"],
        "keywords": ["tennis", "racquet", "racket club"],
        "exclude": ["swim", "pool", "aqua", "skatepark", "golf", "bowling",
                    "marina", "library"],
        "facility_label": "Tennis Court",
        "label_variants": {},
        "section_keywords": ["park", "court", "tennis", "club"],
        "min_pitch_length_ft": 75,
    },
    "Volleyball": {
        "osm_sports": ["volleyball", "beachvolleyball"],
        "keywords": ["volleyball", "beach volleyball"],
        "exclude": ["swim", "pool", "aqua", "skatepark", "golf", "bowling",
                    "marina", "library"],
        "facility_label": "Volleyball Court",
        "label_variants": {
            "beach": "Beach Volleyball Court",
        },
        "section_keywords": ["park", "court", "beach", "gym"],
        "min_pitch_length_ft": 55,
    },
}

_TOO_SMALL_NAME_FRAGMENTS = [
    "tot lot", "tot-lot", "toddler", "mini park", "mini-park",
    "pocket park", "dog park", "dog run", "skate park", "skatepark",
    "splash pad", "spray park", "butterfly garden", "community garden",
    "meditation garden", "memorial garden", "rose garden",
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) *
         math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def clean_name(name):
    if not name:
        return ""
    return re.sub(r"\s+", " ", name).strip()


def normalize_key(name):
    n = name.lower().strip()
    for suffix in [" park", " field", " fields", " court", " courts"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)].strip()
    return re.sub(r"[^a-z0-9]", "", n)


def _photon_request(path, params, timeout=20, max_attempts=4):
    """Call Photon (Komoot) geocoder. Less throttled than Nominatim."""
    base = "https://photon.komoot.io"
    last_error = None
    for attempt in range(max_attempts):
        resp = None
        try:
            resp = requests.get(f"{base}{path}", params=params,
                                headers=HEADERS, timeout=timeout)
            if resp.status_code in (429, 503):
                last_error = f"Photon HTTP {resp.status_code}"
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json(), base
        except requests.exceptions.Timeout:
            last_error = f"Photon timeout (attempt {attempt+1})"
        except requests.exceptions.ConnectionError:
            last_error = f"Photon connection refused (attempt {attempt+1})"
        except requests.exceptions.HTTPError:
            status = resp.status_code if resp is not None else 0
            last_error = f"Photon HTTP {status}"
            break
        except Exception as e:
            last_error = f"Photon {type(e).__name__}: {e}"
            break
        if attempt < max_attempts - 1:
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Photon unreachable. Last error: {last_error}")


def lookup_city_bbox(city, county, state="California", country="USA", use_cache=True):
    """Resolve city via Photon. Returns bbox + OSM relation/way id for
    Qlever's spatial join (`ogc:sfContains`)."""
    key = _cache_key("city_bbox_qlever_v2", city, county, state, country)
    if use_cache:
        cached = cache_get(key)
        if cached is not None:
            return cached

    target_lower = city.lower().strip()
    target_state_lower = (state or "").lower().strip()

    params = [("q", city), ("limit", "10"),
              ("osm_tag", "place:city"), ("osm_tag", "place:town"),
              ("osm_tag", "place:village"), ("osm_tag", "boundary:administrative")]
    try:
        data, _ = _photon_request("/api/", params)
    except RuntimeError as e:
        st.error(f"Photon lookup failed: {e}")
        return None

    valid_item = None
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        name = (props.get("name") or "").lower()
        st_name = (props.get("state") or "").lower()
        if target_lower not in name and name not in target_lower:
            continue
        if target_state_lower and target_state_lower not in st_name:
            continue
        valid_item = feat
        break

    if not valid_item:
        return None

    props = valid_item.get("properties", {})
    geom = valid_item.get("geometry", {})
    coords = geom.get("coordinates", [])
    ext = props.get("extent")  # [west, north, east, south]
    if ext and len(ext) == 4:
        min_lon, max_lat, max_lon, min_lat = ext
    elif coords and len(coords) == 2:
        lon, lat = coords
        min_lat, max_lat = lat - 0.05, lat + 0.05
        min_lon, max_lon = lon - 0.05, lon + 0.05
    else:
        return None

    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    lat_buf = max(lat_span * 0.08, 0.005)
    lon_buf = max(lon_span * 0.08, 0.005)

    result = {
        "min_lat": min_lat - lat_buf,
        "max_lat": max_lat + lat_buf,
        "min_lon": min_lon - lon_buf,
        "max_lon": max_lon + lon_buf,
        "match_display": props.get("name", ""),
        "osm_type": props.get("osm_type", ""),   # "R" | "W" | "N"
        "osm_id": props.get("osm_id", 0),
    }
    if use_cache:
        cache_set(key, result)
    return result


QLEVER_ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/osm-planet"

SPARQL_PREFIXES = """
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX ogc: <http://www.opengis.net/rdf#>
PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX osm: <https://www.openstreetmap.org/>
PREFIX osmrel: <https://www.openstreetmap.org/relation/>
PREFIX osmway: <https://www.openstreetmap.org/way/>
PREFIX osmnode: <https://www.openstreetmap.org/node/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

_OSM_TYPE_PREFIX = {"R": "osmrel:", "W": "osmway:", "N": "osmnode:"}


def _city_uri(bbox):
    """Build the SPARQL term referring to the city as an OSM entity."""
    t = bbox.get("osm_type") or "R"
    oid = bbox.get("osm_id")
    if not oid:
        return None
    return f"{_OSM_TYPE_PREFIX.get(t, 'osmrel:')}{oid}"


_INDOOR_SPORTS = {"Basketball", "Volleyball"}


def build_qlever_queries(bbox, sport_config, sport_choice=""):
    """SPARQL queries with sport-aware container filtering."""
    city = _city_uri(bbox)
    sports_filter = " || ".join(
        [f'CONTAINS(LCASE(STR(?sport)), "{s.lower()}")'
         for s in sport_config["osm_sports"]]
    )
    spatial = f"{city} ogc:sfContains ?osm_id ." if city else ""

    # Pitches: sport tag is OPTIONAL so untagged / unnamed pitches are still
    # returned. Match the requested sport when a sport tag exists, but also let
    # through pitches that carry NO sport tag at all (very common in OSM) — this
    # is the main lever for finding "every field even unnamed".
    pitches = f"""{SPARQL_PREFIXES}
SELECT ?osm_id ?name ?sport ?lit ?hoops ?wkt WHERE {{
  {spatial}
  ?osm_id osmkey:leisure "pitch" ;
          geo:hasGeometry/geo:asWKT ?wkt .
  OPTIONAL {{ ?osm_id osmkey:sport ?sport }}
  OPTIONAL {{ ?osm_id osmkey:name ?name }}
  OPTIONAL {{ ?osm_id osmkey:lit ?lit }}
  OPTIONAL {{ ?osm_id osmkey:hoops ?hoops }}
  FILTER(!BOUND(?sport) || ({sports_filter}))
}}
LIMIT 5000"""

    parks = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?osm_id ?name ?wkt WHERE {{
  {spatial}
  ?osm_id osmkey:leisure "park" ;
          osmkey:name ?name ;
          geo:hasGeometry/geo:asWKT ?wkt .
}}
LIMIT 2000"""

    schools = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?osm_id ?name ?amenity ?wkt WHERE {{
  {spatial}
  ?osm_id osmkey:amenity ?amenity ;
          osmkey:name ?name ;
          geo:hasGeometry/geo:asWKT ?wkt .
  FILTER(?amenity IN ("school", "college", "university"))
}}
LIMIT 2000"""

    sports_centres = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?osm_id ?name ?leisure ?wkt WHERE {{
  {spatial}
  ?osm_id osmkey:leisure ?leisure ;
          osmkey:name ?name ;
          geo:hasGeometry/geo:asWKT ?wkt .
  FILTER(?leisure IN ("sports_centre", "fitness_centre", "stadium"))
}}
LIMIT 1000"""

    rec_grounds = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?osm_id ?name ?wkt WHERE {{
  {spatial}
  ?osm_id osmkey:landuse "recreation_ground" ;
          osmkey:name ?name ;
          geo:hasGeometry/geo:asWKT ?wkt .
}}
LIMIT 1000"""

    queries = {
        f"{sport_config['facility_label']}s": ("pitch", pitches),
        "Parks": ("park", parks),
        "Schools": ("school", schools),
        "Sports centres + stadiums": ("sports_centre", sports_centres),
        "Recreation grounds": ("park", rec_grounds),
    }

    if sport_choice in _INDOOR_SPORTS:
        indoor_query = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?osm_id ?name ?leisure ?amenity ?building ?wkt WHERE {{
  {spatial}
  ?osm_id osmkey:name ?name ;
          geo:hasGeometry/geo:asWKT ?wkt .
  OPTIONAL {{ ?osm_id osmkey:leisure ?leisure }}
  OPTIONAL {{ ?osm_id osmkey:amenity ?amenity }}
  OPTIONAL {{ ?osm_id osmkey:building ?building }}
  FILTER(
    ?leisure IN ("sports_centre", "fitness_centre", "sports_hall") ||
    ?amenity IN ("community_centre", "gym") ||
    ?building IN ("sports_hall", "gymnasium")
  )
}}
LIMIT 1500"""
        queries["Indoor gyms / community centres"] = ("sports_centre", indoor_query)

    return queries


def _parse_wkt_point(wkt):
    """Return (lat, lon) from a WKT geometry. Centroid for polygons."""
    if not wkt:
        return None, None
    s = wkt.strip()
    if s.startswith("POINT"):
        m = re.search(r"POINT\s*\(\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", s)
        if m:
            return float(m.group(2)), float(m.group(1))
    nums = re.findall(r"-?\d+\.\d+", s)
    if len(nums) >= 2:
        lons = [float(x) for x in nums[0::2]]
        lats = [float(y) for y in nums[1::2]]
        return sum(lats)/len(lats), sum(lons)/len(lons)
    return None, None


def query_qlever(name, query, status_callback, use_cache=True, timeout=90):
    """POST SPARQL to Qlever. Returns (name, list-of-row-dicts, raw_payload)."""
    if use_cache:
        k = _cache_key("qlever", QLEVER_ENDPOINT, query)
        cached = cache_get(k)
        if cached is not None:
            with _log_lock:
                status_callback(f"  [{name}] cached ({len(cached)} rows)")
            return name, cached, {"cached": True, "row_count": len(cached)}

    headers = {**HEADERS, "Accept": "application/sparql-results+json"}
    try:
        with _log_lock:
            status_callback(f"  [{name}] querying Qlever...")
        with _QLEVER_SEM:
            # Throttle: keep at least _QLEVER_MIN_INTERVAL between POSTs.
            wait = _QLEVER_MIN_INTERVAL - (time.time() - _qlever_last_call[0])
            if wait > 0:
                time.sleep(wait)
            resp = requests.post(QLEVER_ENDPOINT, data={"query": query},
                                  headers=headers, timeout=timeout)
            _qlever_last_call[0] = time.time()
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("results", {}).get("bindings", [])
        flat = []
        for r in rows:
            flat.append({k: v.get("value", "") for k, v in r.items()})
        with _log_lock:
            status_callback(f"  [{name}] OK {len(flat)} rows")
        if use_cache and flat:
            cache_set(_cache_key("qlever", QLEVER_ENDPOINT, query), flat)
        return name, flat, payload
    except Exception as e:
        with _log_lock:
            status_callback(f"  [{name}] ERR {type(e).__name__}: {e}")
        return name, [], {"error": f"{type(e).__name__}: {e}"}


def _resolve_sport_choice(sport_config):
    """Recover the SPORTS_CONFIG key from a sport_config dict."""
    for k, v in SPORTS_CONFIG.items():
        if v is sport_config:
            return k
    return ""


def fetch_overpass(bbox, sport_config, overpass_url, status_callback,
                   is_local=False, use_cache=True):
    """Fetch facilities via Qlever SPARQL (Overpass replacement)."""
    status_callback("Source 1: Qlever SPARQL (OSM-planet)...")
    sport_choice = _resolve_sport_choice(sport_config)
    queries = build_qlever_queries(bbox, sport_config, sport_choice)
    results = []
    raw_payloads = {}
    with ThreadPoolExecutor(max_workers=1) as ex:
        futures = {
            ex.submit(query_qlever, qname, qtext, status_callback, use_cache):
                (qname, kind)
            for qname, (kind, qtext) in queries.items()
        }
        for fut in as_completed(futures):
            qname, kind = futures[fut]
            try:
                _, rows, raw_payload = fut.result()
                raw_payloads[qname] = raw_payload
                for r in rows:
                    lat, lon = _parse_wkt_point(r.get("wkt", ""))
                    if lat is None:
                        continue
                    leisure = "pitch" if kind == "pitch" else (
                        "park" if kind == "park" else (
                            r.get("leisure", "") if kind == "sports_centre" else ""
                        )
                    )
                    amenity = r.get("amenity", "") if kind == "school" else ""
                    results.append({
                        "source": "qlever",
                        "name": clean_name(r.get("name", "")),
                        "lat": lat, "lon": lon,
                        "sport": r.get("sport", ""),
                        "leisure": leisure,
                        "amenity": amenity,
                        "building": "",
                        "tags": {
                            "lit": r.get("lit", ""),
                            "hoops": r.get("hoops", ""),
                        },
                        "length_ft": None,
                        "width_ft": None,
                    })
            except Exception as e:
                status_callback(f"  Worker error: {e}")

    status_callback(f"  Qlever total: {len(results)} raw elements")
    return results, raw_payloads


def fetch_nominatim(city, state, bbox, sport_config, status_callback, use_cache=True):
    """No-op in Qlever mode — Qlever already returns named facilities."""
    status_callback("Source 2: Nominatim — skipped (Qlever provides names).")
    return []


NCES_API_BASE = "https://educationdata.urban.org/api/v1/schools/ccd/directory"
NCES_YEAR = 2020
NCES_CACHE_TTL_SECONDS = 30 * 24 * 3600

_US_STATE_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}


def _state_to_abbr(state):
    s = (state or "").strip()
    if len(s) == 2:
        return s.upper()
    return _US_STATE_ABBR.get(s.lower(), s[:2].upper())


def _fetch_nces_state_raw(state_abbr, use_cache=True):
    """Fetch every public school in a state (paginated). 30-day cache."""
    key = _cache_key("nces_state_v1", NCES_YEAR, state_abbr)
    if use_cache:
        cached = cache_get(key, max_age_seconds=NCES_CACHE_TTL_SECONDS)
        if cached is not None:
            return cached

    all_results = []
    url = f"{NCES_API_BASE}/{NCES_YEAR}/"
    params = {"state_location": state_abbr}
    for _page in range(20):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=90)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break
        all_results.extend(data.get("results", []))
        nxt = data.get("next")
        if not nxt:
            break
        url = nxt
        params = None

    if use_cache and all_results:
        cache_set(_cache_key("nces_state_v1", NCES_YEAR, state_abbr), all_results)
    return all_results


def _nces_level_to_category(lvl):
    try:
        lvl = int(lvl)
    except (TypeError, ValueError):
        return "OTHER FACILITIES"
    return {
        1: "ELEMENTARY SCHOOLS",
        2: "MIDDLE SCHOOLS",
        3: "HIGH SCHOOLS",
        4: "OTHER FACILITIES",
    }.get(lvl, "OTHER FACILITIES")


def fetch_nces_schools(target_city, target_state, bbox, status_callback,
                       use_cache=True):
    """Fetch NCES public schools in target_city; returns entries in the same
    schema Qlever produces so downstream merge/geocode is unchanged."""
    state_abbr = _state_to_abbr(target_state)
    status_callback(f"Source 2: NCES CCD schools ({state_abbr})...")
    all_rows = _fetch_nces_state_raw(state_abbr, use_cache=use_cache)
    if not all_rows:
        status_callback(f"  NCES: 0 rows for {state_abbr}")
        return []

    target = (target_city or "").strip().lower()
    matched = []
    for r in all_rows:
        loc = (r.get("city_location") or "").strip().lower()
        lat = r.get("latitude")
        lon = r.get("longitude")
        try:
            lat = float(lat) if lat is not None else None
            lon = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            lat = lon = None
        if lat is None or lon is None:
            continue
        in_bbox_ok = (bbox and
                      bbox["min_lat"] <= lat <= bbox["max_lat"] and
                      bbox["min_lon"] <= lon <= bbox["max_lon"])
        if loc != target and not in_bbox_ok:
            continue

        name = (r.get("school_name") or "").strip()
        if not name:
            continue
        matched.append({
            "source": "nces",
            "name": clean_name(name),
            "lat": lat, "lon": lon,
            "sport": "",
            "leisure": "",
            "amenity": "school",
            "building": "",
            "tags": {
                "addr:city": r.get("city_location", ""),
                "nces_level": r.get("school_level", ""),
            },
            "nces_level": r.get("school_level", ""),
            "length_ft": None,
            "width_ft": None,
            "_nces_raw": r,
        })
    status_callback(f"  NCES matched: {len(matched)} schools in {target_city}")
    return matched


def is_confirmed_sport(entry, sport_config):
    sport = entry.get("sport", "").lower()
    return any(s in sport for s in sport_config["osm_sports"])


def is_facility(entry):
    leisure = entry.get("leisure", "").lower()
    amenity = entry.get("amenity", "").lower()
    building = entry.get("building", "").lower()
    name = entry.get("name", "").lower()
    return (leisure in ("park", "sports_centre", "fitness_centre",
                        "sports_hall", "stadium", "school", "college",
                        "university", "recreation_ground", "playground") or
            amenity in ("school", "college", "university",
                        "community_centre", "leisure", "amenity") or
            building in ("sports_hall",) or
            entry.get("source") == "nominatim" or
            any(k in name for k in ["park", "school", "college",
                                     "recreation", "field", "playground"]))


def is_too_small(entry, sport_config):
    name = entry.get("name", "").lower()
    if any(frag in name for frag in _TOO_SMALL_NAME_FRAGMENTS):
        return True

    leisure = entry.get("leisure", "").lower()
    is_confirmed = is_confirmed_sport(entry, sport_config)
    if leisure == "playground" and not is_confirmed:
        return True

    length_ft = entry.get("length_ft")
    if length_ft:
        min_ft = sport_config.get("min_pitch_length_ft", 0)
        if length_ft < min_ft:
            return True

    children = entry.get("child_pitches", [])
    if children:
        min_ft = sport_config.get("min_pitch_length_ft", 0)
        all_small = all(
            (c.get("length_ft") or 0) > 0 and (c.get("length_ft") or 0) < min_ft
            for c in children
        )
        if all_small:
            return True

    return False


def merge_and_deduplicate(all_sources, sport_config, status_callback):
    status_callback(f"Total raw entries: {len(all_sources)}")

    coord_seen = set()
    deduped = []
    for entry in all_sources:
        if entry.get("lat") and entry.get("lon"):
            ck = (round(entry["lat"], 5), round(entry["lon"], 5))
            if ck in coord_seen:
                continue
            coord_seen.add(ck)
        deduped.append(entry)
    status_callback(f"After coord dedup: {len(deduped)}")

    confirmed = []
    facilities = []
    untagged_pitches = 0
    for entry in deduped:
        if entry.get("leisure") == "pitch":
            # Accept pitches confirmed for this sport AND sport-untagged pitches
            # (no OSM sport tag) so unnamed / unclassified fields are kept.
            if is_confirmed_sport(entry, sport_config):
                confirmed.append(entry)
            elif not entry.get("sport", "").strip():
                untagged_pitches += 1
                confirmed.append(entry)
            continue
        if is_facility(entry):
            facilities.append(entry)

    status_callback(f"Confirmed pitches: {len(confirmed)} "
                    f"({untagged_pitches} sport-untagged included)")
    status_callback(f"Facilities: {len(facilities)}")

    fac_seen = {}
    exclude_list = sport_config["exclude"]
    for entry in facilities:
        name = entry["name"].strip()
        if not name:
            continue
        if any(k in name.lower() for k in exclude_list):
            continue
        key = normalize_key(name)
        if not key:
            continue
        if key not in fac_seen:
            fac_seen[key] = entry
        else:
            existing = fac_seen[key]
            if not existing.get("lat") and entry.get("lat"):
                entry["name"] = entry["name"] or existing["name"]
                fac_seen[key] = entry

    facility_list = list(fac_seen.values())

    _SCHOOL_PRIORITY = {
        "high school": 0, "high sch": 0, "preparatory": 0, "prep school": 0,
        "middle school": 1, "middle sch": 1, "junior high": 1, "intermediate": 1,
        "elementary": 2, "primary school": 2,
        "college": 3, "university": 3,
    }

    def _school_priority(name):
        n = name.lower()
        for kw, rank in _SCHOOL_PRIORITY.items():
            if kw in n:
                return rank
        return 99

    def _institution_stem(name):
        n = name.lower()
        for kw in ["high school", "middle school", "junior high", "elementary school",
                   "elementary", "primary school", "preparatory", "prep school",
                   "college", "university", "intermediate school", "intermediate"]:
            n = n.replace(kw, "").strip()
        return re.sub(r"[^a-z0-9]", "", n)

    SAME_CAMPUS_RADIUS = 60
    suppressed = set()
    fl = facility_list
    for i in range(len(fl)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(fl)):
            if j in suppressed:
                continue
            a, b = fl[i], fl[j]
            if not (a.get("lat") and a.get("lon") and b.get("lat") and b.get("lon")):
                continue
            dist = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if dist > SAME_CAMPUS_RADIUS:
                continue
            stem_a = _institution_stem(a.get("name", ""))
            stem_b = _institution_stem(b.get("name", ""))
            if not stem_a or not stem_b or stem_a != stem_b:
                continue
            pri_a = _school_priority(a.get("name", ""))
            pri_b = _school_priority(b.get("name", ""))
            if pri_a == 99 and pri_b == 99:
                continue
            if pri_a <= pri_b:
                suppressed.add(j)
            else:
                suppressed.add(i)

    if suppressed:
        status_callback(f"  Same-campus dedup removed {len(suppressed)} co-located duplicate(s)")
        facility_list = [fl[i] for i in range(len(fl)) if i not in suppressed]

    PROXIMITY_RADIUS = 500
    for pitch in confirmed:
        if not pitch.get("lat") or not pitch.get("lon"):
            continue
        best_fac = None
        best_dist = PROXIMITY_RADIUS + 1
        for fac in facility_list:
            if not fac.get("lat") or not fac.get("lon"):
                continue
            dist = haversine(pitch["lat"], pitch["lon"], fac["lat"], fac["lon"])
            if dist < best_dist:
                best_dist = dist
                best_fac = fac
        if best_fac:
            if "child_pitches" not in best_fac:
                best_fac["child_pitches"] = []
            best_fac["child_pitches"].append(pitch)
        else:
            name = pitch.get("name", "")
            if name:
                key = normalize_key(name)
                if key and key not in fac_seen:
                    fac_seen[key] = pitch
                    facility_list.append(pitch)
            elif pitch.get("lat"):
                pitch["name"] = f"{sport_config['facility_label']} ({pitch['lat']:.4f}, {pitch['lon']:.4f})"
                facility_list.append(pitch)

    results = []
    for fac in facility_list:
        has_pitches = len(fac.get("child_pitches", [])) > 0
        name_lower = fac.get("name", "").lower()
        is_sport_name = any(k in name_lower for k in sport_config["keywords"])
        is_nces = fac.get("source") == "nces"
        if has_pitches or is_sport_name or is_nces:
            results.append(fac)

    results = [r for r in results if r.get("name")]

    before_size = len(results)
    results = [r for r in results if not is_too_small(r, sport_config)]
    removed_small = before_size - len(results)
    if removed_small:
        status_callback(f"  Removed {removed_small} too-small / wrong-type entries "
                        f"(tot lots, playgrounds, undersized pitches)")

    multi = sum(1 for r in results if len(r.get("child_pitches", [])) > 1)
    status_callback(f"After merge: {len(results)} facilities ({multi} multi-court/field)")
    return results


def _fallback_address(target_city, target_state="", postcode="", lat=None, lon=None):
    state_abbr = ""
    if target_state:
        s = target_state.strip()
        state_abbr = s[:2].upper() if len(s) >= 2 else s.upper()
    parts = [p for p in [target_city, state_abbr, postcode] if p]
    out = ", ".join([parts[0]] + ([" ".join(parts[1:])] if len(parts) > 1 else []))
    if lat is not None and lon is not None:
        out = f"{out} (@ {lat:.4f}, {lon:.4f})".strip()
    return out


def _reverse_geocode_one(entry, target_city, nominatim_url=None,
                         use_cache=True, target_state="", retries=2):
    """Reverse geocode via Photon."""
    tags = entry.get("tags", {})
    street = tags.get("addr:street", "")
    number = tags.get("addr:housenumber", "")
    if street:
        city = tags.get("addr:city", target_city)
        state_tag = tags.get("addr:state", "")
        postcode_tag = tags.get("addr:postcode", "")
        state_abbr = state_tag[:2].upper() if state_tag else (
            target_state[:2].upper() if target_state else "")
        street_line = f"{number} {street}".strip()
        loc_line = f"{city}, {state_abbr} {postcode_tag}".strip().rstrip(",")
        entry["address"] = f"{street_line}, {loc_line}".strip(", ")
        entry["verified_city"] = city.lower()
        entry["zipcode"] = postcode_tag
        return entry, "osm_tags"

    lat_r = round(entry["lat"], 5)
    lon_r = round(entry["lon"], 5)
    if use_cache:
        key = _cache_key("photon_reverse_v1", lat_r, lon_r)
        cached = cache_get(key)
        if cached is not None:
            entry["address"] = cached.get("address") or _fallback_address(
                target_city, target_state, lat=entry["lat"], lon=entry["lon"])
            entry["verified_city"] = cached.get("verified_city", "")
            entry["zipcode"] = cached.get("zipcode", "")
            return entry, "cached"

    last_err = None
    for attempt in range(retries + 1):
        try:
            params = {"lat": entry["lat"], "lon": entry["lon"], "lang": "en"}
            data, _ = _photon_request("/reverse", params, timeout=20)
            feats = data.get("features", [])
            if not feats:
                raise RuntimeError("no features")
            props = feats[0].get("properties", {})
            road = props.get("street", "") or props.get("name", "")
            house = props.get("housenumber", "")
            city = props.get("city", "") or props.get("town", "") or props.get("village", "")
            postcode = props.get("postcode", "")
            state = props.get("state", "")
            entry["verified_city"] = city.lower() if city else ""
            entry["zipcode"] = postcode
            display_city = city if city else target_city
            state_abbr = state[:2].upper() if state else (
                target_state[:2].upper() if target_state else "")
            if road:
                street_line = f"{house} {road}".strip()
                loc_line = f"{display_city}, {state_abbr} {postcode}".strip().rstrip(",")
                entry["address"] = f"{street_line}, {loc_line}".strip(", ")
            else:
                entry["address"] = _fallback_address(
                    display_city, target_state, postcode,
                    lat=entry["lat"], lon=entry["lon"])
            if use_cache:
                cache_set(_cache_key("photon_reverse_v1", lat_r, lon_r), {
                    "address": entry["address"],
                    "verified_city": entry["verified_city"],
                    "zipcode": entry["zipcode"],
                })
            return entry, "api"
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
            continue
    entry["address"] = _fallback_address(
        target_city, target_state, lat=entry["lat"], lon=entry["lon"])
    entry["verified_city"] = ""
    entry["zipcode"] = ""
    return entry, "api_failed"


def reverse_geocode_all(entries, target_city, status_callback,
                        use_local_nominatim=False, use_cache=True,
                        target_state=""):
    nominatim_url = NOMINATIM_URL
    n = len(entries)
    status_callback(f"Reverse geocoding {n} facilities...")

    osm_count = 0
    cache_count = 0
    api_needed = []
    for entry in entries:
        tags = entry.get("tags", {})
        if tags.get("addr:street"):
            _reverse_geocode_one(entry, target_city, nominatim_url, use_cache,
                                 target_state=target_state)
            osm_count += 1
            continue
        if use_cache:
            lat_r = round(entry["lat"], 5)
            lon_r = round(entry["lon"], 5)
            key = _cache_key("reverse_geocode_v2", lat_r, lon_r)
            cached = cache_get(key)
            if cached is not None:
                entry["address"] = cached.get("address") or _fallback_address(
                    target_city, target_state, lat=entry["lat"], lon=entry["lon"])
                entry["verified_city"] = cached.get("verified_city", "")
                entry["zipcode"] = cached.get("zipcode", "")
                cache_count += 1
                continue
        api_needed.append(entry)

    if osm_count:
        status_callback(f"  Used OSM addr tags: {osm_count}")
    if cache_count:
        status_callback(f"  Used cache: {cache_count} 💾")

    if not api_needed:
        status_callback(f"  All {n} addresses resolved without API calls")
        return

    if use_local_nominatim:
        status_callback(f"  Parallel reverse geocode {len(api_needed)} (local)...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(
                lambda e: _reverse_geocode_one(e, target_city, nominatim_url,
                                               use_cache, target_state=target_state),
                api_needed
            ))
    else:
        status_callback(f"  Sequential reverse geocode {len(api_needed)} "
                        f"(public Nominatim, 1 req/sec)...")
        for i, entry in enumerate(api_needed, 1):
            _reverse_geocode_one(entry, target_city, nominatim_url, use_cache,
                                 target_state=target_state)
            if i % 10 == 0:
                status_callback(f"    progress: {i}/{len(api_needed)}")
            time.sleep(1.1)

    status_callback(f"  Reverse geocoding complete")


def get_city_neighborhoods(city, state, country="USA", use_cache=True):
    """Simplified in Qlever/Photon mode — return just the city alias set."""
    return {city.lower()}


def _parse_city_from_address(address):
    if not address:
        return ""

    parts = [p.strip() for p in address.split(",")]
    if len(parts) < 2:
        return ""

    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^\d{5}(-\d{4})?$", part):
            continue
        if re.match(r"^[A-Z]{2}(\s+\d{5}(-\d{4})?)?$", part):
            continue
        if re.match(r"^\d+$", part):
            continue
        if part.lower() in ("usa", "united states", "us"):
            continue
        city_token = re.split(r"\s+[A-Z]{2}\s+\d{5}", part)[0].strip()
        if city_token:
            return city_token.lower()

    return ""


def _parse_zip_from_address(address):
    if not address:
        return ""
    m = re.search(r"\b(\d{5}(?:-\d{4})?)\b", address)
    return m.group(1) if m else ""


def filter_wrong_city(entries, target_city, target_state, bbox, status_callback,
                      use_cache=True):
    status_callback(f"Filtering facilities outside {target_city}...")
    target = target_city.lower().strip()

    valid_aliases = get_city_neighborhoods(target_city, target_state,
                                            use_cache=use_cache)

    if len(valid_aliases) > 1:
        sample = ", ".join(sorted(valid_aliases))[:80]
        status_callback(f"  Recognizing {len(valid_aliases)} aliases: {sample}")

    filtered = []
    removed = []

    for entry in entries:
        v_city = entry.get("verified_city", "").lower().strip()
        address = entry.get("address", "")

        address_city = _parse_city_from_address(address)

        if address_city:
            if address_city in valid_aliases or target in address_city or address_city in target:
                filtered.append(entry)
                continue
            else:
                removed.append(
                    f"{entry['name']} (address city: '{address_city}' ≠ '{target}')"
                )
                continue

        if v_city:
            if v_city in valid_aliases or target in v_city or v_city in target:
                filtered.append(entry)
            else:
                removed.append(f"{entry['name']} (verified city: '{v_city}' ≠ '{target}')")
            continue

        filtered.append(entry)

    status_callback(f"Removed {len(removed)} facilities outside {target_city}")
    status_callback(f"Kept {len(filtered)} facilities")
    return filtered, removed


def categorize(entries, sport_config, status_callback):
    exclude = sport_config["exclude"]
    entries = [e for e in entries if not any(k in e["name"].lower() for k in exclude)]

    categories = {
        "PUBLIC PARKS & RECREATION": [],
        "GYMNASIUM / INDOOR FACILITIES": [],
        "HIGH SCHOOLS": [],
        "MIDDLE SCHOOLS": [],
        "ELEMENTARY SCHOOLS": [],
        "COLLEGE": [],
        "OTHER FACILITIES": [],
    }

    for entry in entries:
        if entry.get("source") == "nces" and entry.get("nces_level") not in (None, ""):
            categories[_nces_level_to_category(entry.get("nces_level"))].append(entry)
            continue

        combined = (entry["name"] + " " + entry.get("address", "")).lower()
        if any(k in combined for k in ["high school", "high sch", "preparatory", "prep school"]):
            categories["HIGH SCHOOLS"].append(entry)
        elif any(k in combined for k in ["middle school", "middle sch", "junior high", "intermediate"]):
            categories["MIDDLE SCHOOLS"].append(entry)
        elif any(k in combined for k in ["elementary", "primary school"]):
            categories["ELEMENTARY SCHOOLS"].append(entry)
        elif any(k in combined for k in ["college", "university"]):
            categories["COLLEGE"].append(entry)
        elif any(k in combined for k in ["gym", "recreation center", "rec center",
                                          "community center", "boys & girls",
                                          "boys and girls", "sports centre",
                                          "sports center", "fitness", "indoor",
                                          "ymca"]):
            categories["GYMNASIUM / INDOOR FACILITIES"].append(entry)
        elif any(k in combined for k in ["park", "field", "memorial", "playground", "recreation"]):
            categories["PUBLIC PARKS & RECREATION"].append(entry)
        else:
            categories["OTHER FACILITIES"].append(entry)

    return categories


def _run_single_job(city, county, state, sport_choice, overpass_url, use_cache,
                    log_lock, status_callback):
    """Run the full QLever search pipeline for one (city, sport). Returns
    (sport_choice, categories_dict_or_None, total, error_or_None)."""
    sport_config = SPORTS_CONFIG[sport_choice]
    job_tag = f"[{city}/{sport_choice}]"

    def _log(msg):
        with log_lock:
            status_callback(msg)

    try:
        bbox = lookup_city_bbox(city, county, state, "United States",
                                use_cache=use_cache)
        if not bbox:
            return sport_choice, None, 0, f"{job_tag} bbox lookup failed"

        is_local = ("localhost" in overpass_url or "127.0.0.1" in overpass_url)
        op_results, _raw_qlever = fetch_overpass(bbox, sport_config, overpass_url, _log,
                                                 is_local=is_local, use_cache=use_cache)
        nm_results = fetch_nominatim(city, state, bbox, sport_config, _log,
                                     use_cache=use_cache)
        nces_results = fetch_nces_schools(city, state, bbox, _log,
                                          use_cache=use_cache)

        merged = merge_and_deduplicate(
            op_results + nm_results + nces_results, sport_config, _log)
        if not merged:
            _log(f"{job_tag} 0 facilities")
            return sport_choice, {}, 0, None

        reverse_geocode_all(merged, city, _log, use_local_nominatim=False,
                            use_cache=use_cache, target_state=state)
        filtered, _ = filter_wrong_city(merged, city, state, bbox, _log,
                                        use_cache=use_cache)
        if not filtered:
            _log(f"{job_tag} 0 after city filter")
            return sport_choice, {}, 0, None

        categories = categorize(filtered, sport_config, _log)
        total = sum(len(v) for v in categories.values())
        _log(f"{job_tag} ✅ {total} facilities")
        return sport_choice, categories, total, None
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        with log_lock:
            status_callback(f"{job_tag} ❌ {type(e).__name__}: {e}")
        return sport_choice, None, 0, f"{job_tag} {e}\n{tb}"


def run_city_search(city, county, state, sports_selected, overpass_url, use_cache,
                    max_workers, status_callback):
    """Run the QLever search for one city across the selected sports in parallel.

    Returns ({sport_choice: (categories, total)}, errors) — the same per-sport
    structure qlever_v_6_json.py produces, minus the Excel/JSON export layer we
    don't need here.
    """
    log_lock = Lock()
    results: Dict[str, Tuple[dict, int]] = {}
    errors: List[str] = []

    status_callback(f"🚀 Searching {city} across {len(sports_selected)} sport(s) "
                    f"on {max_workers} worker(s)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_single_job, city, county, state, sport, overpass_url,
                      use_cache, log_lock, status_callback): sport
            for sport in sports_selected
        }
        done = 0
        for fut in as_completed(futures):
            sport = futures[fut]
            try:
                sp, cats, total, err = fut.result()
                if err:
                    errors.append(err)
                if cats:
                    results[sp] = (cats, total)
            except Exception as e:
                errors.append(f"[{city}/{sport}] worker crashed: {e}")
            done += 1
            with log_lock:
                status_callback(f"  Progress: {done}/{len(sports_selected)}  "
                                f"({time.time() - t0:.1f}s elapsed)")
    return results, errors


# =============================================================================
# GOOGLE EARTH PIPELINE — COPIED VERBATIM FROM NewGoogleEarthNew.py
# (satellite image, grouping, database insertion, YOLO recentering)
# =============================================================================
def getImage(lat, lon, key, zoom, width, height):
    if lat != 'Error' and lon != 'Error':
        return f"https://maps.googleapis.com/maps/api/staticmap?key={key}&center={lat},{lon}&zoom={zoom}&size={width}x{height}&maptype=satellite"
    return 'Error'


def _latlon_to_global_px(lat, lon, zoom):
    """Web-Mercator lat/lon -> global pixel coords at a given zoom (256px tiles)."""
    n = (2 ** zoom) * _TILE_SIZE
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _fetch_esri_image_array(lat, lon, zoom, width, height):
    """Fetch a Google-Static-Maps-equivalent image from FREE Esri World Imagery.

    Downloads the zoom-`zoom` XYZ tiles covering a `width`x`height` window
    centred on (lat, lon) and stitches them into one RGB array. Same zoom as the
    Google path, so the ground scale (metres/pixel) — and therefore YOLO's
    pixel->GPS offset factors — are identical.
    """
    cx, cy = _latlon_to_global_px(lat, lon, zoom)
    left = cx - width / 2.0
    top = cy - height / 2.0

    tx0, tx1 = int(left // _TILE_SIZE), int((left + width - 1) // _TILE_SIZE)
    ty0, ty1 = int(top // _TILE_SIZE), int((top + height - 1) // _TILE_SIZE)
    max_tile = 2 ** zoom - 1

    def _get_tile(tx, ty):
        if tx < 0 or ty < 0 or tx > max_tile or ty > max_tile:
            return tx, ty, None
        url = _ESRI_TILE_URL.format(z=zoom, y=ty, x=tx)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return tx, ty, Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return tx, ty, None

    coords = [(tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
    canvas = Image.new("RGB",
                       ((tx1 - tx0 + 1) * _TILE_SIZE, (ty1 - ty0 + 1) * _TILE_SIZE))
    got_any = False
    with ThreadPoolExecutor(max_workers=8) as ex:
        for tx, ty, tile in ex.map(lambda c: _get_tile(*c), coords):
            if tile is None:
                continue
            got_any = True
            canvas.paste(tile, ((tx - tx0) * _TILE_SIZE, (ty - ty0) * _TILE_SIZE))
    if not got_any:
        return None

    crop_x = int(round(left - tx0 * _TILE_SIZE))
    crop_y = int(round(top - ty0 * _TILE_SIZE))
    cropped = canvas.crop((crop_x, crop_y, crop_x + width, crop_y + height))
    return np.array(cropped)


def get_satellite_image_array(gps_location, zoom_level=GEARTH_ZOOM_LEVEL,
                              size=(800, 850), provider=None):
    """Return an RGB satellite image array centred on the given GPS point.

    Provider defaults to the process-wide selection (Esri = free, Google = paid).
    Return shape/scale is identical for both so YOLO behaviour is unchanged.
    """
    lat = gps_location['latitude']
    lon = gps_location['longitude']
    provider = provider or _ACTIVE_SATELLITE_PROVIDER

    if provider == "esri":
        try:
            return _fetch_esri_image_array(lat, lon, zoom_level, size[0], size[1])
        except Exception as e:
            print(f"Error fetching Esri satellite image: {e}")
            return None

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


def get_base_name(name):
    """Strips trailing parenthesis sports lists for raw facility name."""
    if not name:
        return ""
    return re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()


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
            # match['number_of_fields'] += 1
        else:
            f_copy = f.copy()
            f_copy['sports'] = {sport_name}
            f_copy['coords_list'] = [(lat, lon)]
            # f_copy['number_of_fields'] = 1
            grouped.append(f_copy)

    for g in grouped:
        lats = [c[0] for c in g['coords_list']]
        lons = [c[1] for c in g['coords_list']]
        g['original_gps_location']['latitude'] = sum(lats) / len(lats)
        g['original_gps_location']['longitude'] = sum(lons) / len(lons)

    return grouped


def delete_duplicates(target_zip):
    """
    Identifies overlapping cross-source fields inside the same park boundary/zip code
    using a 80-meter spatial radius and removes the duplicates dynamically.
    """
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()

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
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            pool.putconn(conn)


def log_processed(state_code, city, zip_code):
    print(f"Log Execution Metric: Processed batch for {city}, {state_code} {zip_code}")


def save_field_data(fields):
    if not fields:
        return

    grouped_fields = group_incoming_fields(fields)
    target_zip = fields[0]['postal_code']

    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()

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
                'formatted_address': row[4]
                # 'number_of_fields': row[5] or 1
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
                        gearth_link       = %s
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
        if cur:
            cur.close()
        if conn:
            pool.putconn(conn)


# Annotated YOLO detection images are written here so you can eyeball what the
# model saw. Toggle from the sidebar (set_detection_image_saving).
DETECTION_IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "yolo_detection_results")
SAVE_DETECTION_IMAGES_DEFAULT = True
_SAVE_DETECTION_IMAGES = SAVE_DETECTION_IMAGES_DEFAULT


def set_detection_image_saving(enabled: bool) -> None:
    """Enable/disable writing annotated YOLO images to DETECTION_IMAGE_DIR."""
    global _SAVE_DETECTION_IMAGES
    _SAVE_DETECTION_IMAGES = bool(enabled)


def _save_detection_image(results, field_id, field_label: str = "") -> Optional[str]:
    """Save the YOLO-annotated image (bounding boxes drawn) to the local disk.

    Uses ultralytics' results[0].plot() (returns a BGR array with boxes) and
    writes it as a JPG named "<field_id>_<field_label>.jpg". Saved even when
    there are 0 detections, so a blank-box image documents WHY YOLO found none.
    Returns the written path, or None on failure.
    """
    try:
        annotated = results[0].plot()          # BGR ndarray with boxes drawn
        rgb = annotated[:, :, ::-1]            # BGR -> RGB for PIL
        os.makedirs(DETECTION_IMAGE_DIR, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (field_label or "").strip())[:80]
        fname = f"{field_id}_{safe}.jpg" if safe else f"{field_id}.jpg"
        path = os.path.join(DETECTION_IMAGE_DIR, fname)
        Image.fromarray(rgb).save(path, quality=90)
        return path
    except Exception as e:
        print(f"Error saving detection image for field {field_id}: {e}")
        return None


def object_detection_based_modification_by_class(field_id, img_array, model, gps_loc, target_classes, zoom_level=18, field_label=""):
    modified_records = []
    try:
        results = model.predict(img_array, verbose=False)
        if _SAVE_DETECTION_IMAGES:
            _save_detection_image(results, field_id, field_label)
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
    if not nge_objects:
        return
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
            conf_percent = obj['confidence_score'] * 100
            desc = f"Detected via YOLO with {conf_percent:.1f}% confidence"

            cur.execute("""
                INSERT INTO public.nge_object (field_search_id, sport_name, description)
                VALUES (%s, %s, %s);
            """, (obj['field_search_id'], obj['sport_name'], desc))

            cur.execute("""
                UPDATE public.new_google_earth
                SET gps_location = %s
                WHERE field_search_id = %s;
            """, (obj['adjusted_gps'], obj['field_search_id']))

        conn.commit()
        print(f"🔄 Successfully updated gps_locations in new_google_earth and logged {len(nge_objects)} details in nge_object.")
    except Exception as e:
        print(f"Error writing to database tables: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            pool.putconn(conn)


# =============================================================================
# SEARCH → FIELD-RECORD ADAPTER  (glue between QLever search and GE pipeline)
# =============================================================================
# Map QLever's sport keys onto NewGoogleEarthNew.py's numeric SPORT_TAGS ids.
QLEVER_TO_SPORT_TAG: Dict[str, int] = {
    "Baseball / Softball": 8,
    "Basketball": 9,
    "Soccer / Football": 79,
    "Tennis": 87,
    "Volleyball": 90,
}


def _entry_to_fields(entry: dict, sport_tag: int, city: str, state: str,
                     city_zip: str) -> List[dict]:
    """Convert one QLever facility into NewGoogleEarthNew `field` dict(s).

    Emits one field per child pitch/court so every individual court survives to
    the database (matching the qlever Excel's per-court granularity). When a
    facility has multiple courts, each court gets a distinct name
    ("<Facility> <Sport> <n>") placed OUTSIDE any parentheses so
    get_base_name / group_incoming_fields treat them as separate records instead
    of collapsing them into one. Single-court facilities keep their plain name.
    """
    address = entry.get("address", "") or f"{city}, {state} {city_zip}".strip()
    street = address.split(",")[0].strip() if address else ""

    children = entry.get("child_pitches", []) or []
    coords: List[Tuple[float, float]] = [
        (c["lat"], c["lon"]) for c in children
        if c.get("lat") is not None and c.get("lon") is not None
    ]
    if not coords:
        coords = [(entry["lat"], entry["lon"])]

    sport_label = SPORT_TAGS.get(sport_tag, "field").title()
    multi = len(coords) > 1
    base_name = entry["name"]

    fields = []
    for i, (lat, lon) in enumerate(coords, 1):
        # Distinct per-court name keeps group_incoming_fields from merging the
        # courts of one facility into a single record.
        field_name = f"{base_name} {sport_label} {i}" if multi else base_name
        fields.append({
            'field_name': field_name,
            'formatted_address': address,
            'postal_code': city_zip,
            'street': street,
            'city': city,
            'state': state,
            'original_gps_location': {'latitude': float(lat), 'longitude': float(lon)},
            'gplace_id': f"qlever_{lat}_{lon}",
            'search_sport_type': sport_tag,
            'gearth_link': build_gearth_link(lat, lon),
            'modified_fields': [],
        })
    return fields


def resolve_city_zip(sport_results: Dict[str, Tuple[dict, int]]) -> str:
    """Determine ONE ZIP code for the whole city (Requirement 6).

    Uses the most common reverse-geocoded ZIP across all found facilities.
    This single ZIP is reused for every record's postal_code, so the original
    single-postal_code DB dedup and YOLO steps run unchanged.
    """
    zips: List[str] = []
    for _sport, (categories, _total) in sport_results.items():
        if not categories:
            continue
        for entries in categories.values():
            for e in entries:
                z = e.get("zipcode") or _parse_zip_from_address(e.get("address", ""))
                if z:
                    zips.append(z.split("-")[0])  # normalise ZIP+4 to 5-digit
    if zips:
        return Counter(zips).most_common(1)[0][0]
    return ""


def adapt_search_to_fields(sport_results: Dict[str, Tuple[dict, int]],
                           city: str, state: str,
                           city_zip: str) -> List[dict]:
    """Flatten the QLever per-sport categorised results into a flat list of
    NewGoogleEarthNew field dicts ready for save_field_data (unchanged)."""
    all_fields: List[dict] = []
    for sport_choice, (categories, _total) in sport_results.items():
        sport_tag = QLEVER_TO_SPORT_TAG.get(sport_choice)
        if sport_tag is None or not categories:
            continue
        for entries in categories.values():
            for entry in entries:
                if entry.get("lat") is None or entry.get("lon") is None:
                    continue
                all_fields.extend(
                    _entry_to_fields(entry, sport_tag, city, state, city_zip))
    return all_fields


def build_preview_dataframe(all_fields: List[dict]) -> pd.DataFrame:
    """Build the preview table of the records that WOULD be inserted.

    Mirrors save_field_data's grouping + final-name construction (without any
    DB access) so the operator sees exactly what the insert will produce, with
    every database column represented. Runs on a deep copy so it never mutates
    the fields that the confirmed insert will consume.
    """
    grouped = group_incoming_fields(deepcopy(all_fields))
    rows = []
    for f in grouped:
        lat = f['original_gps_location']['latitude']
        lon = f['original_gps_location']['longitude']
        base_name = get_base_name(f['field_name'])
        sports_str = ", ".join(sorted(f['sports']))
        rows.append({
            "Facility Name": f"{base_name} ({sports_str})",
            "Sport": SPORT_TAGS.get(f['search_sport_type'], "Field").title(),
            "Address": f['formatted_address'],
            "Street": f['street'],
            "City": f['city'],
            "State": f['state'],
            "ZIP": f['postal_code'],
            "Latitude": round(lat, 6),
            "Longitude": round(lon, 6),
            # "Number of Fields": f['number_of_fields'],
            "GPlace ID": f['gplace_id'],
            "Google Earth Link": f['gearth_link'],
        })
    return pd.DataFrame(rows)


# =============================================================================
# YOLO SATELLITE RECENTERING STAGE  (post-insert, behaviour preserved)
# =============================================================================
def _fetch_db_fields_for_zip(zip_code: str,
                             status_callback: Callable[[str], None]) -> List[dict]:
    """Pull the just-inserted rows for this ZIP (verbatim from the original
    __main__ database query)."""
    conn, cur = None, None
    db_fields: List[dict] = []
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
        status_callback(f"Error fetching filtered fields from database: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            pool.putconn(conn)
    return db_fields


def run_yolo_recentering_stage(zip_code: str, model,
                               status_callback: Callable[[str], None]) -> int:
    """Reproduce the original YOLO recentering pipeline.

    Satellite image downloads (I/O bound) run in parallel; YOLO inference stays
    serial because the model instance is not thread-safe. Behaviour and outputs
    (nge_object rows, updated gps_location) are identical to the original.
    """
    db_fields = _fetch_db_fields_for_zip(zip_code, status_callback)
    status_callback(f"Rows matching criteria for YOLO processing: {len(db_fields)}")

    def _download(row: dict):
        try:
            lat, lon = row['gps_location'].split(",")
            gps_loc = {'latitude': float(lat), 'longitude': float(lon)}
        except (ValueError, AttributeError):
            return row, None, None
        return row, gps_loc, get_satellite_image_array(gps_loc)

    fetched: List[Tuple[dict, Optional[dict], Any]] = []
    if db_fields:
        with ThreadPoolExecutor(max_workers=8) as ex:
            fetched = list(ex.map(_download, db_fields))

    nge_object: List[dict] = []
    for row, gps_loc, img_array in fetched:
        if gps_loc is None or img_array is None:
            continue
        status_callback(f"Processing field: {row['field_name']} at {row['gps_location']}")
        nge_object.extend(
            object_detection_based_modification_by_class(
                row['field_search_id'], img_array, model, gps_loc,
                ALL_TARGET_CLASS_IDS, zoom_level=18,
                field_label=row.get('field_name', '')))

    if nge_object:
        save_object_data(nge_object)
    return len(nge_object)


# =============================================================================
# SELF-HEALING BACKFILL  (guarantee every field has an nge_object row)
# =============================================================================
def _default_object_for_row(row: dict, reason: str) -> dict:
    """Build a default nge_object payload for a row YOLO could not recenter.

    sport_name is derived from the searched sport tag so the backfilled row
    still carries a meaningful label; description records WHY YOLO gave zero.
    """
    sport_name = SPORT_TAGS.get(row.get("search_sport_type"), "Field").title()
    return {
        "field_search_id": row["field_search_id"],
        "sport_name": sport_name,
        "description": reason,
    }


def _insert_default_objects(default_objects: List[dict],
                            status_callback: Callable[[str], None]) -> int:
    """Insert zero-detection / image-unavailable placeholder rows.

    Unlike save_object_data this does NOT delete existing rows and does NOT
    touch new_google_earth.gps_location — these fields keep their original GPS.
    sport_name/description are truncated to the VARCHAR(64) column width.
    Returns the number of rows actually committed (0 on failure/rollback).
    """
    if not default_objects:
        return 0
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        for obj in default_objects:
            cur.execute("""
                INSERT INTO public.nge_object (field_search_id, sport_name, description)
                VALUES (%s, %s, %s);
            """, (obj["field_search_id"],
                  obj["sport_name"][:DESCRIPTION_MAX_LEN],
                  obj["description"][:DESCRIPTION_MAX_LEN]))
        conn.commit()
        status_callback(
            f"Backfilled {len(default_objects)} default nge_object row(s) "
            f"(0-detection / image-unavailable), each with a reason.")
        return len(default_objects)
    except Exception as e:
        status_callback(f"Error inserting default nge_object rows: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if cur:
            cur.close()
        if conn:
            pool.putconn(conn)


def backfill_missing_objects(zip_code: Optional[str] = None,
                             model=None,
                             status_callback: Optional[Callable[[str], None]] = None) -> int:
    """Ensure every new_google_earth row has a matching nge_object entry.

    Scans for new_google_earth rows that lack an nge_object row (LEFT JOIN).
    For each missing row it re-attempts YOLO detection; real detections are
    written via save_object_data (GPS recentered, as normal). When the image
    cannot be fetched or YOLO yields 0 detections, a default nge_object row is
    inserted whose description explains WHY YOLO produced zero — so no field is
    ever left without a record.

    Args:
        zip_code: restrict the scan to one postal_code; None scans all rows.
        model: a loaded YOLO model; if None it is loaded on demand.
        status_callback: progress sink; defaults to print.

    Returns:
        Number of nge_object rows created (real detections + defaults).
    """
    cb = status_callback or print

    conn, cur = None, None
    missing_fields: List[dict] = []
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        query = """
            SELECT nge.field_search_id, nge.field_name, nge.gps_location,
                   nge.search_sport_type, nge.gearth_link
            FROM public.new_google_earth nge
            LEFT JOIN public.nge_object obj
                   ON nge.field_search_id = obj.field_search_id
            WHERE obj.nge_object_id IS NULL
        """
        params: List[Any] = []
        if zip_code:
            query += " AND nge.postal_code = %s;"
            params.append(zip_code)
        else:
            query += ";"
        cur.execute(query, tuple(params))
        columns = [desc[0] for desc in cur.description]
        missing_fields = [dict(zip(columns, row)) for row in cur.fetchall()]
        cb(f"Self-Healing Check: Found {len(missing_fields)} record(s) "
           f"missing from nge_object.")
    except Exception as e:
        cb(f"Error checking for missing object records: {e}")
        return 0
    finally:
        if cur:
            cur.close()
        if conn:
            pool.putconn(conn)

    if not missing_fields:
        cb("Safety check passed: every new_google_earth record has an "
           "nge_object entry.")
        return 0

    if model is None:
        model = load_object_detection_model()

    # Parallel satellite downloads (I/O bound); serial YOLO inference after.
    def _download(row: dict):
        gps_raw = row.get("gps_location")
        try:
            lat, lon = gps_raw.split(",")
            gps_loc = {"latitude": float(lat), "longitude": float(lon)}
        except (ValueError, AttributeError):
            return row, None, None  # invalid GPS
        return row, gps_loc, get_satellite_image_array(gps_loc)

    with ThreadPoolExecutor(max_workers=8) as ex:
        fetched = list(ex.map(_download, missing_fields))

    real_detections: List[dict] = []
    default_objects: List[dict] = []
    for row, gps_loc, img_array in fetched:
        if gps_loc is None:
            default_objects.append(_default_object_for_row(row, INVALID_GPS_REASON))
            continue
        if img_array is None:
            default_objects.append(_default_object_for_row(row, IMAGE_UNAVAILABLE_REASON))
            continue
        dets = object_detection_based_modification_by_class(
            row["field_search_id"], img_array, model, gps_loc,
            ALL_TARGET_CLASS_IDS, zoom_level=18,
            field_label=row.get("field_name", ""))
        if dets:
            real_detections.extend(dets)
        else:
            default_objects.append(
                _default_object_for_row(row, YOLO_ZERO_DETECTION_REASON))

    if real_detections:
        cb(f"Backfill: {len(real_detections)} real detection(s) recentered.")
        save_object_data(real_detections)
    inserted_defaults = _insert_default_objects(default_objects, cb)

    return len(real_detections) + inserted_defaults


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================
def _run_insert_and_yolo(all_fields: List[dict], city_zip: str,
                         city: str, state: str) -> None:
    """Execute the confirmed database insertion, then the YOLO stage."""
    log_messages: List[str] = []

    def log(msg: str) -> None:
        log_messages.append(str(msg))

    status = st.status("Inserting into database...", expanded=True)
    with status:
        # Existing insertion logic, reused exactly as implemented.
        save_field_data(all_fields)
        log(f"Inserted/updated {len(all_fields)} candidate records into new_google_earth")

        try:
            log_processed(state, city, city_zip)
        except NameError:
            pass

        st.write("Loading object-detection model...")
        model = load_object_detection_model()
        st.write(f"Running YOLO satellite recentering for ZIP {city_zip}...")
        detections = run_yolo_recentering_stage(city_zip, model, log)
        log(f"YOLO recentering complete: {detections} detection(s) applied")

        # Self-healing: any field with 0 YOLO detections (or a failed image
        # fetch) still gets an nge_object row explaining why YOLO gave zero.
        st.write("Backfilling nge_object for fields YOLO could not recenter...")
        backfilled = backfill_missing_objects(city_zip, model, log)
        log(f"Backfill complete: {backfilled} nge_object row(s) created")
        status.update(label="✅ Insertion + recentering complete", state="complete")

    st.success("Data input process completed successfully.")
    with st.expander(f"📜 Insert log ({len(log_messages)} lines)"):
        st.code("\n".join(log_messages), language="text")


def main() -> None:
    st.set_page_config(
        page_title="Google Earth Facility Finder",
        page_icon="🌍",
        layout="wide",
    )
    st.title("🌍 Google Earth Sports Facility Finder")
    st.markdown("""
    Search a US city for sports facilities using the **QLever SPARQL** OSM index
    plus **NCES** public-school data, review every record that will be written,
    then confirm insertion into the Google Earth database. Confirmed inserts are
    followed by the YOLO satellite-recentering stage — identical to the original
    pipeline.
    """)

    if pool is None:
        st.error(
            "Database connection pool unavailable — check `config.ini` has a "
            f"valid `[database]` section.\n\nDetails: {_POOL_ERROR}"
        )
        st.stop()

    # ---- Sidebar (mirrors qlever_v_6_json.py layout) --------------------
    with st.sidebar:
        st.header("⚙️ Settings")
        sports_selected = st.multiselect(
            "Sports to fetch",
            options=list(SPORTS_CONFIG.keys()),
            default=list(SPORTS_CONFIG.keys()),
        )
        max_workers = st.slider(
            "Parallel workers",
            min_value=1, max_value=8, value=1,
            help=("Threads running per-sport search jobs concurrently. The "
                  "public QLever endpoint rate-limits aggressively — QLever "
                  "POSTs are serialized globally regardless, so keep this at 1 "
                  "(raise only if you stop seeing throttling)."),
        )
        use_cache = st.checkbox("Use response cache", value=True)

        st.divider()
        st.caption("🛰️ Satellite source (for YOLO)")
        sat_choice = st.radio(
            "Satellite imagery source",
            options=["Esri World Imagery (free)", "Google Static Maps (paid)"],
            index=0,
            label_visibility="collapsed",
            help=("Esri = free, no API key, sub-metre in the US, zoom 18 — the "
                  "YOLO model was trained on Google imagery so Esri may miss a "
                  "few; Google matches training exactly but bills per image."),
        )
        set_satellite_provider("esri" if sat_choice.startswith("Esri") else "google")

        save_det_imgs = st.checkbox(
            "Save YOLO detection images", value=SAVE_DETECTION_IMAGES_DEFAULT,
            help=("Write each satellite image with YOLO boxes drawn to "
                  "yolo_detection_results/ so you can verify detections."))
        set_detection_image_saving(save_det_imgs)
        if save_det_imgs:
            st.caption(f"🖼️ Images → `{DETECTION_IMAGE_DIR}`")

        count, size = cache_stats()
        if count > 0:
            st.caption(f"💾 Cache: {count} entries, {size/1024/1024:.1f} MB")
            if st.button("🗑️ Clear cache", use_container_width=True):
                cache_clear()
                st.success("Cleared.")
        else:
            st.caption("💾 Cache empty")

    # ---- Search inputs --------------------------------------------------
    st.subheader("1. Search a city")
    c1, c2, c3 = st.columns(3)
    city = c1.text_input("City", placeholder="Daly City")
    state = c2.text_input("State", placeholder="California")
    county = c3.text_input("County (optional)", placeholder="San Mateo County")
    run_search = st.button("🔎 Search", type="primary", use_container_width=True)

    if run_search:
        if not city.strip() or not state.strip():
            st.error("City and State are required.")
        elif not sports_selected:
            st.error("Pick at least one sport in the sidebar.")
        else:
            log_messages: List[str] = []

            def _log(msg: str) -> None:
                log_messages.append(str(msg))

            status = st.status(f"Searching {city.strip()}...", expanded=True)
            with status:
                t0 = time.time()
                sport_results, errors = run_city_search(
                    city.strip(), county.strip(), state.strip(),
                    sports_selected, DEFAULT_OVERPASS_URL, use_cache,
                    max_workers, _log)
                city_zip = resolve_city_zip(sport_results)
                all_fields = adapt_search_to_fields(
                    sport_results, city.strip(), state.strip(), city_zip)
                status.update(
                    label=f"✅ Search done in {time.time()-t0:.1f}s — "
                          f"{len(all_fields)} candidate records",
                    state="complete")

            st.session_state["nge_result"] = {
                "city": city.strip(),
                "state": state.strip(),
                "city_zip": city_zip,
                "all_fields": all_fields,
                "errors": errors,
                "log_messages": log_messages,
                "inserted": False,
            }

    # ---- Preview + confirmation gate ------------------------------------
    result = st.session_state.get("nge_result")
    if not result:
        st.info("👆 Enter a city and state, then press Search.")
        return

    st.subheader(f"2. Preview — {result['city']}, {result['state']}")
    if st.button("🔄 Clear result / new search", key="clear_result"):
        del st.session_state["nge_result"]
        st.rerun()

    if result["errors"]:
        with st.expander(f"⚠️ {len(result['errors'])} search error(s)"):
            for e in result["errors"]:
                st.text(e)

    with st.expander(f"📜 Search log ({len(result['log_messages'])} lines)"):
        st.code("\n".join(result["log_messages"]), language="text")

    all_fields = result["all_fields"]
    if not all_fields:
        st.warning("No facilities found for this city. Nothing to insert.")
        return

    if result.get("inserted"):
        st.success("✅ Records already inserted for this search. "
                   "Start a new search to run again.")
        return

    city_zip = result["city_zip"]
    if city_zip:
        st.caption(f"City-level ZIP resolved once and reused: **{city_zip}**")
    else:
        st.warning("Could not resolve a city ZIP from search results; "
                   "records will be inserted with an empty postal_code.")

    preview_df = build_preview_dataframe(all_fields)
    st.markdown(f"### 🧾 Preview Ready — {len(preview_df)} record(s) to insert")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    b1, b2 = st.columns(2)
    insert_clicked = b1.button("✅ Insert Into Database", type="primary",
                               use_container_width=True)
    cancel_clicked = b2.button("❌ Cancel", use_container_width=True)

    if cancel_clicked:
        del st.session_state["nge_result"]
        st.info("Cancelled. No database writes were performed.")
        st.rerun()

    if insert_clicked:
        _run_insert_and_yolo(all_fields, city_zip, result["city"], result["state"])
        st.session_state["nge_result"]["inserted"] = True


if __name__ == "__main__":
    main()
