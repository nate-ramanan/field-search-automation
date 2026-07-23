import urllib.parse
import urllib.request
import cv2
import numpy as np
import pyproj
from pyproj import Transformer
import io
from ultralytics import YOLO
from config_getImagesObject import get_field_data
import pandas as pd
from findShift import *
import os
import shutil
from configparser import ConfigParser
import collections.abc
from ConnectionPool import pool

# Get config data
file = './config.ini'
config = ConfigParser()
config.read(file)
small_img_paths = config['image_paths']['total_zoom_small_path']
large_img_paths = config['image_paths']['total_zoom_large_path']
getImage_total_error = config['error']['getImage_total_error']
obd_model_path = config['model_paths']['obd_model']

target_class_ids = [4, 5, 9, 10, 14, 16]
SMALL_IMG_WIDTH = 300
SMALL_IMG_HEIGHT = 350
LARGE_IMG_WIDTH = 700
LARGE_IMG_HEIGHT = 750

KEY = 'AIzaSyC5cT2KgRuUuz51GQ71DvY8gB_VN8O8EtE'

display_names = {
    4: 'Baseball', 5: 'Basketball', 9: 'Golf',
    10: 'Soccer', 14: 'Stadium', 16: 'Tennis'
}
ALLOWED_SPORT_NAMES = {'Baseball', 'Basketball', 'Golf', 'Soccer', 'Stadium', 'Tennis'}
ALLOWED_SPORT_CLASS_IDS = {4, 5, 9, 10, 14, 16}

def gpsLocation(link):
    """
    Extracts the latitude and longitude dynamically from a Google Earth link,
    an OpenStreetMap link, or a raw coordinate string. Captures float NaN and 
    None values early to prevent parsing crashes.
    """
    if not link or pd.isna(link) or not isinstance(link, str):
        return 'Error', 'Error'
        
    link = link.strip()

    # 1. Handle Google Earth link format (e.g., '@43.1798322,-71.8227760,4.1972381a,15000d')
    if '@' in link:
        start_index = link.find('@') + 1 
        first_comma_index = link.find(',', start_index)
        second_comma_index = link.find(',', first_comma_index + 1)
        
        if first_comma_index != -1 and second_comma_index != -1:
            lat = link[start_index:first_comma_index]
            lon = link[first_comma_index + 1:second_comma_index]
            return lat.strip(), lon.strip()

    # 2. Handle OpenStreetMap link format (e.g., 'https://www.openstreetmap.org/#map=19/43.1798322/-71.8227760')
    if '#map=' in link:
        try:
            map_part = link.split('#map=')[1]
            parts = map_part.split('/')
            if len(parts) >= 3:
                lat = parts[1]
                lon = parts[2]
                return lat.strip(), lon.strip()
        except Exception:
            pass

    # 3. Handle raw 'lat,lon' coordinate string fallback
    if ',' in link:
        try:
            parts = link.split(',')
            if len(parts) >= 2:
                float(parts[0])
                float(parts[1])
                return parts[0].strip(), parts[1].strip()
        except ValueError:
            pass

    return 'Error', 'Error'

def getImage(lat, lon, key, zoom, width, height):
    """Generates the URL for a satellite image using extracted coordinates."""
    if lat != 'Error' and lon != 'Error':
        return "https://maps.googleapis.com/maps/api/staticmap?key={}&center={},{}&zoom={}&size={}x{}&maptype=satellite".format(
            key, lat, lon, zoom, width, height
        )
    return 'Error'

def saveImg(image_url, save_path, crop_bottom):
    """Downloads an image from a URL, crops it, and saves it to a file."""
    try:
        image_data = urllib.request.urlopen(image_url).read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        height, width, _ = image.shape
        cropped_image = image[0:height - crop_bottom, 0:width]
        
        cv2.imwrite(save_path, cropped_image)
        print(f"Saved image to: {save_path}")
    except Exception as e:
        print(f"Error saving image from {image_url} to {save_path}: {e}")

def _safe_object_id_str(nge_object_id):
    """Safely converts object IDs to strings; defaults to 'orig' for unclassified/OSM records."""
    if pd.isna(nge_object_id) or nge_object_id is None:
        return 'orig'
    try:
        return str(int(float(nge_object_id)))
    except (TypeError, ValueError):
        return 'orig'

def ProcessImage(current_map, small_img_paths, current_field_id, current_name, large_img_paths, df_error, sport_name, nge_object_id):
    if not current_map or pd.isna(current_map):
        try:
            return str(int(float(nge_object_id)))
        except (TypeError, ValueError):
            return None

def ProcessImage(current_map,small_img_paths,current_field_id,current_name,large_img_paths,df_error,sport_name, nge_object_id):
    if current_map:

        try:
            lat, lon = gpsLocation(current_map)
            obj_id_str = _safe_object_id_str(nge_object_id)
            if obj_id_str is None:
                print(f"Skipping field_id {current_field_id} because nge_object_id is missing or invalid.")
                return df_error
            if sport_name not in ALLOWED_SPORT_NAMES:
                print(f"Skipping unsupported sport label for field_id {current_field_id}: {sport_name}")
                return df_error
            save_path_small = small_img_paths + '/' + obj_id_str + '_' + str(current_field_id) + '_' + sport_name + '.png'
            # Use the newly centered GPS location for the next image grabs
            if ('Basketball' in sport_name) or ('Tennis' in sport_name):
                image_url_small = getImage(
                    lat,
                    lon,
                    KEY,
                    19,
                    SMALL_IMG_WIDTH,
                    SMALL_IMG_HEIGHT
                )
            elif ('Baseball' in sport_name):
                image_url_small = getImage(
                    lat,
                    lon,
                    KEY,
                    18,
                    SMALL_IMG_WIDTH,
                    SMALL_IMG_HEIGHT
                )
            else:
                image_url_small = getImage(
                    lat,
                    lon,
                    KEY,
                    19,
                    SMALL_IMG_WIDTH,
                    SMALL_IMG_HEIGHT
                )
            saveImg(image_url_small, save_path_small, 30)

            image_url_large = getImage(
                lat,
                lon,
                KEY,
                18,
                LARGE_IMG_WIDTH,
                LARGE_IMG_HEIGHT
            )
            save_path_large = large_img_paths + '/' + obj_id_str + '_' + str(current_field_id) + '_' + sport_name + '.png'

            saveImg(image_url_large, save_path_large, 30)
        except Exception as error:
            print(f"Error processing field_id {current_field_id}: {error}")
            new_row_data = {
                'field_name': current_name,
                'field_map': current_map,
                'field_id': current_field_id,
                'nge_object_id': nge_object_id
            }
            new_row_df = pd.DataFrame([new_row_data])
            df_error = pd.concat([df_error, new_row_df], ignore_index=True)
    else:
        # There was no map so add to error sheet
        print(f"No map available for field_id: {current_field_id}")
        new_row_df = pd.DataFrame([{'field_name': current_name, 'field_map': current_map, 'field_id': current_field_id, 'nge_object_id': nge_object_id}])
        return pd.concat([df_error, new_row_df], ignore_index=True)

    try:
        lat, lon = gpsLocation(current_map)
        if lat == 'Error' or lon == 'Error':
            raise ValueError(f"Could not parse GPS coordinates from field_map: {current_map!r}")
            
        obj_id_str = _safe_object_id_str(nge_object_id)
        sport_name_str = 'Unclassified' if pd.isna(sport_name) else str(sport_name).strip()

        save_path_small = f"{small_img_paths}/{obj_id_str}_{current_field_id}_{sport_name_str}.png"

        # Apply specific target zoom configurations
        if 'Baseball' in sport_name_str:
            zoom_level = 18
        elif 'Basketball' in sport_name_str or 'Tennis' in sport_name_str:
            zoom_level = 19
        else:
            zoom_level = 19

        image_url_small = getImage(lat, lon, KEY, zoom_level, SMALL_IMG_WIDTH, SMALL_IMG_HEIGHT)
        saveImg(image_url_small, save_path_small, 30)

        image_url_large = getImage(lat, lon, KEY, 18, LARGE_IMG_WIDTH, LARGE_IMG_HEIGHT)
        save_path_large = f"{large_img_paths}/{obj_id_str}_{current_field_id}_{sport_name_str}.png"
        saveImg(image_url_large, save_path_large, 30)
        
    except Exception as error:
        print(f"Error processing field_id {current_field_id}: {error}")
        new_row_df = pd.DataFrame([{'field_name': current_name, 'field_map': current_map, 'field_id': current_field_id, 'nge_object_id': nge_object_id}])
        df_error = pd.concat([df_error, new_row_df], ignore_index=True)
        
    return df_error


if __name__ == '__main__':
    if os.path.exists(small_img_paths): 
        shutil.rmtree(small_img_paths)
    if os.path.exists(large_img_paths): 
        shutil.rmtree(large_img_paths)
    print("Results for previous runs wiped!")
    os.makedirs(small_img_paths, exist_ok=True)
    os.makedirs(large_img_paths, exist_ok=True)
    
    df = get_field_data(pool)
    print(f"Total rows read: {len(df)}")
    
    df_error = pd.DataFrame(columns=['searched_sport_id', 'field_name', 'field_map', 'field_id', 'sport_name'])
    obd_model = YOLO(obd_model_path)
    
    processed = 0
    for index, row in df.iterrows():
        current_name = row['field_name']
        current_map = row['field_map']
        current_field_id = row['field_id']
        sport_name = row['sport_name']
        nge_object_id = row['nge_object_id']

        print(f"Processing field: {current_name}, field_id: {current_field_id}")
    
        # Check if this row represents a newly imported OpenStreetMap entry 
        # or an unclassified spot that needs the YOLO model initialization loop
        needs_detection = pd.isna(sport_name) or sport_name == 'orig' or sport_name not in ALLOWED_SPORT_NAMES

        if not needs_detection:
            # Straight pipeline path for entries with confirmed sports matches
            df_error = ProcessImage(current_map, small_img_paths, current_field_id, current_name, large_img_paths, df_error, sport_name, nge_object_id)
            processed += 1
        else:
            # OpenStreetMap compatibility branch with live classification fallback
            if not current_map or pd.isna(current_map):
                print(f"Skipping detection for field_id {current_field_id}: missing or nan field map.")
                new_row_df = pd.DataFrame([{'field_name': current_name, 'field_map': current_map, 'field_id': current_field_id, 'nge_object_id': nge_object_id}])
                df_error = pd.concat([df_error, new_row_df], ignore_index=True)
                continue

            lat, lon = gpsLocation(current_map)
            if lat == 'Error' or lon == 'Error':
                print(f"Skipping detection for field_id {current_field_id}: could not parse GPS coordinates from field_map: {current_map!r}")
                new_row_df = pd.DataFrame([{'field_name': current_name, 'field_map': current_map, 'field_id': current_field_id, 'nge_object_id': nge_object_id}])
                df_error = pd.concat([df_error, new_row_df], ignore_index=True)
                continue

            image_url = getImage(lat, lon, KEY, 18, LARGE_IMG_WIDTH, LARGE_IMG_HEIGHT)
            try:
                image_data = urllib.request.urlopen(image_url).read()
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                results = obd_model.predict(image, verbose=False)
                
                # Filter classes to match only allowed categories
                detected_names = [int(box.cls) for res in results for box in res.boxes if int(box.cls) in ALLOWED_SPORT_CLASS_IDS]
                
                if detected_names:
                    assigned_class = max(set(detected_names), key=detected_names.count)
                    assigned_sport_name = display_names.get(assigned_class, "Unclassified")
                    df_error = ProcessImage(current_map, small_img_paths, current_field_id, current_name, large_img_paths, df_error, assigned_sport_name, nge_object_id)
                    processed += 1
                else:
                    print(f"No valid sport field detected via YOLO for field_id {current_field_id}")
                    new_row_df = pd.DataFrame([{'field_name': current_name, 'field_map': current_map, 'field_id': current_field_id, 'nge_object_id': nge_object_id}])
                    df_error = pd.concat([df_error, new_row_df], ignore_index=True)
            except Exception as e:
                print(f"Skipping detection for field_id {current_field_id} due to image fetch error: {e}")
                new_row_df = pd.DataFrame([{'field_name': current_name, 'field_map': current_map, 'field_id': current_field_id, 'nge_object_id': nge_object_id}])
                df_error = pd.concat([df_error, new_row_df], ignore_index=True)

    df_error.to_excel(getImage_total_error, index=False)
    print(f"Errors successfully written to: {getImage_total_error}")
    print(f"Execution complete. Successfully processed {processed} facilities.")