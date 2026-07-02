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
# total_temp_small_path = config['image_paths']['total_temp_small_path']
# total_temp_large_path = config['image_paths']['total_temp_large_path']
getImage_total_error = config['error']['getImage_total_error']
obd_model_path = config['model_paths']['obd_model']
target_class_ids = [4, 5, 9, 10, 14, 16]
# IMG_FOR_RECENTER_WIDTH = 700
# IMG_FOR_RECENTER_HEIGHT = 750
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

# This function gets the latitude and longitude from the google earth link 
def gpsLocation(link):
    """
    Extracts the latitude and longitude from a Google Earth link.
    e.g., '@43.1798322,-71.8227760,4.1972381a,15000d'
    """
    start_index = link.find('@') + 1 
    if start_index == 0:
        return 'Error', 'Error'
        
    first_comma_index = link.find(',', start_index)
    second_comma_index = link.find(',', first_comma_index + 1)
    
    if first_comma_index != -1 and second_comma_index != -1:
        lat = link[start_index:first_comma_index]
        lon = link[first_comma_index + 1:second_comma_index]
        return lat, lon
    else:
        return 'Error', 'Error'

# Gets the url for the field image
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

def saveImg(image_url, save_path, crop_bottom):
    """Downloads an image from a URL, crops it, and saves it to a file."""
    try:
        image_data = urllib.request.urlopen(image_url).read() # Download the image as binary
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Crop the image to remove the bottom Google Earth tag
        height, width, _ = image.shape
        cropped_image = image[0:height - crop_bottom, 0:width]
        
        # Save the cropped image
        cv2.imwrite(save_path, cropped_image)
        print(f"Saved image to: {save_path}")
    except Exception as e:
        print(f"Error saving image from {image_url} to {save_path}: {e}")

def _safe_object_id_str(nge_object_id):
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
        new_row_data = {
            'field_name': current_name,
            'field_map': current_map,
            'field_id': current_field_id,
            'nge_object_id': nge_object_id
        }
        new_row_df = pd.DataFrame([new_row_data])
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
    print(df['sport_name'].value_counts())
    print(df[['field_id', 'field_name', 'sport_name', 'field_map']].head(15))
    df_error = pd.DataFrame(columns=['searched_sport_id', 'field_name', 'field_map', 'field_id','sport_name']) #create another df to hold any errors
    obd_model = YOLO(obd_model_path) # Trained with DIOR dataset
    print(f"Retrieved {len(df)} entries from the database")
    processed = 0
    for index, row in df.iterrows():
        if np.isnan(row['searched_sport_id']):
            searched_sport = 0
        else:
            searched_sport = int(row['searched_sport_id'])
        current_name = row['field_name']
        current_map = row['field_map']
        current_field_id = row['field_id']
        sport_name = row['sport_name']
        nge_object_id = row['nge_object_id']

        print(f"Processing field: {current_name}, field_id: {current_field_id}, field_map: {current_map}")
        if sport_name not in ALLOWED_SPORT_NAMES:
            print(f"Skipping unsupported detected sport for field_id {current_field_id}: {sport_name}")
            continue
        if pd.isna(nge_object_id):
            print(f"Skipping field_id {current_field_id} because nge_object_id is missing.")
            continue
        print(f"Processing detected facility: {current_name}, field_id: {current_field_id}, sport_name: {sport_name}")
        df_error = ProcessImage(current_map, small_img_paths, current_field_id, current_name, large_img_paths, df_error, sport_name, nge_object_id)
        processed += 1
        print(f"Processed {processed} of {df.shape[0]} detected facilities")

    df_error.to_excel(getImage_total_error)
    print(f"Errors saved to: {getImage_total_error}")
