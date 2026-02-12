from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
import datetime
from configparser import ConfigParser
from tf_keras.models import load_model
import gc
import shutil
from ConnectionPool import pool

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

NOW = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

config_file = './config.ini'
config = ConfigParser()
config.read(config_file)

small_img_pth = config['image_paths']['total_zoom_small_path']
large_img_pth = config['image_paths']['total_zoom_large_path']
cls_sport_model_path = config['model_paths']['sport_model']
cls_type_model_path = config['model_paths']['type_model']
obd_model_path = config['model_paths']['obd_model']
obd_images_path = config['image_paths']['obj_detect_img_path']

SPORT_CLASSES = ['Buildings', 'Soccer', 'Baseball', 'Tennis', 'Basketball']
TYPE_CLASSES = ['Grass', 'Turf']
CONFIDENCE_THR = 0.45

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _list_image_files(directory: str):
    if not os.path.isdir(directory):
        print(f"Warning: directory does not exist: {directory}")
        return []
    exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    return [f for f in os.listdir(directory) if f.lower().endswith(exts)]


def _parse_ids_from_filename(filename: str):
    """
    Parse GetImagesTotal filename format: nge_object_id_field_search_id_sportname.png
    e.g. 123_456_Soccer.png, 1000.0_9802_Tennis.png, orig_456_Baseball.png
    Returns (field_search_id, nge_object_id).
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    field_search_id = None
    nge_object_id = None

    if len(parts) >= 2:
        try:
            first = parts[0]
            second = parts[1]
            if str(first).lower() == 'orig':
                nge_object_id = None
                field_search_id = int(float(second))
            else:
                nge_object_id = int(float(first))
                field_search_id = int(float(second))
        except (ValueError, TypeError):
            pass

    return field_search_id, nge_object_id


# -----------------------------------------------------------------------------
# Classification on small images
# -----------------------------------------------------------------------------

def Classification(small_image_dir_path: str,
                   cls_sport_model_path: str,
                   cls_type_model_path: str,
                   db_ids_df: pd.DataFrame = None) -> pd.DataFrame:
    # Get database IDs if not provided
    if db_ids_df is None or db_ids_df.empty:
        print("Fetching nge_object IDs from database...")
        db_ids_df = get_nge_object_ids_from_database()
        if db_ids_df.empty:
            print("Warning: No nge_object IDs found in database.")
            return pd.DataFrame(columns=[
                'field_search_id',
                'nge_object_id',
                'predicted_field_type',
                'predicted_sport',
                'predicted_field_type_probability',
                'predicted_sport_type_probability'
            ])
    
    # Match images to database IDs
    matched_images = _match_image_to_database_ids(small_image_dir_path, db_ids_df)
    if not matched_images:
        print(f"No matching images found in {small_image_dir_path} for database records")
        return pd.DataFrame(columns=[
            'field_search_id',
            'nge_object_id',
            'predicted_field_type',
            'predicted_sport',
            'predicted_field_type_probability',
            'predicted_sport_type_probability'
        ])

    type_model = load_model(cls_type_model_path)
    sport_model = load_model(cls_sport_model_path)
    print(f'Finished loading classification models for SMALL images. Processing {len(matched_images)} matched images.', flush=True)

    num_images = len(matched_images)
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE

    df = pd.DataFrame(columns=[
        'field_search_id',
        'nge_object_id',
        'predicted_field_type',
        'predicted_sport',
        'predicted_field_type_probability',
        'predicted_sport_type_probability'
    ])

    print('Total small-image batches:', num_batches)
    for batch_idx in range(num_batches):
        print(f'[SMALL] Processing batch {batch_idx + 1}/{num_batches}', flush=True)
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_images)
        batch_items = matched_images[start_idx:end_idx]

        batch_images = []
        batch_ids = []

        for image_path, field_search_id, nge_object_id, sport_name in batch_items:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: could not read image {image_path}, skipping.")
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            batch_images.append(img)
            batch_ids.append((field_search_id, nge_object_id))

        if not batch_images:
            continue

        batch_images = np.array(batch_images, dtype="uint8")

        predicted_field_type_batch = type_model.predict(batch_images, verbose=0)
        predicted_sport_batch = sport_model.predict(batch_images, verbose=0)

        labels_type = np.argmax(predicted_field_type_batch, axis=1)
        labels_sport = np.argmax(predicted_sport_batch, axis=1)
        field_type_probabilities = np.max(predicted_field_type_batch, axis=1)
        sport_type_probabilities = np.max(predicted_sport_batch, axis=1)

        for i, (field_search_id, nge_object_id) in enumerate(batch_ids):
            if field_search_id is None:
                print(f"Warning: field_search_id is None for batch index {i}, skipping.")
                continue

            classified_field_type = TYPE_CLASSES[labels_type[i]]
            classified_sport = SPORT_CLASSES[labels_sport[i]]
            field_type_probability = float(field_type_probabilities[i])
            sport_type_probability = float(sport_type_probabilities[i])

            # Domain overrides
            if classified_sport in ('Basketball', 'Tennis'):
                classified_field_type = 'Synthetic'
            if classified_sport == 'Buildings':
                classified_field_type = 'Buildings'

            df = pd.concat([df, pd.DataFrame({
                'field_search_id': field_search_id,
                'nge_object_id': nge_object_id,
                'predicted_field_type': classified_field_type,
                'predicted_sport': classified_sport,
                'predicted_field_type_probability': field_type_probability,
                'predicted_sport_type_probability': sport_type_probability
            }, index=[0])], ignore_index=True)

    return df


# -----------------------------------------------------------------------------
# Classification on large images
# -----------------------------------------------------------------------------

def Classification_large(large_image_dir_path: str,
                         cls_sport_model_path: str,
                         cls_type_model_path: str,
                         db_ids_df: pd.DataFrame = None) -> pd.DataFrame:
    # Get database IDs if not provided
    if db_ids_df is None or db_ids_df.empty:
        print("Fetching nge_object IDs from database...")
        db_ids_df = get_nge_object_ids_from_database()
        if db_ids_df.empty:
            print("Warning: No nge_object IDs found in database.")
            return pd.DataFrame(columns=[
                'field_search_id',
                'nge_object_id',
                'predicted_large_field_type',
                'predicted_large_sport',
                'large_predicted_field_type_probability',
                'large_predicted_sport_type_probability'
            ])
    
    # Match images to database IDs
    matched_images = _match_image_to_database_ids(large_image_dir_path, db_ids_df)
    if not matched_images:
        print(f"No matching images found in {large_image_dir_path} for database records")
        return pd.DataFrame(columns=[
            'field_search_id',
            'nge_object_id',
            'predicted_large_field_type',
            'predicted_large_sport',
            'large_predicted_field_type_probability',
            'large_predicted_sport_type_probability'
        ])

    type_model = load_model(cls_type_model_path)
    sport_model = load_model(cls_sport_model_path)
    print(f'Finished loading classification models for LARGE images. Processing {len(matched_images)} matched images.', flush=True)

    num_images = len(matched_images)
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE

    df = pd.DataFrame(columns=[
        'field_search_id',
        'nge_object_id',
        'predicted_large_field_type',
        'predicted_large_sport',
        'large_predicted_field_type_probability',
        'large_predicted_sport_type_probability'
    ])

    print('Total large-image batches:', num_batches)
    for batch_idx in range(num_batches):
        print(f'[LARGE] Processing batch {batch_idx + 1}/{num_batches}', flush=True)
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_images)
        batch_items = matched_images[start_idx:end_idx]

        batch_images = []
        batch_ids = []

        for image_path, field_search_id, nge_object_id, sport_name in batch_items:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: could not read image {image_path}, skipping.")
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            batch_images.append(img)
            batch_ids.append((field_search_id, nge_object_id))

        if not batch_images:
            continue

        batch_images = np.array(batch_images, dtype="uint8")

        predicted_field_type_batch = type_model.predict(batch_images, verbose=0)
        predicted_sport_batch = sport_model.predict(batch_images, verbose=0)

        labels_type = np.argmax(predicted_field_type_batch, axis=1)
        labels_sport = np.argmax(predicted_sport_batch, axis=1)
        field_type_probabilities = np.max(predicted_field_type_batch, axis=1)
        sport_type_probabilities = np.max(predicted_sport_batch, axis=1)

        for i, (field_search_id, nge_object_id) in enumerate(batch_ids):
            if field_search_id is None:
                continue

            classified_field_type = TYPE_CLASSES[labels_type[i]]
            classified_sport = SPORT_CLASSES[labels_sport[i]]
            field_type_probability = float(field_type_probabilities[i])
            sport_type_probability = float(sport_type_probabilities[i])

            if classified_sport in ('Basketball', 'Tennis'):
                classified_field_type = 'Synthetic'
            if classified_sport == 'Buildings':
                classified_field_type = 'Buildings'

            df = pd.concat([df, pd.DataFrame({
                'field_search_id': field_search_id,
                'nge_object_id': nge_object_id,
                'predicted_large_field_type': classified_field_type,
                'predicted_large_sport': classified_sport,
                'large_predicted_field_type_probability': field_type_probability,
                'large_predicted_sport_type_probability': sport_type_probability
            }, index=[0])], ignore_index=True)

    return df


# -----------------------------------------------------------------------------
# Object Detection on large images
# -----------------------------------------------------------------------------

def Object_Detection(large_image_dir_path: str,
                     obd_model_path: str,
                     obd_image_path: str,
                     db_ids_df: pd.DataFrame = None):
    model = YOLO(obd_model_path)

    display_names = {
        0: 'Expressway-Service-area', 1: 'Expressway-toll-station', 2: 'airplane',
        3: 'airport', 4: 'Baseball', 5: 'Basketball',
        6: 'bridge', 7: 'chimney', 8: 'dam', 9: 'Golf',
        10: 'Soccer', 11: 'harbor', 12: 'overpass',
        13: 'ship', 14: 'Stadium', 15: 'storagetank',
        16: 'Tennis', 17: 'trainstation', 18: 'vehicle',
        19: 'windmill'
    }
    target_sport_class_ids = [4, 5, 9, 10, 14, 16]

    os.makedirs(obd_image_path, exist_ok=True)
    
    # Get database IDs if not provided
    if db_ids_df is None or db_ids_df.empty:
        print("Fetching nge_object IDs from database for object detection...")
        db_ids_df = get_nge_object_ids_from_database()
        if db_ids_df.empty:
            print("Warning: No nge_object IDs found in database.")
            return pd.DataFrame(columns=['field_search_id', 'nge_object_id', 'detected_sport', 'detect_confidence']), \
                   pd.DataFrame(columns=['field_search_id', 'nge_object_id', 'sport_name', 'detect_confidence'])
    
    # Match images to database IDs
    matched_images = _match_image_to_database_ids(large_image_dir_path, db_ids_df)
    if not matched_images:
        print(f"No matching images found in {large_image_dir_path} for database records")
        return pd.DataFrame(columns=['field_search_id', 'nge_object_id', 'detected_sport', 'detect_confidence']), \
               pd.DataFrame(columns=['field_search_id', 'nge_object_id', 'sport_name', 'detect_confidence'])

    df_obj_detection = pd.DataFrame(columns=[
        'field_search_id', 'nge_object_id', 'detected_sport', 'detect_confidence'
    ])
    df_obj_detection_individual = pd.DataFrame(columns=[
        'field_search_id', 'nge_object_id', 'sport_name', 'detect_confidence'
    ])

    print(f"[YOLO] Processing {len(matched_images)} matched images for object detection...")
    for image_path, field_search_id, nge_object_id, sport_name in matched_images:
        img_filename = os.path.basename(image_path)
        print(f"[YOLO] Processing image: {img_filename} (field_search_id={field_search_id}, nge_object_id={nge_object_id})", flush=True)

        results_generator = model.predict(
            source=image_path,
            save=False,
            conf=CONFIDENCE_THR,
            iou=0.65,
            stream=True
        )

        detected_sports_for_image = []
        detect_confidence_for_image = []

        for result in results_generator:
            img = result.orig_img.copy()
            h, w, _ = img.shape

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)

                    if confidence < CONFIDENCE_THR:
                        continue
                    if class_id not in target_sport_class_ids:
                        continue

                    label = display_names.get(class_id, model.names[class_id])

                    df_obj_detection_individual = pd.concat([
                        df_obj_detection_individual,
                        pd.DataFrame({
                            'field_search_id': field_search_id,
                            'nge_object_id': nge_object_id,
                            'sport_name': label,
                            'detect_confidence': confidence
                        }, index=[0])
                    ], ignore_index=True)

                    detected_sports_for_image.append(label)
                    detect_confidence_for_image.append(confidence)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    text = f"{label} {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

                    text_x = x1
                    text_y = y1 - 5

                    if text_y - text_size[1] - 10 < 0:
                        text_y = y1 + text_size[1] + 15
                        if text_y + 5 > h:
                            text_y = y1 + text_size[1] + 5
                            if text_y > y2:
                                text_y = y1 + text_size[1] + 5

                    if text_x < 0:
                        text_x = 0

                    bg_x1 = text_x
                    bg_y1 = text_y - text_size[1] - 10
                    bg_x2 = text_x + text_size[0]
                    bg_y2 = text_y

                    bg_x1 = max(0, bg_x1)
                    bg_y1 = max(0, bg_y1)
                    bg_x2 = min(w, bg_x2)
                    bg_y2 = min(h, bg_y2)

                    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                    adjusted_text_y = bg_y1 + text_size[1] + 5
                    cv2.putText(
                        img, text, (text_x, adjusted_text_y), font,
                        font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
                    )
            else:
                print(f"No detections found for image: {img_filename}.")

            # Save annotated image for this result (use original filename)
            output_image_path = os.path.join(obd_image_path, img_filename)
            cv2.imwrite(output_image_path, img)
            print(f"Saved detection image to: {output_image_path}")

            # Summary row for this image
            if detected_sports_for_image:
                sports_unique = list(set(detected_sports_for_image))
                conf_unique = list(set(detect_confidence_for_image))
                detected_sports_str = ', '.join(sports_unique)
                confidence_str = ', '.join(f"{c:.2f}" for c in conf_unique)
            else:
                detected_sports_str = 'None'
                confidence_str = 'None'

            df_obj_detection = pd.concat([
                df_obj_detection,
                pd.DataFrame({
                    'field_search_id': field_search_id,
                    'nge_object_id': nge_object_id,
                    'detected_sport': detected_sports_str,
                    'detect_confidence': confidence_str
                }, index=[0])
            ], ignore_index=True)

            del img
            gc.collect()

    print(f"All object detection visualizations saved to: {obd_images_path}")
    return df_obj_detection, df_obj_detection_individual


# -----------------------------------------------------------------------------
# Database Functions
# -----------------------------------------------------------------------------

def get_nge_object_ids_from_database():
    """
    Fetch all nge_object_id and field_search_id pairs from nge_object table.
    Returns a DataFrame with columns: nge_object_id, field_search_id, sport_name
    """
    conn = None
    cur = None
    
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        query = """
            SELECT 
                nge_object_id,
                field_search_id,
                sport_name
            FROM public.nge_object
            WHERE nge_object_id IS NOT NULL
            ORDER BY nge_object_id, field_search_id
        """
        cur.execute(query)
        rows = cur.fetchall()
        
        df = pd.DataFrame(rows, columns=['nge_object_id', 'field_search_id', 'sport_name'])
        return df
        
    except Exception as error:
        print(f"Error fetching nge_object IDs from database: {error}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            pool.putconn(conn)


def _match_image_to_database_ids(image_dir_path: str, db_ids_df: pd.DataFrame):
    """
    Match image files to database IDs.
    Images are named as: {nge_object_id}_{field_search_id}_{sport_name}.png
    Returns a list of tuples: (image_path, field_search_id, nge_object_id, sport_name)
    """
    image_files = _list_image_files(image_dir_path)
    matched_images = []
    
    # Create a mapping from (nge_object_id, field_search_id) to sport_name
    db_mapping = {}
    for _, row in db_ids_df.iterrows():
        key = (int(row['nge_object_id']), int(row['field_search_id']))
        db_mapping[key] = row['sport_name']
    
    for img_file in image_files:
        # Parse filename: nge_object_id_field_search_id_sportname.png
        base = os.path.splitext(img_file)[0]
        parts = base.split("_")
        
        if len(parts) >= 2:
            try:
                nge_obj_id = int(float(parts[0]))
                field_search_id = int(float(parts[1]))
                
                # Check if this (nge_object_id, field_search_id) exists in database
                key = (nge_obj_id, field_search_id)
                if key in db_mapping:
                    image_path = os.path.join(image_dir_path, img_file)
                    sport_name = db_mapping[key]
                    matched_images.append((image_path, field_search_id, nge_obj_id, sport_name))
            except (ValueError, TypeError):
                # Skip files that don't match expected format
                continue
    
    return matched_images


def get_database_fields(field_search_ids=None, nge_object_ids=None):
    """
    Fetch all database fields from new_google_earth and nge_object tables.
    If field_search_ids or nge_object_ids are provided, filter by them.
    Otherwise, fetch all records that have corresponding nge_object entries.
    """
    conn = None
    cur = None
    
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        
        # Build the query to get all relevant fields
        if field_search_ids is not None and len(field_search_ids) > 0 and nge_object_ids is not None and len(nge_object_ids) > 0:
            # Filter by both IDs
            query = """
                SELECT 
                    nge.field_search_id,
                    nge.field_name,
                    nge.object_sport,
                    nge.formatted_address,
                    nge.postal_code,
                    nge.street,
                    nge.city,
                    nge.state,
                    nge.gps_location,
                    nge.gplace_id,
                    nge.search_sport_type,
                    nge.gearth_link,
                    nge.object_gearth_link,
                    nge.object_gps_location,
                    obj.nge_object_id,
                    obj.sport_name,
                    obj.image_url,
                    obj.description
                FROM public.new_google_earth nge
                LEFT JOIN public.nge_object obj ON obj.field_search_id = nge.field_search_id
                WHERE (nge.field_search_id = ANY(%s) OR obj.nge_object_id = ANY(%s))
                ORDER BY nge.field_search_id, obj.nge_object_id
            """
            cur.execute(query, (field_search_ids, nge_object_ids))
        elif field_search_ids is not None and len(field_search_ids) > 0:
            # Filter by field_search_id only
            query = """
                SELECT 
                    nge.field_search_id,
                    nge.field_name,
                    nge.object_sport,
                    nge.formatted_address,
                    nge.postal_code,
                    nge.street,
                    nge.city,
                    nge.state,
                    nge.gps_location,
                    nge.gplace_id,
                    nge.search_sport_type,
                    nge.gearth_link,
                    nge.object_gearth_link,
                    nge.object_gps_location,
                    obj.nge_object_id,
                    obj.sport_name,
                    obj.image_url,
                    obj.description
                FROM public.new_google_earth nge
                LEFT JOIN public.nge_object obj ON obj.field_search_id = nge.field_search_id
                WHERE nge.field_search_id = ANY(%s)
                ORDER BY nge.field_search_id, obj.nge_object_id
            """
            cur.execute(query, (field_search_ids,))
        elif nge_object_ids is not None and len(nge_object_ids) > 0:
            # Filter by nge_object_id only
            query = """
                SELECT 
                    nge.field_search_id,
                    nge.field_name,
                    nge.object_sport,
                    nge.formatted_address,
                    nge.postal_code,
                    nge.street,
                    nge.city,
                    nge.state,
                    nge.gps_location,
                    nge.gplace_id,
                    nge.search_sport_type,
                    nge.gearth_link,
                    nge.object_gearth_link,
                    nge.object_gps_location,
                    obj.nge_object_id,
                    obj.sport_name,
                    obj.image_url,
                    obj.description
                FROM public.new_google_earth nge
                LEFT JOIN public.nge_object obj ON obj.field_search_id = nge.field_search_id
                WHERE obj.nge_object_id = ANY(%s)
                ORDER BY nge.field_search_id, obj.nge_object_id
            """
            cur.execute(query, (nge_object_ids,))
        else:
            # Get all records that have nge_object entries
            query = """
                SELECT 
                    nge.field_search_id,
                    nge.field_name,
                    nge.object_sport,
                    nge.formatted_address,
                    nge.postal_code,
                    nge.street,
                    nge.city,
                    nge.state,
                    nge.gps_location,
                    nge.gplace_id,
                    nge.search_sport_type,
                    nge.gearth_link,
                    nge.object_gearth_link,
                    nge.object_gps_location,
                    obj.nge_object_id,
                    obj.sport_name,
                    obj.image_url,
                    obj.description
                FROM public.new_google_earth nge
                INNER JOIN public.nge_object obj ON obj.field_search_id = nge.field_search_id
                ORDER BY nge.field_search_id, obj.nge_object_id
            """
            cur.execute(query)
        
        rows = cur.fetchall()
        
        # Create DataFrame with all columns
        df = pd.DataFrame(rows, columns=[
            'field_search_id',
            'field_name',
            'object_sport',
            'formatted_address',
            'postal_code',
            'street',
            'city',
            'state',
            'gps_location',
            'gplace_id',
            'search_sport_type',
            'gearth_link',
            'object_gearth_link',
            'object_gps_location',
            'nge_object_id',
            'sport_name',
            'image_url',
            'description'
        ])
        
        return df
        
    except Exception as error:
        print(f"Error fetching database fields: {error}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            pool.putconn(conn)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Reset YOLO output folder
    if os.path.exists(obd_images_path):
        shutil.rmtree(obd_images_path)
    os.makedirs(obd_images_path, exist_ok=True)
    print("Previous detection images cleared.")

    # Get database IDs once at the start
    print("Fetching nge_object IDs from database...")
    db_ids_df = get_nge_object_ids_from_database()
    if db_ids_df.empty:
        print("Error: No nge_object IDs found in database. Cannot proceed.")
        exit(1)
    print(f"Found {len(db_ids_df)} nge_object records in database")
    
    # Classification
    print("Starting Classification...")
    df_small = Classification(small_img_pth, cls_sport_model_path, cls_type_model_path, db_ids_df)
    df_large = Classification_large(large_img_pth, cls_sport_model_path, cls_type_model_path, db_ids_df)

    if df_small.empty or df_large.empty:
        print("Warning: one of the classification DataFrames is empty.")

    # Merge small and large classifications; use outer to keep all rows from either set
    if not df_small.empty and not df_large.empty:
        classification_results_df = pd.merge(
            df_small,
            df_large,
            on=['field_search_id', 'nge_object_id'],
            how='outer'
        )
    elif not df_small.empty:
        classification_results_df = df_small.copy()
        # Add large-only columns as NaN so downstream expects them
        for c in ['predicted_large_field_type', 'predicted_large_sport',
                  'large_predicted_field_type_probability', 'large_predicted_sport_type_probability']:
            if c not in classification_results_df.columns:
                classification_results_df[c] = np.nan
    elif not df_large.empty:
        classification_results_df = df_large.copy()
        for c in ['predicted_field_type', 'predicted_sport',
                  'predicted_field_type_probability', 'predicted_sport_type_probability']:
            if c not in classification_results_df.columns:
                classification_results_df[c] = np.nan
    else:
        classification_results_df = pd.DataFrame()

    print("\nClassification Results (merged small+large):")
    print(classification_results_df.head())
    print(f"Classification results entries: {len(classification_results_df)}")

    # Object detection
    print("\nStarting Object Detection...")
    object_detection_results_df, object_detection_individual_results_df = Object_Detection(
        large_img_pth, obd_model_path, obd_images_path, db_ids_df
    )
    print("\nObject Detection Results (per image):")
    print(object_detection_results_df.head())
    print(f"Object detection entries: {len(object_detection_results_df)}")

    # Final merge of classification and detection; use left to keep all classification rows
    if not classification_results_df.empty and not object_detection_results_df.empty:
        combined_df = pd.merge(
            classification_results_df,
            object_detection_results_df,
            on=['field_search_id', 'nge_object_id'],
            how='left'
        )
        # Ensure detection columns exist (NaN where no match)
        if 'detected_sport' not in combined_df.columns:
            combined_df['detected_sport'] = np.nan
        if 'detect_confidence' not in combined_df.columns:
            combined_df['detect_confidence'] = np.nan
        print("\nCombined Results (classification + detection):")
        print(combined_df.head())
    elif not classification_results_df.empty:
        combined_df = classification_results_df.copy()
        combined_df['detected_sport'] = np.nan
        combined_df['detect_confidence'] = np.nan
        print("\nUsing classification results only (no detection results)")
    elif not object_detection_results_df.empty:
        combined_df = object_detection_results_df.copy()
        for c in ['predicted_field_type', 'predicted_sport', 'predicted_field_type_probability',
                  'predicted_sport_type_probability', 'predicted_large_field_type', 'predicted_large_sport',
                  'large_predicted_field_type_probability', 'large_predicted_sport_type_probability']:
            if c not in combined_df.columns:
                combined_df[c] = np.nan
        print("\nUsing detection results only (no classification results)")
    else:
        combined_df = pd.DataFrame()
        print("\nWarning: No classification or detection results available.")
    
    # Fetch database fields and merge with results
    if not combined_df.empty:
        # Get unique field_search_ids and nge_object_ids from results
        field_search_ids_raw = combined_df['field_search_id'].dropna().unique().tolist()
        nge_object_ids_raw = combined_df['nge_object_id'].dropna().unique().tolist()
        
        # Convert to list format for PostgreSQL ANY clause, ensuring they are integers
        field_search_ids = []
        if field_search_ids_raw:
            for x in field_search_ids_raw:
                try:
                    field_search_ids.append(int(x))
                except (ValueError, TypeError):
                    continue
        
        nge_object_ids = []
        if nge_object_ids_raw:
            for x in nge_object_ids_raw:
                try:
                    nge_object_ids.append(int(x))
                except (ValueError, TypeError):
                    continue
        
        print(f"\nFetching database fields for {len(field_search_ids)} field_search_ids and {len(nge_object_ids)} nge_object_ids...")
        db_df = get_database_fields(
            field_search_ids=field_search_ids if field_search_ids else None,
            nge_object_ids=nge_object_ids if nge_object_ids else None
        )
        
        if not db_df.empty:
            print(f"Fetched {len(db_df)} database records")
            print("Database fields sample:")
            print(db_df.head())
            
            # Ensure ID columns are the same type for proper merging
            for col in ['field_search_id', 'nge_object_id']:
                if col in combined_df.columns:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                if col in db_df.columns:
                    db_df[col] = pd.to_numeric(db_df[col], errors='coerce')
            
            # Merge database fields with combined results
            # Use left join to keep all classification/detection results and add database fields
            final_df = pd.merge(
                combined_df,
                db_df,
                on=['field_search_id', 'nge_object_id'],
                how='left',
                suffixes=('', '_db')
            )
            
            # Remove duplicate columns (keep the ones from combined_df if they exist)
            # If there are _db suffixed columns, we might want to keep the db version for some fields
            # For now, we'll keep the original columns and drop _db suffixed ones if they're exact duplicates
            cols_to_drop = []
            for col in final_df.columns:
                if col.endswith('_db'):
                    base_col = col[:-3]
                    if base_col in final_df.columns:
                        # If the base column has all nulls but _db has values, keep _db
                        if final_df[base_col].isna().all() and not final_df[col].isna().all():
                            final_df[base_col] = final_df[col]
                        cols_to_drop.append(col)
            
            if cols_to_drop:
                final_df = final_df.drop(columns=cols_to_drop)
            
            print(f"Merged results: {len(final_df)} records")
        else:
            print("Warning: No database fields found. Using classification/detection results only.")
            final_df = combined_df.copy()
    else:
        # If no classification/detection results, try to get all database records
        print("\nNo classification/detection results. Fetching all database fields...")
        final_df = get_database_fields()
    
    # Prepare final output - Only include columns needed for updating nge_object table
    if not final_df.empty:
        # Define columns needed for updating nge_object table (from update_database.py)
        # These are the columns that will be used to update the nge_object table
        columns_for_nge_object_update = [
            'nge_object_id',  # Required ID column for matching records
            'field_search_id',  # Keep for reference
            'predicted_field_type',
            'predicted_sport',
            'predicted_field_type_probability',
            'predicted_sport_type_probability',
            'predicted_large_field_type',
            'predicted_large_sport',
            'large_predicted_field_type_probability',
            'large_predicted_sport_type_probability',
            'detected_sport',
            'detect_confidence'
        ]
        
        # Filter to only include columns that exist in final_df and are needed for update
        available_columns = [col for col in columns_for_nge_object_update if col in final_df.columns]
        missing = [c for c in columns_for_nge_object_update if c not in final_df.columns]
        if missing:
            print(f"Warning: Expected columns missing from results (will be omitted): {missing}")
        
        # Create filtered DataFrame with only the required columns
        final_df = final_df[available_columns].copy()
        
        # Coerce confidence/probability columns to float so Excel has proper numeric values
        numeric_cols = [
            'predicted_field_type_probability', 'predicted_sport_type_probability',
            'large_predicted_field_type_probability', 'large_predicted_sport_type_probability'
        ]
        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        # detect_confidence can be comma-separated string; keep as-is for DB update
        if 'detect_confidence' in final_df.columns:
            pass  # leave as string e.g. "0.87, 0.92"
        
        # Ensure nge_object_id is present (required for updates)
        if 'nge_object_id' not in final_df.columns:
            print("Warning: nge_object_id column is missing. Cannot update nge_object table without it.")
        
        print("\nFinal Results (columns for nge_object table update):")
        print(final_df.head())
        print(f"\nTotal records: {len(final_df)}")
        print(f"Columns included: {list(final_df.columns)}")
        print(f"\nNote: Only columns needed for updating nge_object table are included.")

        # Write to script directory so output is easy to find (e.g. combined_results_2026-01-22T10-33-30.xlsx)
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        output_excel_filename = os.path.join(_script_dir, f"combined_results_{NOW}.xlsx")
        output_individual_excel_filename = os.path.join(_script_dir, f"obd_individual_results_{NOW}.xlsx")
        try:
            final_df.to_excel(output_excel_filename, index=False)
            print(f"\nCombined results (columns for nge_object table update) saved to {output_excel_filename}")
            
            if not object_detection_individual_results_df.empty:
                object_detection_individual_results_df.to_excel(
                    output_individual_excel_filename, index=False
                )
                print(f"Individual detection results saved to {output_individual_excel_filename}")
        except Exception as e:
            print(f"Error saving results to Excel: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nError: Final DataFrame is empty. Cannot save to Excel.")
