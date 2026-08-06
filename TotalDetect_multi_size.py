import collections
import datetime
import gc
import os
import shutil
from configparser import ConfigParser
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_keras.models import load_model
from ultralytics import YOLO
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


class _SavedModelPredictor:
    def __init__(self, model_path: str):
        self._model = tf.saved_model.load(model_path)
        signature_names = list(self._model.signatures.keys())
        if not signature_names:
            raise ValueError(f"No signatures found in SavedModel: {model_path}")
        self._signature = self._model.signatures.get("serving_default", self._model.signatures[signature_names[0]])
        self._input_name = next(iter(self._signature.structured_input_signature[1].keys()))
        structured_outputs = self._signature.structured_outputs
        self._output_name = next(iter(structured_outputs.keys())) if isinstance(structured_outputs, dict) else None

    def predict(self, batch_images, verbose=0):
        del verbose
        batch_images = np.asarray(batch_images, dtype=np.float32)
        outputs = self._signature(**{self._input_name: batch_images})
        if isinstance(outputs, dict):
            output = outputs.get(self._output_name, next(iter(outputs.values())))
        else:
            output = outputs
        return output.numpy()


def _load_compat_model(model_path: str):
    try:
        return load_model(model_path)
    except Exception as exc:
        print(f"[WARN] Falling back to SavedModel inference wrapper for {model_path}: {exc}")
        return _SavedModelPredictor(model_path)


def _list_image_files(directory: str):
    if not os.path.isdir(directory):
        print(f"Warning: directory does not exist: {directory}")
        return []
    exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    return [f for f in os.listdir(directory) if f.lower().endswith(exts)]


def get_nge_object_ids_from_database():
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        query = """
            SELECT nge_object_id, field_search_id, sport_name
            FROM public.nge_object
            WHERE nge_object_id IS NOT NULL
            ORDER BY nge_object_id, field_search_id
        """
        cur.execute(query)
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=['nge_object_id', 'field_search_id', 'sport_name'])
    except Exception as error:
        print(f"Error fetching nge_object IDs from database: {error}")
        return pd.DataFrame()
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)


def _match_image_to_database_ids(image_dir_path: str, db_ids_df: pd.DataFrame):
    """
    Robust image matcher that handles:
      - {nge_object_id}_{field_search_id}_{sport_name}.png
      - {field_search_id}_{nge_object_id}_{sport_name}.png
      - orig_{field_search_id}_{sport_name}.png
      - {field_search_id}_{sport_name}.png
    Falls back to field_search_id matching so new nge_object records receive predictions.
    """
    image_files = _list_image_files(image_dir_path)
    if db_ids_df is None or db_ids_df.empty or not image_files:
        return []

    exact_map = {}
    field_map = collections.defaultdict(list)
    object_map = collections.defaultdict(list)

    for _, row in db_ids_df.iterrows():
        try:
            nge_id = int(float(row['nge_object_id']))
            field_id = int(float(row['field_search_id']))
            sport = str(row['sport_name']) if pd.notna(row['sport_name']) else 'Sports Facility'
            exact_map[(nge_id, field_id)] = sport
            field_map[field_id].append((nge_id, sport))
            object_map[nge_id].append((field_id, sport))
        except (ValueError, TypeError):
            continue

    matched_set = set()

    for img_file in image_files:
        image_path = os.path.join(image_dir_path, img_file)
        base = os.path.splitext(img_file)[0]
        parts = base.split("_")

        nums = []
        for p in parts:
            if p.lower() == 'orig':
                continue
            try:
                nums.append(int(float(p)))
            except (ValueError, TypeError):
                continue

        if len(nums) >= 2:
            n1, n2 = nums[0], nums[1]
            if (n1, n2) in exact_map:
                matched_set.add((image_path, n2, n1, exact_map[(n1, n2)]))
            elif (n2, n1) in exact_map:
                matched_set.add((image_path, n1, n2, exact_map[(n2, n1)]))
            else:
                for fid in (n2, n1):
                    if fid in field_map:
                        for nid, sport in field_map[fid]:
                            matched_set.add((image_path, fid, nid, sport))
        elif len(nums) == 1:
            num = nums[0]
            if num in field_map:
                for nid, sport in field_map[num]:
                    matched_set.add((image_path, num, nid, sport))
            elif num in object_map:
                for fid, sport in object_map[num]:
                    matched_set.add((image_path, fid, num, sport))

    return list(matched_set)


def Classification(small_image_dir_path: str, cls_sport_model_path: str, cls_type_model_path: str, db_ids_df: pd.DataFrame = None) -> pd.DataFrame:
    if db_ids_df is None or db_ids_df.empty:
        db_ids_df = get_nge_object_ids_from_database()
    matched_images = _match_image_to_database_ids(small_image_dir_path, db_ids_df)
    if not matched_images:
        return pd.DataFrame(columns=['field_search_id', 'nge_object_id', 'predicted_field_type', 'predicted_sport', 'predicted_field_type_probability', 'predicted_sport_type_probability'])

    type_model = _load_compat_model(cls_type_model_path)
    sport_model = _load_compat_model(cls_sport_model_path)

    num_images = len(matched_images)
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE
    records = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_images)
        batch_items = matched_images[start_idx:end_idx]

        batch_images, batch_ids = [], []
        for image_path, field_search_id, nge_object_id, _ in batch_items:
            img = cv2.imread(image_path)
            if img is None: continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            batch_images.append(img)
            batch_ids.append((field_search_id, nge_object_id))

        if not batch_images: continue
        batch_images = np.array(batch_images, dtype="uint8")

        pred_type = type_model.predict(batch_images, verbose=0) if hasattr(type_model, 'predict') else type_model(batch_images)
        pred_sport = sport_model.predict(batch_images, verbose=0) if hasattr(sport_model, 'predict') else sport_model(batch_images)

        labels_type = np.argmax(pred_type, axis=1)
        labels_sport = np.argmax(pred_sport, axis=1)
        probs_type = np.max(pred_type, axis=1)
        probs_sport = np.max(pred_sport, axis=1)

        for i, (field_search_id, nge_object_id) in enumerate(batch_ids):
            c_type = TYPE_CLASSES[labels_type[i]]
            c_sport = SPORT_CLASSES[labels_sport[i]]
            if c_sport in ('Basketball', 'Tennis'): c_type = 'Synthetic'
            if c_sport == 'Buildings': c_type = 'Buildings'

            records.append({
                'field_search_id': field_search_id,
                'nge_object_id': nge_object_id,
                'predicted_field_type': c_type,
                'predicted_sport': c_sport,
                'predicted_field_type_probability': float(probs_type[i]),
                'predicted_sport_type_probability': float(probs_sport[i])
            })

    return pd.DataFrame(records).drop_duplicates(subset=['field_search_id', 'nge_object_id'])


def Classification_large(large_image_dir_path: str, cls_sport_model_path: str, cls_type_model_path: str, db_ids_df: pd.DataFrame = None) -> pd.DataFrame:
    if db_ids_df is None or db_ids_df.empty:
        db_ids_df = get_nge_object_ids_from_database()
    matched_images = _match_image_to_database_ids(large_image_dir_path, db_ids_df)
    if not matched_images:
        return pd.DataFrame(columns=['field_search_id', 'nge_object_id', 'predicted_large_field_type', 'predicted_large_sport', 'large_predicted_field_type_probability', 'large_predicted_sport_type_probability'])

    type_model = _load_compat_model(cls_type_model_path)
    sport_model = _load_compat_model(cls_sport_model_path)

    num_images = len(matched_images)
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE
    records = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_images)
        batch_items = matched_images[start_idx:end_idx]

        batch_images, batch_ids = [], []
        for image_path, field_search_id, nge_object_id, _ in batch_items:
            img = cv2.imread(image_path)
            if img is None: continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            batch_images.append(img)
            batch_ids.append((field_search_id, nge_object_id))

        if not batch_images: continue
        batch_images = np.array(batch_images, dtype="uint8")

        pred_type = type_model.predict(batch_images, verbose=0) if hasattr(type_model, 'predict') else type_model(batch_images)
        pred_sport = sport_model.predict(batch_images, verbose=0) if hasattr(sport_model, 'predict') else sport_model(batch_images)

        labels_type = np.argmax(pred_type, axis=1)
        labels_sport = np.argmax(pred_sport, axis=1)
        probs_type = np.max(pred_type, axis=1)
        probs_sport = np.max(pred_sport, axis=1)

        for i, (field_search_id, nge_object_id) in enumerate(batch_ids):
            c_type = TYPE_CLASSES[labels_type[i]]
            c_sport = SPORT_CLASSES[labels_sport[i]]
            if c_sport in ('Basketball', 'Tennis'): c_type = 'Synthetic'
            if c_sport == 'Buildings': c_type = 'Buildings'

            records.append({
                'field_search_id': field_search_id,
                'nge_object_id': nge_object_id,
                'predicted_large_field_type': c_type,
                'predicted_large_sport': c_sport,
                'large_predicted_field_type_probability': float(probs_type[i]),
                'large_predicted_sport_type_probability': float(probs_sport[i])
            })

    return pd.DataFrame(records).drop_duplicates(subset=['field_search_id', 'nge_object_id'])


def Object_Detection(large_image_dir_path: str, obd_model_path: str, obd_image_path: str, db_ids_df: pd.DataFrame = None):
    model = YOLO(obd_model_path)
    display_names = {4: 'Baseball', 5: 'Basketball', 9: 'Golf', 10: 'Soccer', 14: 'Stadium', 16: 'Tennis'}
    target_sport_class_ids = [4, 5, 9, 10, 14, 16]

    if db_ids_df is None or db_ids_df.empty:
        db_ids_df = get_nge_object_ids_from_database()
    matched_images = _match_image_to_database_ids(large_image_dir_path, db_ids_df)
    if not matched_images:
        return pd.DataFrame(columns=['field_search_id', 'nge_object_id', 'detected_sport', 'detect_confidence']), pd.DataFrame()

    summary_records = []
    individual_records = []

    for image_path, field_search_id, nge_object_id, _ in matched_images:
        results = model.predict(source=image_path, save=False, conf=CONFIDENCE_THR, iou=0.65, stream=True)
        sports_found, confs_found = [], []

        for res in results:
            if res.boxes is not None:
                for box in res.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    if conf >= CONFIDENCE_THR and cls_id in target_sport_class_ids:
                        label = display_names.get(cls_id, f"Class_{cls_id}")
                        sports_found.append(label)
                        confs_found.append(conf)
                        individual_records.append({
                            'field_search_id': field_search_id,
                            'nge_object_id': nge_object_id,
                            'sport_name': label,
                            'detect_confidence': conf
                        })

        summary_records.append({
            'field_search_id': field_search_id,
            'nge_object_id': nge_object_id,
            'detected_sport': ', '.join(set(sports_found)) if sports_found else 'None',
            'detect_confidence': ', '.join(f"{c:.2f}" for c in set(confs_found)) if confs_found else 'None'
        })

    df_sum = pd.DataFrame(summary_records).drop_duplicates(subset=['field_search_id', 'nge_object_id'])
    df_ind = pd.DataFrame(individual_records)
    return df_sum, df_ind


def get_database_fields():
    conn, cur = None, None
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        query = """
            SELECT 
                obj.nge_object_id,
                nge.field_search_id,
                nge.field_name,
                obj.sport_name,
                obj.description
            FROM public.nge_object obj
            INNER JOIN public.new_google_earth nge ON obj.field_search_id = nge.field_search_id
            ORDER BY obj.nge_object_id;
        """
        cur.execute(query)
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=['nge_object_id', 'field_search_id', 'field_name', 'sport_name', 'description'])
    except Exception as e:
        print(f"Error fetching database fields: {e}")
        return pd.DataFrame()
    finally:
        if cur: cur.close()
        if conn: pool.putconn(conn)


if __name__ == '__main__':
    os.makedirs(obd_images_path, exist_ok=True)
    db_ids_df = get_nge_object_ids_from_database()

    df_small = Classification(small_img_pth, cls_sport_model_path, cls_type_model_path, db_ids_df)
    df_large = Classification_large(large_img_pth, cls_sport_model_path, cls_type_model_path, db_ids_df)

    # Outer merge classifications
    if not df_small.empty and not df_large.empty:
        cls_df = pd.merge(df_small, df_large, on=['field_search_id', 'nge_object_id'], how='outer')
    else:
        cls_df = df_small if not df_small.empty else df_large

    # Outer merge object detections
    obj_df, obj_ind_df = Object_Detection(large_img_pth, obd_model_path, obd_images_path, db_ids_df)
    if not cls_df.empty and not obj_df.empty:
        combined_df = pd.merge(cls_df, obj_df, on=['field_search_id', 'nge_object_id'], how='outer')
    else:
        combined_df = cls_df if not cls_df.empty else obj_df

    # Fetch full database nge_object records and merge
    db_df = get_database_fields()
    if not db_df.empty:
        if not combined_df.empty:
            for col in ['field_search_id', 'nge_object_id']:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                db_df[col] = pd.to_numeric(db_df[col], errors='coerce')

            final_df = pd.merge(db_df, combined_df, on=['field_search_id', 'nge_object_id'], how='left')
        else:
            final_df = db_df.copy()
    else:
        final_df = combined_df

    cols_for_update = [
        'nge_object_id',
        'field_search_id',
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

    for col in cols_for_update:
        if col not in final_df.columns:
            final_df[col] = np.nan

    final_df = final_df[cols_for_update].copy()
    final_df = final_df.drop_duplicates(subset=['nge_object_id'])

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    output_excel = os.path.join(_script_dir, f"combined_results_{NOW}.xlsx")
    final_df.to_excel(output_excel, index=False)
    print(f"Successfully processed predictions for all records. Output saved to {output_excel}")