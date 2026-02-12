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
    Expected filename pattern (without extension):
      nge_object_id_field_search_id_sport_name

    Example:
      1234_5678_Soccer.png
      -> nge_object_id = 1234
         field_search_id = 5678

    Returns: (field_search_id, nge_object_id)
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    field_search_id = None
    nge_object_id = None

    # nge_object_id is the first part
    if len(parts) > 0:
        try:
            nge_object_id = int(parts[0])
        except ValueError:
            nge_object_id = None

    # field_search_id is the second part
    if len(parts) > 1:
        try:
            field_search_id = int(parts[1])
        except ValueError:
            field_search_id = None

    return field_search_id, nge_object_id

# -----------------------------------------------------------------------------
# Classification on small images
# -----------------------------------------------------------------------------

def Classification(small_image_dir_path: str,
                   cls_sport_model_path: str,
                   cls_type_model_path: str) -> pd.DataFrame:
    image_files = _list_image_files(small_image_dir_path)
    if not image_files:
        print(f"No small images found in {small_image_dir_path}")
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
    print('Finished loading classification models for SMALL images.', flush=True)

    num_images = len(image_files)
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
        batch_files = image_files[start_idx:end_idx]

        batch_images = []
        batch_ids = []

        for image_file in batch_files:
            image_path = os.path.join(small_image_dir_path, image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: could not read image {image_path}, skipping.")
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            batch_images.append(img)
            batch_ids.append(_parse_ids_from_filename(image_file))

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
                         cls_type_model_path: str) -> pd.DataFrame:
    image_files = _list_image_files(large_image_dir_path)
    if not image_files:
        print(f"No large images found in {large_image_dir_path}")
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
    print('Finished loading classification models for LARGE images.', flush=True)

    num_images = len(image_files)
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
        batch_files = image_files[start_idx:end_idx]

        batch_images = []
        batch_ids = []

        for image_file in batch_files:
            image_path = os.path.join(large_image_dir_path, image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: could not read image {image_path}, skipping.")
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            batch_images.append(img)
            batch_ids.append(_parse_ids_from_filename(image_file))

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
                     obd_image_path: str):
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
    image_files = _list_image_files(large_image_dir_path)

    df_obj_detection = pd.DataFrame(columns=[
        'field_search_id', 'nge_object_id', 'detected_sport', 'detect_confidence'
    ])
    df_obj_detection_individual = pd.DataFrame(columns=[
        'field_search_id', 'nge_object_id', 'sport_name', 'detect_confidence'
    ])

    for img_filename in image_files:
        image_path = os.path.join(large_image_dir_path, img_filename)
        print(f"[YOLO] Processing image: {img_filename}", flush=True)

        results_generator = model.predict(
            source=image_path,
            save=False,
            conf=CONFIDENCE_THR,
            iou=0.65,
            stream=True
        )

        field_search_id, nge_object_id = _parse_ids_from_filename(img_filename)
        if field_search_id is None:
            print(f"Warning: could not parse IDs from {img_filename}, skipping.")
            continue

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

            output_image_path = os.path.join(obd_image_path, img_filename)
            cv2.imwrite(output_image_path, img)
            print(f"Saved detection image to: {output_image_path}")

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
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Reset YOLO output folder
    if os.path.exists(obd_images_path):
        shutil.rmtree(obd_images_path)
    os.makedirs(obd_images_path, exist_ok=True)
    print("Previous detection images cleared.")

    # Classification
    print("Starting Classification...")
    df_small = Classification(small_img_pth, cls_sport_model_path, cls_type_model_path)
    df_large = Classification_large(large_img_pth, cls_sport_model_path, cls_type_model_path)

    if df_small.empty or df_large.empty:
        print("Warning: one of the classification DataFrames is empty.")

    # Merge small and large classifications on both ids
    if not df_small.empty and not df_large.empty:
        classification_results_df = pd.merge(
            df_small,
            df_large,
            on=['field_search_id', 'nge_object_id'],
            how='inner'
        )
    else:
        classification_results_df = pd.DataFrame()

    print("\nClassification Results (merged small+large):")
    print(classification_results_df.head())
    print(f"Classification results entries: {len(classification_results_df)}")

    # Object detection
    print("\nStarting Object Detection...")
    object_detection_results_df, object_detection_individual_results_df = Object_Detection(
        large_img_pth, obd_model_path, obd_images_path
    )
    print("\nObject Detection Results (per image):")
    print(object_detection_results_df.head())
    print(f"Object detection entries: {len(object_detection_results_df)}")

    # Final merge
    if not classification_results_df.empty and not object_detection_results_df.empty:
        combined_df = pd.merge(
            classification_results_df,
            object_detection_results_df,
            on=['field_search_id', 'nge_object_id'],
            how='inner'
        )
        print("\nCombined Results (classification + detection):")
        print(combined_df.head())

        output_excel_filename = f"combined_results_{NOW}.xlsx"
        output_individual_excel_filename = f"obd_individual_results_{NOW}.xlsx"
        try:
            combined_df.to_excel(output_excel_filename, index=False)
            print(f"\nCombined results saved to {output_excel_filename}")
            object_detection_individual_results_df.to_excel(
                output_individual_excel_filename, index=False
            )
            print(f"Individual detection results saved to {output_individual_excel_filename}")
        except Exception as e:
            print(f"Error saving results to Excel: {e}")
    else:
        print("\nCannot perform final merge: one or both DataFrames are empty.")
