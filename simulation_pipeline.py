import argparse
import datetime as dt
import json
import os
import shutil
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from configparser import ConfigParser
from pyproj import Transformer
from tf_keras.models import load_model
from ultralytics import YOLO


KEY = "AIzaSyC5cT2KgRuUuz51GQ71DvY8O8EtE"

SPORT_CLASSES = ["Buildings", "Soccer", "Baseball", "Tennis", "Basketball"]
TYPE_CLASSES = ["Grass", "Turf"]
CONFIDENCE_THR = 0.45

SMALL_IMG_WIDTH = 300
SMALL_IMG_HEIGHT = 350
LARGE_IMG_WIDTH = 700
LARGE_IMG_HEIGHT = 750

TARGET_CLASS_IDS = [4, 5, 9, 10, 14, 16]
DISPLAY_NAMES = {
    0: "Expressway-Service-area",
    1: "Expressway-toll-station",
    2: "airplane",
    3: "airport",
    4: "Baseball",
    5: "Basketball",
    6: "bridge",
    7: "chimney",
    8: "dam",
    9: "Golf",
    10: "Soccer",
    11: "harbor",
    12: "overpass",
    13: "ship",
    14: "Stadium",
    15: "storagetank",
    16: "Tennis",
    17: "trainstation",
    18: "vehicle",
    19: "windmill",
}


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _build_output_root(output_root: Optional[str], input_stem: str) -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base = output_root or os.path.join(_script_dir(), "simulation_outputs")
    return os.path.join(base, f"{input_stem}_{stamp}")


def _safe_mkdir(path: str, reset: bool = False) -> None:
    if reset and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _make_placeholder_image(width: int, height: int, title: str) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (45, 52, 66)
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (200, 200, 200), 2)
    lines = [line.strip() for line in title.split("\n") if line.strip()]
    y = 45
    for line in lines:
        cv2.putText(img, line[:60], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
        y += 32
    cv2.putText(
        img,
        "SIMULATION PLACEHOLDER",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _parse_google_earth_link(link: str) -> Tuple[str, str]:
    if not isinstance(link, str) or "@" not in link:
        return "Error", "Error"
    start_index = link.find("@") + 1
    first_comma_index = link.find(",", start_index)
    second_comma_index = link.find(",", first_comma_index + 1)
    if first_comma_index == -1 or second_comma_index == -1:
        return "Error", "Error"
    return link[start_index:first_comma_index], link[first_comma_index + 1:second_comma_index]


def _get_image_url(lat: float, lon: float, zoom: int, width: int, height: int) -> str:
    return (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?key={KEY}&center={lat},{lon}&zoom={zoom}&size={width}x{height}&maptype=satellite"
    )


def _fetch_satellite_image(lat: float, lon: float, zoom: int, width: int, height: int) -> Optional[np.ndarray]:
    try:
        response = requests.get(_get_image_url(lat, lon, zoom, width, height), timeout=60)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)
    except Exception as exc:
        print(f"[WARN] Failed to fetch satellite image for ({lat}, {lon}) at zoom {zoom}: {exc}")
        return None


def _save_cropped_image(img_array: np.ndarray, save_path: str, crop_bottom: int = 30) -> None:
    if img_array is None:
        raise ValueError("Cannot save a None image")
    height, width, _ = img_array.shape
    cropped = img_array[0:max(1, height - crop_bottom), 0:width]
    cv2.imwrite(save_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))


def _recenter_image(
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    original_center: Tuple[int, int],
    original_gps_location: Dict[str, float],
    zoom_level: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    new_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    offset_x = new_center[0] - original_center[0]
    offset_y = new_center[1] - original_center[1]

    earth_radius_meters = 6378137
    earth_circumference_meters = 40075017

    transformer_4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    center_x_3857, center_y_3857 = transformer_4326_to_3857.transform(
        original_gps_location["longitude"],
        original_gps_location["latitude"],
    )

    old_ground_resolution = (
        earth_circumference_meters
        * np.cos(np.radians(original_gps_location["latitude"]))
        / (256 * (2 ** zoom_level))
    )
    ground_resolution = (
        np.cos(original_gps_location["latitude"] * np.pi / 180)
        * 2
        * np.pi
        * earth_radius_meters
        / (256 * 2 ** zoom_level)
    )

    meters_offset_x = offset_x * ground_resolution
    meters_offset_y = -offset_y * ground_resolution
    new_x_3857 = center_x_3857 + meters_offset_x
    new_y_3857 = center_y_3857 + meters_offset_y

    transformer_3857_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    new_lon, new_lat = transformer_3857_to_4326.transform(new_x_3857, new_y_3857)

    new_coordinates = {"latitude": float(new_lat), "longitude": float(new_lon)}
    ground_resolutions = {
        "old_ground_resolution": float(old_ground_resolution),
        "new_ground_resolution": float(ground_resolution),
    }
    return new_coordinates, ground_resolutions


def _object_detection_based_modification_by_class(
    field_search_id: int,
    img_array: np.ndarray,
    obd_model: YOLO,
    original_gps_location: Dict[str, float],
    target_class_ids: List[int],
    zoom_level: int,
) -> List[Dict[str, object]]:
    try:
        results = obd_model.predict(source=img_array, save=False, conf=0.45, iou=0.65, stream=False)
    except Exception as exc:
        print(f"[WARN] YOLO recenter pass failed for field_search_id={field_search_id}: {exc}")
        return []

    new_gps_locations: List[Dict[str, object]] = []
    if results and results[0].boxes is not None:
        height, width, _ = img_array.shape
        original_center = (width // 2, height // 2)
        for box in results[0].boxes:
            class_id = int(box.cls)
            if class_id not in target_class_ids:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            new_coordinates, ground_resolutions = _recenter_image(
                x1, x2, y1, y2, original_center, original_gps_location, zoom_level
            )
            description = (
                f"box=({x1},{y1})-({x2},{y2}); "
                f"image={width}x{height}; "
                f"old_ground_resolution={ground_resolutions['old_ground_resolution']:.6f}; "
                f"new_ground_resolution={ground_resolutions['new_ground_resolution']:.6f}"
            )
            row = {
                "field_search_id": field_search_id,
                "class_id": class_id,
                "class_name": DISPLAY_NAMES.get(class_id, f"Unknown Field Type {class_id}"),
                "description": description,
                "latitude": new_coordinates["latitude"],
                "longitude": new_coordinates["longitude"],
            }
            new_gps_locations.append(row)

    return new_gps_locations


def _load_models(config: ConfigParser):
    obd_model_path = config.get("model_paths", "obd_model")
    cls_sport_model_path = config.get("model_paths", "sport_model")
    cls_type_model_path = config.get("model_paths", "type_model")

    print(f"Loading YOLO model from {obd_model_path}")
    obd_model = YOLO(obd_model_path, verbose=False)

    type_model = None
    sport_model = None
    try:
        print(f"Loading type classification model from {cls_type_model_path}")
        type_model = load_model(cls_type_model_path)
    except Exception as exc:
        print(f"[WARN] Type model could not be loaded: {exc}")

    try:
        print(f"Loading sport classification model from {cls_sport_model_path}")
        sport_model = load_model(cls_sport_model_path)
    except Exception as exc:
        print(f"[WARN] Sport model could not be loaded: {exc}")

    return obd_model, type_model, sport_model


def _build_source_dataframe(input_excel: str) -> pd.DataFrame:
    df = pd.read_excel(input_excel)
    df = df.copy()
    df = df[df["Lat"].notna() & df["Lon"].notna()].reset_index(drop=True)
    df.insert(0, "field_search_id", np.arange(1, len(df) + 1))
    df["field_name"] = df["Name of Facility"].astype(str)
    df["primary_sport"] = df["Primary Sport"].fillna("Unknown").astype(str)
    df["formatted_address"] = df["Address"].astype(str)
    df["postal_code"] = pd.to_numeric(df["ZIP"], errors="coerce")
    df["gearth_link"] = df.apply(
        lambda r: f"https://earth.google.com/web/@{r['Lat']},{r['Lon']},4.1972381a,15000d", axis=1
    )
    return df


def _classify_images(image_dir: str, type_model, sport_model, kind: str) -> pd.DataFrame:
    rows = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()

    if not image_files:
        return pd.DataFrame(columns=[
            "field_search_id",
            f"predicted_{kind}_field_type",
            f"predicted_{kind}_sport",
            f"{kind}_predicted_field_type_probability",
            f"{kind}_predicted_sport_type_probability",
            f"{kind}_image_path",
        ])

    if type_model is None or sport_model is None:
        for img_file in image_files:
            path = os.path.join(image_dir, img_file)
            base = os.path.splitext(img_file)[0]
            parts = base.split("_")
            try:
                field_search_id = int(float(parts[0]))
            except Exception:
                continue
            rows.append({
                "field_search_id": field_search_id,
                f"predicted_{kind}_field_type": "Unavailable",
                f"predicted_{kind}_sport": "Unavailable",
                f"{kind}_predicted_field_type_probability": np.nan,
                f"{kind}_predicted_sport_type_probability": np.nan,
                f"{kind}_image_path": path,
            })
        return pd.DataFrame(rows)

    batch_images = []
    batch_meta = []
    for img_file in image_files:
        path = os.path.join(image_dir, img_file)
        base = os.path.splitext(img_file)[0]
        parts = base.split("_")
        try:
            field_search_id = int(float(parts[0]))
        except Exception:
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read image {path}")
            continue
        img = cv2.resize(img, (224, 224))
        batch_images.append(img)
        batch_meta.append((field_search_id, path))

    if not batch_images:
        return pd.DataFrame()

    batch_images = np.array(batch_images, dtype="uint8")
    predicted_field_type_batch = type_model.predict(batch_images, verbose=0)
    predicted_sport_batch = sport_model.predict(batch_images, verbose=0)

    labels_type = np.argmax(predicted_field_type_batch, axis=1)
    labels_sport = np.argmax(predicted_sport_batch, axis=1)
    field_type_probabilities = np.max(predicted_field_type_batch, axis=1)
    sport_type_probabilities = np.max(predicted_sport_batch, axis=1)

    for i, (field_search_id, path) in enumerate(batch_meta):
        classified_field_type = TYPE_CLASSES[labels_type[i]]
        classified_sport = SPORT_CLASSES[labels_sport[i]]
        if classified_sport in ("Basketball", "Tennis"):
            classified_field_type = "Synthetic"
        if classified_sport == "Buildings":
            classified_field_type = "Buildings"

        rows.append({
            "field_search_id": field_search_id,
            f"predicted_{kind}_field_type": classified_field_type,
            f"predicted_{kind}_sport": classified_sport,
            f"{kind}_predicted_field_type_probability": float(field_type_probabilities[i]),
            f"{kind}_predicted_sport_type_probability": float(sport_type_probabilities[i]),
            f"{kind}_image_path": path,
        })

    return pd.DataFrame(rows)


def _annotate_and_detect(
    large_image_dir: str,
    obd_model: YOLO,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    individual_rows = []
    summary_columns = ["field_search_id", "detected_sport", "detect_confidence", "annotated_image_path"]

    image_files = [f for f in os.listdir(large_image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()

    if obd_model is None:
        for img_file in image_files:
            src = os.path.join(large_image_dir, img_file)
            dst = os.path.join(output_dir, img_file)
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass
            base = os.path.splitext(img_file)[0]
            parts = base.split("_")
            try:
                field_search_id = int(float(parts[0]))
            except Exception:
                continue
            summary_rows.append({
                "field_search_id": field_search_id,
                "detected_sport": "Unavailable",
                "detect_confidence": "Unavailable",
                "annotated_image_path": dst,
            })
        return pd.DataFrame(summary_rows, columns=summary_columns), pd.DataFrame(individual_rows)

    if not image_files:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=["field_search_id", "sport_name", "detect_confidence"])

    for img_file in image_files:
        path = os.path.join(large_image_dir, img_file)
        base = os.path.splitext(img_file)[0]
        parts = base.split("_")
        try:
            field_search_id = int(float(parts[0]))
        except Exception:
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read image {path}")
            continue

        results_generator = obd_model.predict(
            source=path,
            save=False,
            conf=CONFIDENCE_THR,
            iou=0.65,
            stream=True,
        )

        detected_sports_for_image = []
        detect_confidence_for_image = []

        for result in results_generator:
            annotated = result.orig_img.copy()
            h, w, _ = annotated.shape
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    if confidence < CONFIDENCE_THR or class_id not in TARGET_CLASS_IDS:
                        continue

                    label = DISPLAY_NAMES.get(class_id, obd_model.names[class_id])
                    individual_rows.append({
                        "field_search_id": field_search_id,
                        "sport_name": label,
                        "detect_confidence": confidence,
                    })
                    detected_sports_for_image.append(label)
                    detect_confidence_for_image.append(confidence)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
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
                    bg_x1 = max(0, text_x)
                    bg_y1 = max(0, text_y - text_size[1] - 10)
                    bg_x2 = min(w, text_x + text_size[0])
                    bg_y2 = min(h, text_y)
                    cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                    cv2.putText(
                        annotated,
                        text,
                        (text_x, bg_y1 + text_size[1] + 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                        cv2.LINE_AA,
                    )
            output_image_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_image_path, annotated)

            summary_rows.append({
                "field_search_id": field_search_id,
                "detected_sport": ", ".join(sorted(set(detected_sports_for_image))) if detected_sports_for_image else "None",
                "detect_confidence": ", ".join(f"{c:.2f}" for c in sorted(set(detect_confidence_for_image))) if detect_confidence_for_image else "None",
                "annotated_image_path": output_image_path,
            })

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    individual_df = pd.DataFrame(individual_rows, columns=["field_search_id", "sport_name", "detect_confidence"])
    return summary_df, individual_df


def run_simulation(input_excel: str, output_root: Optional[str] = None, limit: Optional[int] = None) -> str:
    config = ConfigParser()
    config.read(os.path.join(_script_dir(), "config.ini"))

    source_df = _build_source_dataframe(input_excel)
    if limit is not None:
        source_df = source_df.head(limit).copy()

    input_stem = os.path.splitext(os.path.basename(input_excel))[0]
    run_root = _build_output_root(output_root, input_stem)
    small_dir = os.path.join(run_root, "images_small")
    large_dir = os.path.join(run_root, "images_large")
    detection_dir = os.path.join(run_root, "obj_detect_images")
    _safe_mkdir(run_root, reset=True)
    _safe_mkdir(small_dir)
    _safe_mkdir(large_dir)
    _safe_mkdir(detection_dir)

    print(f"Simulation output root: {run_root}")
    print(f"Rows to process: {len(source_df)}")

    try:
        obd_model, type_model, sport_model = _load_models(config)
    except Exception as exc:
        raise RuntimeError(f"Failed to load local models: {exc}") from exc

    field_rows = []
    object_rows = []
    image_rows = []
    error_rows = []

    for _, row in source_df.iterrows():
        field_search_id = int(row["field_search_id"])
        field_name = str(row["field_name"])
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        sport_name = str(row["primary_sport"])
        gearth_link = str(row["gearth_link"])

        print(f"[SIM] Processing field_search_id={field_search_id} field_name={field_name}")

        small_zoom = 19 if sport_name in ("Basketball", "Tennis") else 18
        large_zoom = 18
        small_img = _fetch_satellite_image(lat, lon, small_zoom, SMALL_IMG_WIDTH, SMALL_IMG_HEIGHT)
        large_img = _fetch_satellite_image(lat, lon, large_zoom, LARGE_IMG_WIDTH, LARGE_IMG_HEIGHT)

        small_path = os.path.join(small_dir, f"{field_search_id}_small_{sport_name}.png")
        large_path = os.path.join(large_dir, f"{field_search_id}_large_{sport_name}.png")
        small_source = "satellite"
        large_source = "satellite"

        try:
            if small_img is None:
                small_img = _make_placeholder_image(
                    SMALL_IMG_WIDTH,
                    SMALL_IMG_HEIGHT,
                    f"{field_name}\n{sport_name}\nsmall image unavailable",
                )
                small_source = "placeholder"
            if large_img is None:
                large_img = _make_placeholder_image(
                    LARGE_IMG_WIDTH,
                    LARGE_IMG_HEIGHT,
                    f"{field_name}\n{sport_name}\nlarge image unavailable",
                )
                large_source = "placeholder"
            if small_img is not None:
                _save_cropped_image(small_img, small_path)
            if large_img is not None:
                _save_cropped_image(large_img, large_path)
        except Exception as exc:
            error_rows.append({
                "field_search_id": field_search_id,
                "field_name": field_name,
                "stage": "image_save",
                "error": str(exc),
            })

        image_rows.append({
            "field_search_id": field_search_id,
            "field_name": field_name,
            "primary_sport": sport_name,
            "latitude": lat,
            "longitude": lon,
            "gearth_link": gearth_link,
            "small_image_path": small_path if os.path.exists(small_path) else None,
            "large_image_path": large_path if os.path.exists(large_path) else None,
            "small_image_source": small_source,
            "large_image_source": large_source,
            "small_zoom": small_zoom,
            "large_zoom": large_zoom,
        })

        if large_img is not None:
            modified_locations = _object_detection_based_modification_by_class(
                field_search_id,
                large_img,
                obd_model,
                {"latitude": lat, "longitude": lon},
                TARGET_CLASS_IDS,
                large_zoom,
            )
            for modified in modified_locations:
                object_rows.append({
                    "field_search_id": field_search_id,
                    "field_name": field_name,
                    "primary_sport": sport_name,
                    "class_id": modified["class_id"],
                    "class_name": modified["class_name"],
                    "latitude": modified["latitude"],
                    "longitude": modified["longitude"],
                    "description": modified["description"],
                    "gearth_link": f"https://earth.google.com/web/@{modified['latitude']},{modified['longitude']},4.1972381a,15000d",
                })

    image_manifest_df = pd.DataFrame(image_rows)
    object_df = pd.DataFrame(object_rows)
    error_df = pd.DataFrame(error_rows)

    small_cls_df = _classify_images(small_dir, type_model, sport_model, "small")
    large_cls_df = _classify_images(large_dir, type_model, sport_model, "large")
    detection_summary_df, detection_individual_df = _annotate_and_detect(large_dir, obd_model, detection_dir)

    combined_df = image_manifest_df.merge(
        small_cls_df,
        on="field_search_id",
        how="left",
    ).merge(
        large_cls_df,
        on="field_search_id",
        how="left",
        suffixes=("", "_large_merge"),
    ).merge(
        detection_summary_df,
        on="field_search_id",
        how="left",
    )

    if not object_df.empty:
        object_summary_df = (
            object_df.groupby(["field_search_id", "field_name", "primary_sport"], as_index=False)
            .agg({
                "class_name": lambda s: ", ".join(sorted(set(s))),
                "latitude": "count",
            })
            .rename(columns={"class_name": "detected_object_types", "latitude": "num_modified_locations"})
        )
    else:
        object_summary_df = pd.DataFrame(columns=[
            "field_search_id",
            "field_name",
            "primary_sport",
            "detected_object_types",
            "num_modified_locations",
        ])

    output_excel = os.path.join(run_root, f"{input_stem}_simulation_results.xlsx")
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        source_df.to_excel(writer, sheet_name="source_data", index=False)
        image_manifest_df.to_excel(writer, sheet_name="image_manifest", index=False)
        small_cls_df.to_excel(writer, sheet_name="small_classification", index=False)
        large_cls_df.to_excel(writer, sheet_name="large_classification", index=False)
        detection_summary_df.to_excel(writer, sheet_name="object_detection", index=False)
        detection_individual_df.to_excel(writer, sheet_name="object_detection_individual", index=False)
        object_df.to_excel(writer, sheet_name="modified_locations", index=False)
        object_summary_df.to_excel(writer, sheet_name="modified_summary", index=False)
        combined_df.to_excel(writer, sheet_name="combined_results", index=False)
        error_df.to_excel(writer, sheet_name="errors", index=False)

    print(f"Saved simulation workbook to: {output_excel}")
    print(f"Saved small images to: {small_dir}")
    print(f"Saved large images to: {large_dir}")
    print(f"Saved annotated detection images to: {detection_dir}")
    return output_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation-only image pipeline for Seattle.xlsx")
    parser.add_argument(
        "--input",
        default=r"C:\Users\owner\Downloads\facilities_batch_1780593810\Seattle.xlsx",
        help="Path to the input Excel file",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Base output directory for simulation artifacts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a quick smoke test",
    )
    args = parser.parse_args()
    run_simulation(args.input, args.output_root, args.limit)


if __name__ == "__main__":
    main()
