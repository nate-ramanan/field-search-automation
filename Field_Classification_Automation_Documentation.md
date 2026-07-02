# Field Classification Automation — Process Notes

This document walks through each script in the field-classification workflow, starting at each script’s `__main__` block, and summarizes the significant functions, image outputs, and database tables/columns touched. It is based on the current code in this workspace and reflects the execution order you specified.

**Workflow Order (as run on the command line)**
1. Locate to the main directory
2. `python NewGoogleEarth.py`
3. `python GetImagesTotal.py`
4. `python TotalDetect_multi_size.py`
5. Check `total_error.xlsx` for errors
6. Check `combined_results_{TIMESTAMP}.xlsx` and `obd_individual_results_{TIMESTAMP}.xlsx`
7. `python update_database.py` (with the nge_object insert/update toggle noted below)

**Timestamp convention**
`{TIMESTAMP}` in `combined_results_{TIMESTAMP}.xlsx` and `obd_individual_results_{TIMESTAMP}.xlsx` is created at script start (`NOW = datetime.datetime.now()` in `TotalDetect_multi_size.py`), not at completion.

---

**Global Inputs and Outputs**
- **Google APIs**: Both `NewGoogleEarth.py` and `GetImagesTotal.py` call the Google Static Maps API for satellite imagery. API key is hardcoded in both files.
- **Models**: YOLO object detection (`obd_model`) and Keras classification models (`sport_model`, `type_model`). Paths are read from `config.ini`.
- **Image folders** (from `config.ini`):
  - Small zoom images: `image_paths.total_zoom_small_path`
  - Large zoom images: `image_paths.total_zoom_large_path`
  - Annotated detection output: `image_paths.obj_detect_img_path`
- **Primary database tables**:
  - `public.new_google_earth`
  - `public.nge_object`
  - `public.maps_api_log`
  - `public.postal_code`, `public.state_province`, `public.schools`

---

**1) `NewGoogleEarth.py` — Collect and Insert Field Records (and initial object centroids)**

**What it does**
Queries Google Places for sports facilities by zip code, inserts base facility records into `public.new_google_earth`, logs processed zip codes, deduplicates, then runs YOLO on satellite images to recenter objects and inserts those object rows into `public.nge_object`.

**Main flow (from `if __name__ == "__main__":`)**
1. Prompt for `state_code`, `city_to_add`, and `single_zip`.
2. `fetch_cities(state_code)` reads cities from `postal_code`/`state_province` tables.
3. `fetch_zip_codes(city, state_code)` returns zip codes not already logged in `maps_api_log`.
4. For each selected zip code:
   - `fetch_schools(zip_code)` loads school records from `public.schools`.
   - For each sport type, `get_fields(query)` calls Google Places Text Search, then `parse_fields(xml_data, sport_type_id)` parses results into field dictionaries.
   - `save_field_data(all_fields)` inserts those fields into `public.new_google_earth`.
   - `log_processed(state, city, zip_code)` inserts into `public.maps_api_log`.
5. `delete_duplicates()` removes duplicate rows from `public.new_google_earth` (dedupe by `gplace_id`, `field_name`, `gps_location`, `object_sport`).
6. `get_field_data()` (from `config_getImages.py`) fetches all `new_google_earth` rows.
7. For each row, `get_satellite_image_array()` downloads the satellite image, then `object_detection_based_modification_by_class()` runs YOLO and computes recentered GPS locations for target sport classes.
8. `save_object_data(nge_object)` bulk-inserts the detected object rows into `public.nge_object`.

**Key functions and what they do**
- `get_fields(query)` — Calls Google Places Text Search API and returns XML text for parsing.
- `parse_fields(xml_data, search_sport_type)` — Converts XML into dicts with address fields, GPS, `gplace_id`, and `gearth_link`.
- `save_field_data(fields)` — Inserts rows into `public.new_google_earth`.
- `get_satellite_image_array(gps_location)` — Downloads a satellite image and returns a NumPy array used for YOLO detection.
- `object_detection_based_modification_by_class(...)` — Runs YOLO, recenters GPS for each detected target object class, and returns a list of refined GPS points.
- `save_object_data(fields)` — Inserts object records into `public.nge_object`.
- `log_processed(state, city, zip_code)` — Logs zip code processing into `public.maps_api_log`.
- `delete_duplicates()` — Deletes duplicates in `public.new_google_earth` with a windowed `ROW_NUMBER()`.

**Database writes from this script**
- `public.new_google_earth` INSERT columns:
  - `field_name`, `object_sport`, `formatted_address`, `postal_code`, `street`, `city`, `state`, `gps_location`, `gplace_id`, `search_sport_type`, `gearth_link`, `object_gearth_link`, `object_gps_location`
- `public.nge_object` INSERT columns:
  - `field_search_id`, `sport_name`, `image_url`, `description`
- `public.maps_api_log` INSERT columns:
  - `state`, `city`, `zip_code`

**Files referenced**
- `NewGoogleEarth.py`
- `config_getImages.py`
- `ConnectionPool.py`

---

**2) `GetImagesTotal.py` — Download Small and Large Images (and fill in missing sports)**

**What it does**
Creates a fresh set of small and large satellite images for each detected object (and any original fields that still need detection), and writes an error sheet for missing/invalid maps.

**Main flow (from `if __name__ == "__main__":`)**
1. Deletes and recreates the small and large image directories.
2. Calls `get_field_data(pool)` from `config_getImagesObject.py` which returns:
   - All `new_google_earth` rows joined with `nge_object` (`sport_name` from `nge_object`), plus
   - “orig” rows from `new_google_earth` without an object (`sport_name = 'orig'`).
3. Loads YOLO object model (`obd_model`).
4. For each row:
   - If `sport_name != 'orig'`: call `ProcessImage(...)` to download and save small and large images.
   - If `sport_name == 'orig'`: run YOLO on a large image to pick a sport class, then call `ProcessImage(...)`.
5. Writes errors to `total_error.xlsx` (path from `config.ini`).

**Key functions and what they do**
- `gpsLocation(link)` — Parses latitude/longitude from a Google Earth link (`@lat,lon,...`).
- `getImage(lat, lon, key, zoom, width, height)` — Builds a Google Static Maps URL for a satellite image.
- `saveImg(image_url, save_path, crop_bottom)` — Downloads image, crops off the bottom watermark, saves to disk.
- `ProcessImage(...)` — Chooses zoom based on sport, downloads small and large images, and saves them.

**Image outputs and naming**
- Small images go to `image_paths.total_zoom_small_path`.
- Large images go to `image_paths.total_zoom_large_path`.
- Filenames are: `{nge_object_id_or_orig}_{field_search_id}_{sport_name}.png`.
  - Example: `123_456_Soccer.png` or `orig_456_Baseball.png`.
- Zoom logic:
  - Basketball/Tennis: zoom 19 for small image.
  - Baseball: zoom 18 for small image.
  - All others: zoom 19 for small image.
  - Large image: zoom 18 for all sports.

**Database reads (no writes in this script)**
- `public.new_google_earth` and `public.nge_object` via `config_getImagesObject.get_field_data`.

**Files referenced**
- `GetImagesTotal.py`
- `config_getImagesObject.py`
- `ConnectionPool.py`

---

**3) `TotalDetect_multi_size.py` — Classify + Detect, then Produce Excel Outputs**

**What it does**
Runs classification on small and large images, runs YOLO object detection on large images, merges results with database metadata, and writes output Excel files for database update.

**Main flow (from `if __name__ == "__main__":`)**
1. Deletes and recreates the object-detection output folder (`obj_detect_images`).
2. Loads `nge_object_id` / `field_search_id` pairs from `public.nge_object`.
3. Runs classification on:
   - Small images (`Classification`)
   - Large images (`Classification_large`)
4. Merges classification outputs on `(field_search_id, nge_object_id)`.
5. Runs YOLO object detection on large images (`Object_Detection`).
6. Merges classification + detection into a combined DataFrame.
7. Fetches related DB fields from `public.new_google_earth` and `public.nge_object` via `get_database_fields` and merges with the combined results.
8. Filters columns to only those needed to update `public.nge_object`.
9. Writes:
   - `combined_results_{TIMESTAMP}.xlsx`
   - `obd_individual_results_{TIMESTAMP}.xlsx`

**Key functions and what they do**
- `get_nge_object_ids_from_database()` — Reads `nge_object_id`, `field_search_id`, `sport_name` from `public.nge_object`.
- `_match_image_to_database_ids(image_dir_path, db_ids_df)` — Matches image filenames to DB IDs.
- `Classification(...)` — Uses small images + Keras models to predict `predicted_field_type` and `predicted_sport`.
- `Classification_large(...)` — Same as above, but on large images and writes `predicted_large_*` columns.
- `Object_Detection(...)` — Runs YOLO on large images, saves annotated images, and returns detection summaries.
- `get_database_fields(...)` — Reads joined fields from `public.new_google_earth` and `public.nge_object`.

**Classification labels and overrides**
- Sports: `['Buildings', 'Soccer', 'Baseball', 'Tennis', 'Basketball']`.
- Types: `['Grass', 'Turf']`.
- Overrides:
  - If predicted sport is Basketball or Tennis, `predicted_field_type = 'Synthetic'`.
  - If predicted sport is Buildings, `predicted_field_type = 'Buildings'`.

**Detection outputs**
- Annotated detections saved to `image_paths.obj_detect_img_path` with the same filename as the input image.
- `object_detection_results_df` includes:
  - `field_search_id`, `nge_object_id`, `detected_sport`, `detect_confidence` (comma-separated strings per image).
- `object_detection_individual_results_df` includes one row per detection with `sport_name` and `detect_confidence`.

**Excel outputs**
- `combined_results_{TIMESTAMP}.xlsx` columns (filtered):
  - `nge_object_id`, `field_search_id`,
  - `predicted_field_type`, `predicted_sport`, `predicted_field_type_probability`, `predicted_sport_type_probability`,
  - `predicted_large_field_type`, `predicted_large_sport`, `large_predicted_field_type_probability`, `large_predicted_sport_type_probability`,
  - `detected_sport`, `detect_confidence`
- `obd_individual_results_{TIMESTAMP}.xlsx` columns:
  - `field_search_id`, `nge_object_id`, `sport_name`, `detect_confidence`

**Files referenced**
- `TotalDetect_multi_size.py`
- `ConnectionPool.py`

---

**4) `update_database.py` — Apply Results to PostgreSQL**

**What it does**
Loads the latest `combined_results_*.xlsx` and updates (or inserts) result columns into `public.new_google_earth` and/or `public.nge_object`.

**Key functions and what they do**
- `update_existing_data(...)` — Executes `UPDATE ... WHERE {id_column} = %s` for each row.
- `insert_data(...)` — Executes `INSERT INTO {table} (...) VALUES (...)` for each row.
- `update_latest_existing_data_with_df(...)` — Picks the latest Excel file by prefix, asks for confirmation, then updates.
- `insert_latest_existing_data_with_df(...)` — Picks the latest Excel file by prefix, asks for confirmation, then inserts.

**Intended table updates**
- `public.new_google_earth` (by `field_search_id`):
  - `predicted_field_type`, `predicted_sport`, `predicted_large_field_type`, `predicted_large_sport`, `detected_sport`
- `public.nge_object` (by `nge_object_id`):
  - `predicted_field_type`, `predicted_sport`, `predicted_field_type_probability`, `predicted_sport_type_probability`,
  - `predicted_large_field_type`, `predicted_large_sport`, `large_predicted_field_type_probability`, `large_predicted_sport_type_probability`,
  - `detected_sport`, `detect_confidence`

**Important note about nge_object insert vs update**
Your instruction about toggling `INSERT` vs `UPDATE` for `public.nge_object` applies here. The script has commented blocks for either insert or update. Before running, confirm which block is uncommented to avoid duplicate or missing data.

---

**File-level references**
- `NewGoogleEarth.py`
- `GetImagesTotal.py`
- `TotalDetect_multi_size.py`
- `update_database.py`
- `config_getImages.py`
- `config_getImagesObject.py`
- `ConnectionPool.py`

---

**Known issues / risks spotted in current code**
- `update_database.py` appears to have a syntax/indentation break near the bottom:
  - `id_column_2` and `columns_to_update_2` are not indented under `if __name__ == '__main__':`.
  - The call `update_latest_existing_data_with_df(yes ...` is invalid Python.
  - This will prevent the script from running as-is and needs to be corrected before use.

---

**Quick cross-reference: where images are created**
- `GetImagesTotal.py` writes raw satellite images:
  - Small images: `total_zoom_small_path`
  - Large images: `total_zoom_large_path`
- `TotalDetect_multi_size.py` writes annotated YOLO detections:
  - `obj_detect_img_path`

---

**Quick cross-reference: where database is written**
- `NewGoogleEarth.py` writes:
  - `public.new_google_earth`
  - `public.nge_object`
  - `public.maps_api_log`
- `update_database.py` updates:
  - `public.new_google_earth`
  - `public.nge_object`

