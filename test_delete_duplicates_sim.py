"""
Simulate delete_duplicates() logic against a TEMP table only.
Does NOT read from or modify public.new_google_earth or any other permanent table.
"""
import sys
from configparser import ConfigParser

import psycopg2

# Exact SQL from NewGoogleEarth.delete_duplicates()
DELETE_DUPLICATES_SQL = """
WITH RANKED AS (
    SELECT
        ctid,
        gplace_id,
        field_name,
        gps_location,
        field_search_id,
        search_sport_type,
        ROW_NUMBER() OVER (
            PARTITION BY
                COALESCE(gplace_id, ''),
                COALESCE(field_name, ''),
                COALESCE(gps_location, '')
            ORDER BY ctid
        ) AS rn
    FROM sim_new_google_earth
)
DELETE FROM sim_new_google_earth
WHERE ctid IN (
    SELECT ctid
    FROM RANKED
    WHERE rn > 1
);
"""

# Simulated rows shaped like pgAdmin new_google_earth data
SIMULATED_ROWS = [
    # Exact duplicate pair (same place re-inserted for soccer)
    (1001, "Lumen Field", "ChIJabc123", "47.5951518,-122.3316394", 79, ""),
    (1002, "Lumen Field", "ChIJabc123", "47.5951518,-122.3316394", 79, ""),
    # Same place/name/gps but different search_sport_type (tennis vs soccer search)
    (1003, "Lumen Field", "ChIJabc123", "47.5951518,-122.3316394", 87, ""),
    # Unique field
    (1004, "Cal Anderson Park", "ChIJdef456", "47.6170185,-122.319127", 79, ""),
    # School row with NULL gplace_id (like schools_data inserts)
    (1005, "Garfield High School", None, "47.60701,-122.3012662", 100, ""),
    (1006, "Garfield High School", None, "47.60701,-122.3012662", 100, ""),
    # Same name but different GPS — should remain
    (1007, "Delridge Playfield", "ChIJghi789", "47.562992,-122.3645196", 79, ""),
    (1008, "Delridge Playfield", "ChIJxyz000", "47.563100,-122.365000", 79, ""),
    # Triple duplicate
    (1009, "Miller Community Center", "ChIJmill01", "47.6217681,-122.306964", 9, "Basketball"),
    (1010, "Miller Community Center", "ChIJmill01", "47.6217681,-122.306964", 9, ""),
    (1011, "Miller Community Center", "ChIJmill01", "47.6217681,-122.306964", 9, ""),
]


def connect():
    config = ConfigParser()
    config.read(r"c:\Users\owner\Documents\Gameplay_FieldAutomation_full_v3\Gameplay_FieldAutomation_full_v3\config.ini")
    return psycopg2.connect(
        user=config.get("database", "username"),
        password=config.get("database", "password"),
        host=config.get("database", "hostname"),
        port=int(config.get("database", "port_id")),
        database=config.get("database", "database"),
    )


def setup_temp_table(cur):
    cur.execute(
        """
        CREATE TEMP TABLE sim_new_google_earth (
            field_search_id INTEGER PRIMARY KEY,
            field_name TEXT,
            object_sport TEXT,
            formatted_address TEXT,
            postal_code TEXT,
            street TEXT,
            city TEXT,
            state TEXT,
            gps_location TEXT,
            gplace_id TEXT,
            search_sport_type INTEGER,
            gearth_link TEXT,
            object_gearth_link TEXT,
            object_gps_location TEXT
        ) ON COMMIT DROP
        """
    )


def insert_simulated_rows(cur):
    cur.executemany(
        """
        INSERT INTO sim_new_google_earth (
            field_search_id, field_name, object_sport, formatted_address,
            postal_code, street, city, state, gps_location, gplace_id,
            search_sport_type, gearth_link, object_gearth_link, object_gps_location
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """,
        [
            (
                row[0],
                row[1],
                row[5],
                "123 Main St, Seattle, WA 98101",
                "98101",
                "123 Main St",
                "SEATTLE",
                "WA",
                row[3],
                row[2],
                row[4],
                f"https://earth.google.com/web/@{row[3]},4.1972381a,15000d",
                "",
                "",
            )
            for row in SIMULATED_ROWS
        ],
    )


def fetch_ids(cur):
    cur.execute(
        "SELECT field_search_id FROM sim_new_google_earth ORDER BY field_search_id"
    )
    return [r[0] for r in cur.fetchall()]


def test_fk_blocks_delete_on_referenced_rows():
    """Simulate nge_object FK blocking delete on rows still referenced by child table."""
    conn = connect()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute(
            """
            CREATE TEMP TABLE sim_new_google_earth (
                field_search_id INTEGER PRIMARY KEY,
                field_name TEXT,
                gplace_id TEXT,
                gps_location TEXT,
                search_sport_type INTEGER
            ) ON COMMIT DROP
            """
        )
        cur.execute(
            """
            CREATE TEMP TABLE sim_nge_object (
                nge_object_id SERIAL PRIMARY KEY,
                field_search_id INTEGER REFERENCES sim_new_google_earth(field_search_id)
            ) ON COMMIT DROP
            """
        )
        cur.execute(
            """
            INSERT INTO sim_new_google_earth VALUES
            (2001, 'Referenced Field', 'ChIJref', '47.1,-122.1', 79),
            (2002, 'Referenced Field', 'ChIJref', '47.1,-122.1', 79)
            """
        )
        cur.execute("INSERT INTO sim_nge_object (field_search_id) VALUES (2002)")

        fk_sql = """
        WITH RANKED AS (
            SELECT ctid, field_search_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(gplace_id,''), COALESCE(field_name,''), COALESCE(gps_location,'')
                       ORDER BY ctid
                   ) AS rn
            FROM sim_new_google_earth
        )
        DELETE FROM sim_new_google_earth
        WHERE ctid IN (SELECT ctid FROM RANKED WHERE rn > 1);
        """
        try:
            cur.execute(fk_sql)
            conn.commit()
            fk_blocked = False
        except psycopg2.Error as exc:
            conn.rollback()
            fk_blocked = "violates foreign key constraint" in str(exc)

        assert fk_blocked, "Expected FK violation when deleting referenced duplicate row"
        print("PASS: FK simulation — delete is blocked when nge_object still references the row.")
        return True

    finally:
        cur.close()
        conn.close()


def run_simulation():
    conn = connect()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # Safety: confirm we are not touching the permanent table row count
        cur.execute("SELECT COUNT(*) FROM public.new_google_earth")
        public_count_before = cur.fetchone()[0]

        setup_temp_table(cur)
        insert_simulated_rows(cur)

        before_count = len(SIMULATED_ROWS)
        before_ids = fetch_ids(cur)
        print(f"Inserted {before_count} simulated rows: {before_ids}")

        cur.execute(DELETE_DUPLICATES_SQL)
        deleted = cur.rowcount
        after_ids = fetch_ids(cur)

        # Partition key is (gplace_id, field_name, gps_location) only — NOT search_sport_type.
        # Lumen Field x3 -> keep 1001, delete 1002 + 1003
        # Garfield HS x2 -> keep 1005, delete 1006
        # Miller x3 -> keep 1009, delete 1010 + 1011
        expected_deleted = 5
        expected_remaining = {1001, 1004, 1005, 1007, 1008, 1009}

        print(f"Rows deleted by SQL: {deleted}")
        print(f"Remaining field_search_ids: {after_ids}")

        assert deleted == expected_deleted, (
            f"Expected {expected_deleted} deletes, got {deleted}"
        )
        assert set(after_ids) == expected_remaining, (
            f"Unexpected survivors. Expected {sorted(expected_remaining)}, got {after_ids}"
        )

        # Roll back everything; TEMP table is session-scoped anyway
        conn.rollback()

        cur.execute("SELECT COUNT(*) FROM public.new_google_earth")
        public_count_after = cur.fetchone()[0]
        assert public_count_before == public_count_after, (
            "public.new_google_earth row count changed — aborting"
        )

        print("\nPASS: delete_duplicates SQL behaves correctly on simulated data.")
        print(f"public.new_google_earth unchanged ({public_count_before} rows).")
        return 0

    except Exception as exc:
        conn.rollback()
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    code = run_simulation()
    if code == 0:
        try:
            test_fk_blocks_delete_on_referenced_rows()
        except Exception as exc:
            print(f"\nFAIL (FK simulation): {exc}", file=sys.stderr)
            code = 1
    sys.exit(code)
