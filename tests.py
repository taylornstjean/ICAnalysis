import sqlite3
from pathlib import Path

DB_PATH = Path("/data/i3store/users/tstjean/data/backend/test/merged/merged.db")
TRUTH_TABLE = "truth"
TARGET_LABELS = ["position_x", "position_y", "position_z"]

def print_header(title):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

def main():
    print_header("GRAPHNET TRUTH TABLE DIAGNOSTIC REPORT")

    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. SCHEMA CHECK
    print_header("STEP 1: SCHEMA CHECK")
    cursor.execute(f"PRAGMA table_info({TRUTH_TABLE});")
    schema = cursor.fetchall()
    all_columns = [row[1] for row in schema]

    print("Available columns in the 'truth' table:")
    for col in all_columns:
        print(f"  - {col}")

    missing = [label for label in TARGET_LABELS if label not in all_columns]
    if missing:
        print(f"\n[WARNING] The following target labels are MISSING from the schema: {missing}")
    else:
        print("\n[OK] All expected target labels are present in the schema.")

    # 2. TOTAL ROW COUNT
    print_header("STEP 2: TOTAL ROW COUNT")
    cursor.execute(f"SELECT COUNT(*) FROM {TRUTH_TABLE};")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows in '{TRUTH_TABLE}' table: {total_rows}")

    # 3. ROWS WITH NULL TARGET LABELS
    print_header("STEP 3: MISSING LABEL CHECK")
    null_query = " OR ".join([f"{label} IS NULL" for label in TARGET_LABELS])
    cursor.execute(f"SELECT COUNT(*) FROM {TRUTH_TABLE} WHERE {null_query};")
    null_rows = cursor.fetchone()[0]
    if null_rows > 0:
        print(f"[WARNING] {null_rows} rows are missing at least one of {TARGET_LABELS}")
    else:
        print(f"[OK] All rows contain non-null values for {TARGET_LABELS}")

    # 4. SAMPLE BAD ROWS
    print_header("STEP 4: SAMPLE BAD ROWS (max 10 shown)")
    cursor.execute(
        f"""
        SELECT event_no, pid, {", ".join(TARGET_LABELS)}
        FROM {TRUTH_TABLE}
        WHERE {null_query}
        LIMIT 10;
        """
    )
    bad_samples = cursor.fetchall()
    if bad_samples:
        for row in bad_samples:
            print(row)
    else:
        print("[OK] No rows with missing target labels.")

    # Done
    print_header("END OF REPORT")
    conn.close()

if __name__ == "__main__":
    main()

