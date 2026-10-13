from pathlib import Path
from collections import Counter
import csv

script_folder = Path(__file__).resolve().parent
filename = "Datasheet-Customer-Churn.csv"

# Mendukung dua lokasi script:
# 1. Script di folder proyek, CSV di data/
# 2. Script dan CSV sama-sama di folder data/
candidate_paths = (
    [script_folder / filename]
    if script_folder.name.lower() == "data"
    else [
        script_folder / "data" / filename,
        script_folder / filename,
    ]
)

csv_file = next(
    (path for path in candidate_paths if path.is_file()),
    None,
)

if csv_file is None:
    checked_paths = "\n".join(str(path) for path in candidate_paths)

    raise FileNotFoundError(
        f"File {filename} tidak ditemukan.\n"
        f"Lokasi yang diperiksa:\n{checked_paths}\n"
        "Periksa nama dan lokasi CSV Anda."
    )

with csv_file.open(
    mode="r",
    encoding="utf-8-sig",
    newline="",
) as file:
    reader = csv.DictReader(file)

    if not reader.fieldnames:
        raise ValueError("CSV tidak memiliki header kolom.")

    # Bersihkan spasi pada nama kolom.
    columns = [name.strip() for name in reader.fieldnames]

    if any(not name for name in columns):
        raise ValueError("Ada nama kolom yang kosong.")

    if len(columns) != len(set(columns)):
        raise ValueError("Ada nama kolom yang duplikat.")

    reader.fieldnames = columns
    rows = []

    for row in reader:
        if None in row:
            raise ValueError(
                f"Baris {reader.line_num}: jumlah nilai "
                "melebihi jumlah kolom."
            )

        if any(value is None for value in row.values()):
            raise ValueError(
                f"Baris {reader.line_num}: jumlah nilai "
                "kurang dari jumlah kolom."
            )

        clean_row = {
            key: value.strip()
            for key, value in row.items()
        }

        # Abaikan baris yang seluruh nilainya kosong.
        if any(clean_row.values()):
            rows.append(clean_row)

if not rows:
    raise ValueError("CSV tidak berisi data pelanggan.")

required_columns = [
    "customerID",
    "tenure",
    "Contract",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

missing_columns = [
    name for name in required_columns
    if name not in columns
]

if missing_columns:
    raise ValueError(
        "Kolom yang diperlukan tidak ditemukan: "
        + ", ".join(missing_columns)
        + "\nKolom yang terbaca: "
        + ", ".join(columns)
        + "\nPastikan CSV menggunakan pemisah koma."
    )

print(f"Lokasi CSV: {csv_file}")
print(f"Jumlah baris pelanggan: {len(rows)}")
print(f"Jumlah kolom: {len(columns)}")

print("\nDAFTAR KOLOM")
print(", ".join(columns))

print("\nPEMERIKSAAN NILAI KOSONG")

missing_counts = {
    column: sum(row[column] == "" for row in rows)
    for column in columns
}

if any(missing_counts.values()):
    for column, count in missing_counts.items():
        if count:
            print(f"{column}: {count} nilai kosong")
else:
    print("Tidak ditemukan nilai kosong.")

id_counts = Counter(
    row["customerID"]
    for row in rows
    if row["customerID"]
)

duplicate_ids = [
    customer_id
    for customer_id, count in id_counts.items()
    if count > 1
]

print("\nPEMERIKSAAN CUSTOMER ID")
print(
    "Customer ID kosong:",
    sum(row["customerID"] == "" for row in rows),
)
print("Jumlah ID yang duplikat:", len(duplicate_ids))

if duplicate_ids:
    print("Contoh ID duplikat:", duplicate_ids[:5])

print("\nDISTRIBUSI LABEL CHURN")

churn_counts = Counter(
    row["Churn"] or "(kosong)"
    for row in rows
)

for label, count in sorted(churn_counts.items()):
    print(f"{label}: {count} pelanggan")

print("\nCONTOH DATA PELANGGAN PERTAMA")

for column, value in rows[0].items():
    print(f"{column}: {value or '(kosong)'}")

print("\nSelesai: CSV berhasil dibaca dan diperiksa.")