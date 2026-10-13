from pathlib import Path
import csv
import json

script_folder = Path(__file__).resolve().parent
filename = "Datasheet-Customer-Churn.csv"

# Mendukung script di folder proyek atau di folder data.
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
        f"Lokasi yang diperiksa:\n{checked_paths}"
    )

# Hasil disimpan di folder yang sama dengan CSV.
output_file = csv_file.parent / "telco_chunks.json"

# Label untuk membantu pembacaan dan pencarian makna.
labels = {
    "customerID": "Customer ID / ID pelanggan",
    "gender": "Gender / Jenis kelamin",
    "SeniorCitizen": "Senior citizen indicator / Indikator lansia",
    "Partner": "Partner / Pasangan",
    "Dependents": "Dependents / Tanggungan",
    "tenure": "Tenure / Lama berlangganan (bulan)",
    "PhoneService": "Phone service / Layanan telepon",
    "MultipleLines": "Multiple lines / Beberapa saluran telepon",
    "InternetService": "Internet service / Layanan internet",
    "OnlineSecurity": "Online security / Keamanan online",
    "OnlineBackup": "Online backup / Pencadangan online",
    "DeviceProtection": "Device protection / Perlindungan perangkat",
    "TechSupport": "Technical support / Dukungan teknis",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies / Streaming film",
    "Contract": "Contract / Jenis kontrak",
    "PaperlessBilling": "Paperless billing / Tagihan tanpa kertas",
    "PaymentMethod": "Payment method / Metode pembayaran",
    "MonthlyCharges": "Monthly charges / Biaya bulanan",
    "TotalCharges": "Total charges / Total biaya",
    "Churn": "Recorded churn label / Label churn tercatat",
}

chunks = []
seen_customer_ids = set()

with csv_file.open(
    mode="r",
    encoding="utf-8-sig",
    newline="",
) as file:
    reader = csv.DictReader(file)

    if not reader.fieldnames:
        raise ValueError("CSV tidak memiliki header kolom.")

    columns = [name.strip() for name in reader.fieldnames]

    if any(not name for name in columns):
        raise ValueError("Ada nama kolom yang kosong.")

    if len(columns) != len(set(columns)):
        raise ValueError("Ada nama kolom yang duplikat.")

    reader.fieldnames = columns

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
        )

    for row in reader:
        csv_line_end = reader.line_num

        if None in row:
            raise ValueError(
                f"Baris {csv_line_end}: jumlah nilai "
                "melebihi jumlah kolom."
            )

        if any(value is None for value in row.values()):
            raise ValueError(
                f"Baris {csv_line_end}: jumlah nilai "
                "kurang dari jumlah kolom."
            )

        clean_row = {
            key: value.strip()
            for key, value in row.items()
        }

        if not any(clean_row.values()):
            continue

        customer_id = clean_row["customerID"]

        if not customer_id:
            raise ValueError(
                f"Customer ID kosong pada baris {csv_line_end}."
            )

        if customer_id in seen_customer_ids:
            raise ValueError(
                f"Customer ID duplikat pada baris {csv_line_end}: "
                f"{customer_id}"
            )

        seen_customer_ids.add(customer_id)

        text_lines = [
            "Telco customer record / Catatan pelanggan telekomunikasi",
            f"Source / Sumber: {csv_file.name}",
        ]

        # Semua kolom dipertahankan, termasuk kolom tambahan.
        for column in columns:
            label = labels.get(column, column)
            value = clean_row[column] or "Tidak tersedia"
            text_lines.append(f"{label}: {value}")

        chunks.append({
            "chunk_id": f"telco_{customer_id}",
            "source": csv_file.name,
            "customer_id": customer_id,
            "record_number": len(chunks) + 1,
            "csv_line_end": csv_line_end,
            "text": "\n".join(text_lines),
        })

if not chunks:
    raise ValueError("Tidak ada data pelanggan untuk dibuat menjadi chunks.")

output_file.write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Sumber dataset: {csv_file}")
print(f"Jumlah pelanggan: {len(chunks)}")
print(f"Jumlah chunks: {len(chunks)}")
print(f"File hasil: {output_file}")

print("\nCONTOH CHUNK PERTAMA")
print(json.dumps(chunks[0], ensure_ascii=False, indent=2))

print("\nSelesai: telco_chunks.json berhasil dibuat.")