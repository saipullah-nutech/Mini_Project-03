"""Local Telco embeddings. No Azure, API key, or .env required.
Install: python -m pip install --upgrade sentence-transformers numpy
Check inputs: python 03_create_embeddings.py --check
Run on CPU: python 03_create_embeddings.py
Use saved model offline: python 03_create_embeddings.py --offline
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "b8ef00830037f9868450f778081ea683e900fe39"
MAX_SEQUENCE_LENGTH = 512


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128:
        raise ValueError("Batch size harus 1 sampai 128.")
    script_folder = Path(__file__).resolve().parent
    candidates = [script_folder / "data", script_folder]
    data_folder = next((p for p in candidates if (p / "Datasheet-Customer-Churn.csv").is_file()), None)
    if data_folder is None:
        raise FileNotFoundError("Datasheet-Customer-Churn.csv tidak ditemukan di folder script atau data/.")
    chunks_path = data_folder / "telco_chunks.json"
    if not chunks_path.is_file():
        raise FileNotFoundError("Jalankan 02_create_chunks.py dahulu untuk membuat data/telco_chunks.json.")
    raw = chunks_path.read_bytes()
    chunks = json.loads(raw.decode("utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("telco_chunks.json tidak berisi daftar chunks.")
    ids = set()
    for c in chunks:
        if not isinstance(c, dict):
            raise ValueError("Chunk bukan object JSON.")
        for field in ("chunk_id", "customer_id", "source", "text"):
            if not isinstance(c.get(field), str) or not c[field].strip():
                raise ValueError(f"Field chunk tidak valid: {field}")
        if c["source"] != "Datasheet-Customer-Churn.csv":
            raise ValueError("Sumber chunks tidak sesuai. Jalankan ulang step 02.")
        if c["chunk_id"] in ids:
            raise ValueError("Chunk ID duplikat.")
        ids.add(c["chunk_id"])
    with (data_folder / "Datasheet-Customer-Churn.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV tidak memiliki header.")
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        if "customerID" not in reader.fieldnames:
            raise ValueError("Kolom customerID tidak ditemukan.")
        csv_ids = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Format CSV tidak sesuai pada baris {reader.line_num}.")
            if any(v.strip() for v in row.values()):
                csv_ids.append(row["customerID"].strip())
    if not all(csv_ids) or len(set(csv_ids)) != len(csv_ids):
        raise ValueError("Ada customerID kosong/duplikat.")
    if csv_ids != [c["customer_id"] for c in chunks]:
        raise ValueError("Chunks berbeda dari urutan/jumlah pelanggan CSV. Jalankan ulang step 02.")
    print(f"Dataset: {len(chunks)} pelanggan/chunks")
    print(f"Model: {MODEL_ID}")
    print(f"Perangkat: CPU; batch size: {args.batch_size}")
    if args.check:
        print("Pemeriksaan input berhasil. Tidak mengunduh model atau membuat embeddings.")
        return

    import numpy as np
    from sentence_transformers import SentenceTransformer

    model_folder = data_folder / "models" / "telco_multilingual_minilm"
    model_marker = model_folder / "telco_model_manifest.json"
    model_config = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "max_sequence_length": MAX_SEQUENCE_LENGTH}
    if model_marker.is_file():
        if json.loads(model_marker.read_text(encoding="utf-8")) != model_config:
            raise ValueError("Konfigurasi model lokal berbeda. Gunakan folder model yang sesuai.")
        print("Memuat model yang sudah tersimpan di komputer...", flush=True)
        model = SentenceTransformer(str(model_folder), device="cpu", local_files_only=True)
    else:
        if args.offline:
            raise FileNotFoundError("Model lokal belum tersedia. Jalankan tanpa --offline dengan koneksi internet dahulu.")
        print("Mengunduh model dari Hugging Face. Tunggu hingga selesai...", flush=True)
        model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu", trust_remote_code=False)
        model.max_seq_length = MAX_SEQUENCE_LENGTH
        model_folder.mkdir(parents=True, exist_ok=True)
        model.save(str(model_folder))
        write_json(model_marker, model_config)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    dimension = model.get_sentence_embedding_dimension()
    if dimension != 384:
        raise ValueError(f"Dimensi model tidak sesuai: {dimension}")
    # Prevent silent truncation of long customer records.
    print("Memeriksa panjang teks agar atribut pelanggan tidak terpotong...", flush=True)
    for start in range(0, len(chunks), 128):
        group = chunks[start:start + 128]
        tokenized = model.tokenizer([c["text"] for c in group], truncation=False, padding=False)
        for c, token_ids in zip(group, tokenized["input_ids"]):
            if len(token_ids) > MAX_SEQUENCE_LENGTH:
                raise ValueError(f"Chunk {c['chunk_id']} melebihi 512 token. Ringkas label/teks pada step 02 dahulu.")

    chunks_hash = hashlib.sha256(raw).hexdigest()
    signature = {**model_config, "chunks_sha256": chunks_hash, "batch_size": args.batch_size, "normalized": True}
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    cache = data_folder / "local_embedding_checkpoints" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "manifest.json", signature)
    total = len(chunks)
    matrix = np.empty((total, dimension), dtype=np.float32)
    for batch_no, start in enumerate(range(0, total, args.batch_size), start=1):
        batch = chunks[start:start + args.batch_size]
        path = cache / f"batch_{batch_no:05d}.npy"
        from_cache = path.is_file()
        if from_cache:
            vectors = np.load(path, allow_pickle=False)
        else:
            vectors = np.asarray(model.encode(
                [c["text"] for c in batch], batch_size=args.batch_size,
                show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
            ), dtype=np.float32)
        if vectors.shape != (len(batch), dimension) or not np.isfinite(vectors).all():
            raise ValueError(f"Vektor batch {batch_no} tidak valid.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError(f"Vektor batch {batch_no} belum ternormalisasi.")
        if not from_cache:
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("wb") as f:
                np.save(f, vectors, allow_pickle=False)
            temporary.replace(path)
        matrix[start:start + len(batch)] = vectors
        status = "checkpoint" if from_cache else "dibuat"
        print(f"Batch {batch_no} {status}: {start + len(batch)}/{total}", flush=True)

    # Use paired unique filenames to prevent mismatching index and vectors.
    vectors_filename = f"telco_embeddings_{cache_key}.npy"
    vectors_path = data_folder / vectors_filename
    temporary = vectors_path.with_name(vectors_path.name + ".tmp")
    with temporary.open("wb") as f:
        np.save(f, matrix, allow_pickle=False)
    temporary.replace(vectors_path)
    index_path = data_folder / "telco_local_index.json"
    write_json(index_path, {
        "embedding_model": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_path": "models/telco_multilingual_minilm",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "dimensions": dimension, "normalized": True,
        "chunk_count": total, "chunks_sha256": chunks_hash,
        "vectors_file": vectors_filename, "records": chunks,
    })
    print(f"\nSelesai: {total} embeddings lokal, dimensi {dimension}.")
    print(f"Index: {index_path}")
    print(f"Vektor: {vectors_path}")
    print("Urutan records pada index sama dengan urutan baris vektor.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDihentikan. Jalankan ulang untuk melanjutkan batch tersimpan.")
        sys.exit(1)
    except ModuleNotFoundError:
        print("Instal paket pada Python yang sama: python -m pip install --upgrade sentence-transformers numpy", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
