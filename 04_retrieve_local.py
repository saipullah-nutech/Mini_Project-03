"""Retrieve customer evidence from the local Telco embedding index.
Run: python 04_retrieve_local.py
Check: python 04_retrieve_local.py --check
Single question: python 04_retrieve_local.py --question "Tampilkan pelanggan dengan kontrak bulanan"
"""
from pathlib import Path
import argparse
import json
import re
import sys


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("Top-k harus minimal 1.")
    folder = Path(__file__).resolve().parent
    data_folder = next((p for p in [folder / "data", folder] if (p / "telco_local_index.json").is_file()), None)
    if data_folder is None:
        raise FileNotFoundError("telco_local_index.json tidak ditemukan. Selesaikan step 03 dahulu.")
    index = json.loads((data_folder / "telco_local_index.json").read_text(encoding="utf-8"))
    records = index.get("records", [])
    if not records or len(records) != index.get("chunk_count"):
        raise ValueError("Jumlah records index tidak sesuai.")
    customer_map = {}
    for position, record in enumerate(records):
        if not all(isinstance(record.get(f), str) and record[f].strip() for f in ("customer_id", "chunk_id", "source", "text")):
            raise ValueError("Record index tidak lengkap.")
        key = record["customer_id"].upper()
        if key in customer_map:
            raise ValueError("Customer ID duplikat di index.")
        customer_map[key] = position
    import numpy as np
    matrix_path = data_folder / index["vectors_file"]
    if not matrix_path.is_file():
        raise FileNotFoundError(f"File vektor tidak ditemukan: {matrix_path}")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(records), index["dimensions"]):
        raise ValueError("Ukuran vektor berbeda dari index. Jalankan ulang step 03.")
    if index.get("normalized") is not True:
        raise ValueError("Index harus berisi vektor ternormalisasi dari step 03.")
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3):
        raise ValueError("Vektor tidak valid atau belum ternormalisasi.")
    model_path = data_folder / index["model_path"]
    if not (model_path / "telco_model_manifest.json").is_file():
        raise FileNotFoundError(f"Model lokal tidak lengkap: {model_path}")
    model_manifest = json.loads((model_path / "telco_model_manifest.json").read_text(encoding="utf-8"))
    if (model_manifest.get("model_id") != index["embedding_model"] or
        model_manifest.get("revision") != index["model_revision"] or
        model_manifest.get("max_sequence_length") != index["max_sequence_length"]):
        raise ValueError("Model pertanyaan harus sama dengan model pembuat index.")
    print(f"Index siap: {len(records)} pelanggan, {index['dimensions']} dimensi.")
    if args.check:
        print("Pemeriksaan index, vektor, dan metadata model berhasil; model belum dimuat.")
        return
    model = None

    def retrieve(question):
        nonlocal model
        # Known IDs use exact lookup; unknown IDs in the dataset's format are reported.
        words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", question.upper())
        requested_ids = list(dict.fromkeys(w for w in words if w in customer_map or re.fullmatch(r"\d{4}-[A-Z0-9]{5}", w)))
        unknown = [w for w in requested_ids if w not in customer_map]
        matches = []
        if requested_ids:
            method = "exact_customer_id"
            for customer_id in requested_ids:
                if customer_id in customer_map:
                    matches.append({**records[customer_map[customer_id]], "similarity": None})
        else:
            method = "semantic_cosine"
            if model is None:
                from sentence_transformers import SentenceTransformer
                print("Memuat model lokal untuk pertanyaan...", flush=True)
                model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
                model.max_seq_length = index["max_sequence_length"]
                if model.get_sentence_embedding_dimension() != index["dimensions"]:
                    raise ValueError("Dimensi model pertanyaan berbeda dari index.")
            tokens = model.tokenizer(question, truncation=False)["input_ids"]
            if len(tokens) > model.max_seq_length:
                raise ValueError("Pertanyaan terlalu panjang; ringkas hingga <= 512 token.")
            vector = np.asarray(model.encode(question, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
            if vector.shape != (index["dimensions"],) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3):
                raise ValueError("Embedding pertanyaan tidak valid.")
            scores = matrix @ vector
            positions = np.argsort(-scores, kind="stable")[:min(args.top_k, len(records))]
            matches = [{**records[int(p)], "similarity": float(scores[p])} for p in positions]
        payload = {
            "question": question, "retrieval_method": method,
            "embedding_model": index["embedding_model"],
            "model_revision": index["model_revision"],
            "missing_customer_ids": unknown, "retrieved_chunks": matches,
            "note": "Similarity is not answer confidence. Inspect evidence; this step does not generate an LLM answer.",
        }
        output = data_folder / "retrieval_output.json"
        write_json(output, payload)
        print(f"\nPertanyaan: {question}")
        print(f"Metode: {method}")
        if unknown:
            print("ID tidak ditemukan: " + ", ".join(unknown))
        for rank, match in enumerate(matches, start=1):
            print(f"\nHASIL {rank}")
            print(f"Customer ID: {match['customer_id']}")
            if match["similarity"] is not None:
                print(f"Cosine similarity: {match['similarity']:.4f}")
            print(f"Sumber: {match['source']}")
            print(f"Chunk: {match['chunk_id']}")
            print(f"Nomor record: {match.get('record_number', 'Tidak tersedia')}")
            print(match["text"])
        print(f"\nHasil tersimpan: {output}")
        print("Periksa apakah teks benar-benar mendukung pertanyaan. Skor bukan probabilitas kebenaran.")

    if args.question is not None:
        if not args.question.strip():
            raise ValueError("Pertanyaan tidak boleh kosong.")
        retrieve(args.question.strip())
    else:
        print("Ketik pertanyaan. Ketik keluar untuk berhenti.")
        while True:
            question = input("\nPertanyaan > ").strip()
            if question.lower() in {"keluar", "exit", "quit"}:
                break
            if question:
                retrieve(question)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nSelesai.")
    except ModuleNotFoundError:
        print("Instal paket: python -m pip install --upgrade sentence-transformers numpy", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
