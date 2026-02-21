# 🔍 Face Search Engine

> Index faces once. Find them instantly — no matter the angle, lighting, or crowd size.

Built on **InsightFace (antelopev2)**, **FAISS**, and **FastAPI**. Runs fully locally, no cloud, no API keys.

---

## 🖼️ Screenshots

### Index Tab — Upload & Index Faces
![Index Tab](assets/Screenshots/index-tab.png)

### Indexed Faces — Browse & Manage
![Indexed Faces](assets/Screenshots/indexed-faces.png)

### Search Results
![Search Results](assets/Screenshots/search-results.png)

---

## ✨ Why this project?

Most face recognition tools are either cloud-only, single-face, or require heavy ML expertise to set up.
This engine gives you a **self-hosted, production-quality** face search system you can run on your own machine in minutes.

| | This project |
|---|---|
| Privacy | 100% local — no data leaves your machine |
| Accuracy | antelopev2 — state-of-the-art recognition model |
| Speed | FAISS vector search scales to **millions of faces** |
| Persistence | Index survives server restarts (SQLite + FAISS on disk) |
| Multi-face | Indexes **every face** in a photo, not just one |
| UI included | Full browser interface — no API knowledge needed |

---

## 📐 How it works

```
Upload image
    → InsightFace detects & embeds each face (512D vector, L2-normalized)
    → Embedding saved to FAISS index (cosine similarity search)
    → Metadata (image path, timestamp) saved to SQLite
    → At search time: query embedding compared against all indexed vectors
    → Returns ranked matches with similarity score (0.0 – 1.0)
```

---

## 📁 Project Structure

```
Face Search Engine/
├── app/
│   ├── main.py          # FastAPI — all endpoints
│   ├── face_engine.py   # InsightFace wrapper — detection & embedding
│   ├── database.py      # SQLite — person metadata
│   └── vector_store.py  # FAISS — vector index (save/load/search)
├── scripts/
│   └── batch_index.py   # Bulk-index an entire folder of images
├── static/
│   └── index.html       # Browser UI
├── assets/
│   └── Screenshots/
└── requirements.txt
```

> `models/` and `storage/` are created automatically on first run.

---

## 🚀 Getting Started

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA 12 *(CPU-only works too, just slower)*

### Install

```bash
pip install -r requirements.txt
```

> For GPU-accelerated FAISS (Linux only): replace `faiss-cpu` with `faiss-gpu` in `requirements.txt`.

### Run

```bash
python -m app.main
```

Open **http://localhost:8000** in your browser.
Swagger docs at **http://localhost:8000/docs**.

---

## 🔌 API

| Method | Endpoint | What it does |
|--------|----------|-------------|
| `POST` | `/faces/index` | Detect & index all faces in an uploaded image |
| `POST` | `/faces/search` | Find the closest match for an uploaded face |
| `GET` | `/faces` | List indexed faces (paginated, sortable) |
| `DELETE` | `/faces/{id}` | Remove a face from the index |
| `GET` | `/faces/stats` | Index statistics |

**Index response example** — 3 faces found in one photo:
```json
{
  "status": "indexed",
  "faces_found": 3,
  "indexed": [
    {"person_id": 1, "det_score": 0.98},
    {"person_id": 2, "det_score": 0.96},
    {"person_id": 3, "det_score": 0.94}
  ]
}
```

---

## ⚡ Batch Indexing

Index an entire folder of photos at once:

```bash
python scripts/batch_index.py
# or with options:
python scripts/batch_index.py --input input_images --workers 8
```

- Supports `jpg`, `jpeg`, `png`, `bmp`, `webp`
- Indexes every face per image (group photos included)
- Shows a live progress bar with per-image stats

---

## ⚙️ Tunable Settings (`face_engine.py`)

| Setting | Default | Effect |
|---------|---------|--------|
| `DET_SCORE_MIN` | `0.70` | Raise to skip low-confidence detections |
| `MIN_FACE_PX` | `40px` | Ignore faces smaller than this (blurry thumbnails) |
| `BLUR_LAP_THR` | `10.0` | Raise to be stricter about image sharpness |
| `DET_SIZE` | `640×640` | Higher = better accuracy, slower on CPU |

---

## 🔧 Technical Notes

- **Cosine similarity** via `IndexFlatIP` — scores range from `0.0` (no match) to `1.0` (identical).
- **Dual storage** — SQLite holds metadata; FAISS holds vectors. Keeps search fast even as the index grows.
- **Local only** — server binds to `127.0.0.1`, not reachable from the network by default.
- **Windows + CUDA** — NVIDIA DLL paths are auto-injected at startup so `onnxruntime-gpu` works without manual PATH changes.
