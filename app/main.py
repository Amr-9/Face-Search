"""
main.py — Face Search Engine
=============================================
Built on:
  - FastAPI  : Web Framework
  - InsightFace (antelopev2): Face embedding extraction
  - FAISS   : Vector search engine
  - SQLite  : Person data storage

Endpoints:
  POST /faces/index   — Index a new face
  POST /faces/search  — Search for an unknown face
  GET  /faces         — List all indexed persons
  DELETE /faces/{id}  — Delete a person from the index
  GET  /faces/stats   — Index statistics
"""
import sys
import io
import warnings
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Add NVIDIA CUDA DLL paths
_nvidia_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia")
if os.path.isdir(_nvidia_dir):
    for _pkg in os.listdir(_nvidia_dir):
        _bin = os.path.join(_nvidia_dir, _pkg, "bin")
        if os.path.isdir(_bin):
            os.add_dll_directory(_bin)
            os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

import io
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import database
from app import face_engine
from app import vector_store

# ============================================================
#  Core directories
# ============================================================
os.makedirs("models",  exist_ok=True)
os.makedirs("storage", exist_ok=True)

STORAGE_DIR = "storage"

# Cosine similarity threshold for antelopev2 — below this is "no match".
# Maps raw score [0.3 → 1.0] to confidence [0% → 100%].
_FACE_THRESHOLD = 0.30


def _to_confidence(cosine_sim: float) -> float:
    """Convert raw cosine similarity to a user-friendly 0–1 confidence score."""
    return round(max(0.0, (cosine_sim - _FACE_THRESHOLD) / (1.0 - _FACE_THRESHOLD)), 4)


# ============================================================
#  Lifespan — Model and database initialization
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executed once when the server starts."""
    print("⏳  Loading InsightFace model…", flush=True)
    face_engine.init_model()
    print("✅  Model ready.", flush=True)

    print("⏳  Initializing SQLite database…", flush=True)
    database.init_db()
    print("✅  Database ready.", flush=True)

    print("⏳  Loading FAISS index…", flush=True)
    vector_store.init_index()
    total = vector_store.get_total()
    print(f"✅  FAISS index ready — {total} face(s) indexed.", flush=True)

    yield  # Server is running here

    print("🔴  Shutting down server.", flush=True)


# ============================================================
#  Application definition
# ============================================================

app = FastAPI(
    title="Face Search Engine",
    description="Face search engine using InsightFace + FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
app.mount("/static",  StaticFiles(directory="static"),   name="static")


# ============================================================
#  Helper
# ============================================================

def _save_image(image_bytes: bytes, filename: str) -> str:
    """Save the image to the storage folder and return the relative path."""
    path = os.path.join(STORAGE_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


# ============================================================
#  POST /faces/index — Index a new face
# ============================================================

@app.post("/faces/index", summary="Index one or more faces")
async def index_face(
    images: List[UploadFile] = File(..., description="One or more images containing faces"),
) -> dict[str, Any]:
    """
    Indexing steps:
    1. Read each uploaded image.
    2. Extract all faces (512D) via InsightFace.
    3. For each face: save to SQLite and FAISS with a unique ID.
    """
    if not images:
        raise HTTPException(status_code=422, detail="No images provided.")

    indexed = []
    skipped = []

    for image in images:
        image_bytes = await image.read()
        if not image_bytes:
            skipped.append({"filename": image.filename, "reason": "Empty file"})
            continue

        ext = os.path.splitext(image.filename or "face.jpg")[1] or ".jpg"

        try:
            results = face_engine.extract_all_faces(image_bytes)
        except (ValueError, RuntimeError) as e:
            skipped.append({"filename": image.filename, "reason": str(e)})
            continue

        if not results:
            skipped.append({"filename": image.filename, "reason": "No clear face detected"})
            continue

        for face_result in results:
            unique_name = f"{uuid.uuid4().hex}{ext}"
            crop_path   = _save_image(face_result["crop_bytes"], unique_name)
            person_id   = database.add_person(name="", image_path=crop_path)
            vector_store.add_embedding(person_id=person_id, embedding=face_result["embedding"])
            indexed.append({
                "person_id":  person_id,
                "image_path": f"/storage/{unique_name}",
                "det_score":  face_result["det_score"],
                "source":     image.filename,
            })

    if not indexed and skipped:
        reasons = "; ".join(s["reason"] for s in skipped)
        raise HTTPException(status_code=400, detail=f"No faces indexed. {reasons}")

    return {
        "status":           "indexed",
        "faces_found":      len(indexed),
        "images_processed": len(images) - len(skipped),
        "skipped":          skipped,
        "indexed":          indexed,
        "total_indexed":    vector_store.get_total(),
    }


# ============================================================
#  POST /faces/search — Search for an unknown face
# ============================================================

@app.post("/faces/search", summary="Search for an unknown face")
async def search_face(
    image: UploadFile = File(..., description="Image containing the face to search for"),
    top_k: int = Form(default=5, ge=1, le=50, description="Number of results to return"),
) -> dict[str, Any]:
    """
    Search steps:
    1. Extract the unknown face embedding.
    2. Query FAISS for the closest top_k faces.
    3. Fetch person names from SQLite.
    4. Return results in JSON format.
    """
    if vector_store.get_total() == 0:
        raise HTTPException(
            status_code=404,
            detail="No faces indexed yet. Index some faces first via /faces/index."
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    try:
        result = face_engine.extract_face(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    matches = vector_store.search(result["embedding"], top_k=top_k)
    if not matches:
        return {"query_det_score": result["det_score"], "results": []}

    # Fetch person data in a single batch from SQLite
    ids = [m["person_id"] for m in matches]
    persons = database.get_persons_by_ids(ids)

    results = []
    for m in matches:
        pid = m["person_id"]
        person = persons.get(pid)
        if not person:
            continue
        results.append({
            "person_id":   pid,
            "name":        person["name"],
            "similarity":  _to_confidence(m["score"]),  # 1.0 = exact match
            "image_path":  f"/{person['image_path'].replace(os.sep, '/')}",
            "created_at":  person["created_at"],
        })

    return {
        "query_det_score": result["det_score"],
        "results": results,
    }


# ============================================================
#  GET /faces — List all indexed persons
# ============================================================

@app.get("/faces", summary="All indexed persons")
async def list_faces(
    skip:  int = Query(default=0,      ge=0,             description="Offset"),
    limit: int = Query(default=200,    ge=1,   le=1000,  description="Page size"),
    order: str = Query(default="desc", pattern="^(asc|desc)$", description="Sort order"),
) -> dict[str, Any]:
    result = database.get_persons_paginated(skip=skip, limit=limit, order=order)
    for p in result["persons"]:
        p["image_path"] = f"/{p['image_path'].replace(os.sep, '/')}"
    return {"total": result["total"], "persons": result["persons"]}


# ============================================================
#  DELETE /faces/{id} — Delete a person
# ============================================================

@app.delete("/faces/{person_id}", summary="Delete a person from the index")
async def delete_face(person_id: int) -> dict[str, Any]:
    person = database.get_person_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"No person found with ID {person_id}.")

    # Remove from FAISS
    vector_store.remove_embedding(person_id)

    # Remove from SQLite
    database.delete_person(person_id)

    # Delete image file if it exists
    img_path = person["image_path"]
    if os.path.exists(img_path):
        try:
            os.remove(img_path)
        except OSError:
            pass

    return {
        "status":    "deleted",
        "person_id": person_id,
        "name":      person["name"],
        "remaining": vector_store.get_total(),
    }


# ============================================================
#  GET /faces/stats — Index statistics
# ============================================================

@app.get("/faces/stats", summary="Index statistics")
async def get_stats() -> dict[str, Any]:
    return {
        "total_indexed": vector_store.get_total(),
        "db_persons":    len(database.get_all_persons()),
        "storage_dir":   os.path.abspath(STORAGE_DIR),
        "index_path":    os.path.abspath(vector_store.INDEX_PATH),
    }


# ============================================================
#  GET /backup/download — Download full backup as ZIP
# ============================================================

@app.get("/backup/download", summary="Download full backup as ZIP")
async def backup_download():
    """
    Creates a ZIP containing:
      - faces.db    (SQLite database)
      - faces.index (FAISS vector index)
      - storage/    (all face crop images)
    WAL checkpoint is performed first to ensure DB consistency.
    """
    # Flush WAL so faces.db contains all committed data
    try:
        with database.get_connection() as conn:
            conn.execute("PRAGMA wal_checkpoint(FULL)")
    except Exception:
        pass

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(database.DB_PATH):
            zf.write(database.DB_PATH, "faces.db")
        if os.path.exists(vector_store.INDEX_PATH):
            zf.write(vector_store.INDEX_PATH, "faces.index")
        if os.path.isdir(STORAGE_DIR):
            for fname in os.listdir(STORAGE_DIR):
                fpath = os.path.join(STORAGE_DIR, fname)
                if os.path.isfile(fpath):
                    zf.write(fpath, f"storage/{fname}")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=face-search-backup-{timestamp}.zip"},
    )


# ============================================================
#  POST /backup/restore — Restore from a backup ZIP
# ============================================================

@app.post("/backup/restore", summary="Restore from a backup ZIP")
async def backup_restore(
    backup_file: UploadFile = File(...),
    mode: str = Form(default="replace", description="'replace' clears existing data; 'merge' adds backup on top"),
) -> dict[str, Any]:
    """
    Accepts a ZIP produced by /backup/download.

    mode=replace  — Deletes all current data then restores from backup (default).
    mode=merge    — Keeps current data and appends backup persons on top.
                    Duplicate faces may appear if the same person is in both datasets.
    """
    if mode not in ("replace", "merge"):
        raise HTTPException(status_code=422, detail="mode must be 'replace' or 'merge'.")

    data = await backup_file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty file.")

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            if "faces.db" not in names:
                raise HTTPException(status_code=400, detail="Invalid backup: faces.db missing.")
            if "faces.index" not in names:
                raise HTTPException(status_code=400, detail="Invalid backup: faces.index missing.")

            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp)

                if mode == "replace":
                    # ── Replace mode: wipe everything, restore from backup ──
                    shutil.copy2(os.path.join(tmp, "faces.db"),    database.DB_PATH)
                    shutil.copy2(os.path.join(tmp, "faces.index"), vector_store.INDEX_PATH)
                    storage_src = os.path.join(tmp, "storage")
                    if os.path.isdir(storage_src):
                        for f in os.listdir(STORAGE_DIR):
                            fp = os.path.join(STORAGE_DIR, f)
                            if os.path.isfile(fp):
                                os.remove(fp)
                        for f in os.listdir(storage_src):
                            shutil.copy2(
                                os.path.join(storage_src, f),
                                os.path.join(STORAGE_DIR, f),
                            )
                    vector_store.reload_index()

                else:
                    # ── Merge mode: keep existing data, add backup on top ──
                    import sqlite3

                    # Read all persons from backup DB
                    backup_conn = sqlite3.connect(os.path.join(tmp, "faces.db"))
                    backup_conn.row_factory = sqlite3.Row
                    backup_persons = backup_conn.execute(
                        "SELECT id, name, image_path, created_at FROM persons ORDER BY id"
                    ).fetchall()
                    backup_conn.close()

                    # Extract all embeddings from backup FAISS index
                    id_to_vec = vector_store.load_all_embeddings(
                        os.path.join(tmp, "faces.index")
                    )

                    storage_src = os.path.join(tmp, "storage")
                    batch: list[tuple[int, Any]] = []

                    for person in backup_persons:
                        old_id = person["id"]
                        emb = id_to_vec.get(old_id)
                        if emb is None:
                            continue  # No embedding — skip

                        # Copy image (generate new name if filename already exists)
                        old_fname = os.path.basename(person["image_path"])
                        src = os.path.join(storage_src, old_fname)
                        if not os.path.exists(src):
                            continue  # Image missing in backup — skip

                        dst_fname = old_fname
                        if os.path.exists(os.path.join(STORAGE_DIR, dst_fname)):
                            ext = os.path.splitext(old_fname)[1]
                            dst_fname = f"{uuid.uuid4().hex}{ext}"
                        shutil.copy2(src, os.path.join(STORAGE_DIR, dst_fname))

                        img_path = f"storage/{dst_fname}"
                        new_id = database.add_person(name=person["name"], image_path=img_path)
                        batch.append((new_id, emb))

                    vector_store.add_embeddings_batch(batch)

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File is not a valid ZIP archive.")

    added = vector_store.get_total()
    return {"status": mode, "total_indexed": added}


# ============================================================
#  GET / — Serve the frontend
# ============================================================

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse("static/index.html")


# ============================================================
#  Run the server directly
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
