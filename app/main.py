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

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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

# Serve index.html directly from root
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")


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

@app.post("/faces/index", summary="Index a new face")
async def index_face(
    image: UploadFile = File(..., description="Image containing one or more faces"),
) -> dict[str, Any]:
    """
    Indexing steps:
    1. Read the uploaded image.
    2. Extract all faces (512D) via InsightFace.
    3. For each face: save to SQLite and FAISS with a unique ID.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    # Extract all faces
    try:
        results = face_engine.extract_all_faces(image_bytes)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not results:
        raise HTTPException(
            status_code=400,
            detail="No clear face detected in the image."
        )

    ext = os.path.splitext(image.filename or "face.jpg")[1] or ".jpg"
    indexed = []

    for face_result in results:
        # Save cropped face with a unique UUID
        unique_name = f"{uuid.uuid4().hex}{ext}"
        crop_path   = _save_image(face_result["crop_bytes"], unique_name)

        # Add face to database
        person_id = database.add_person(name="", image_path=crop_path)

        # Add embedding to FAISS
        vector_store.add_embedding(person_id=person_id, embedding=face_result["embedding"])

        indexed.append({
            "person_id":  person_id,
            "image_path": f"/storage/{unique_name}",
            "det_score":  face_result["det_score"],
        })

    return {
        "status":        "indexed",
        "faces_found":   len(indexed),
        "indexed":       indexed,
        "total_indexed": vector_store.get_total(),
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
            "similarity":  m["score"],         # 1.0 = exact match
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
