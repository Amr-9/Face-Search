"""
batch_index.py — Fast batch indexing from the input_images folder
=================================================================
Usage:
    python batch_index.py
    python batch_index.py --input input_images --workers 4

Steps:
  1. Load the model, database, and FAISS index.
  2. Read all images from the --input folder.
  3. Extract face embeddings for each image (multi-threaded GPU batching).
  4. Save the embedding and cropped face to storage/.
  5. Delete the original image from --input after successful indexing.
"""
import sys
import io
import warnings
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# NVIDIA CUDA DLLs
_nvidia_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia")
if os.path.isdir(_nvidia_dir):
    for _pkg in os.listdir(_nvidia_dir):
        _bin = os.path.join(_nvidia_dir, _pkg, "bin")
        if os.path.isdir(_bin):
            os.add_dll_directory(_bin)
            os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

import argparse
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import sys

# Add parent directory to sys.path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database
from app import face_engine
from app import vector_store

# ── Default settings ──
DEFAULT_INPUT   = "input_images"
DEFAULT_STORAGE = "storage"
SUPPORTED_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── ANSI colors ──
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ============================================================
#  Progress bar
# ============================================================

def _progress(current: int, total: int, ok: int, skip: int, err: int, elapsed: float):
    bar_w = 40
    frac  = current / total if total else 0
    filled = int(bar_w * frac)
    bar  = "█" * filled + "░" * (bar_w - filled)
    pct  = frac * 100
    eta  = (elapsed / current * (total - current)) if current else 0
    t_el = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
    t_et = f"{int(eta//60):02d}:{int(eta%60):02d}"
    sys.stdout.write(
        f"\r{CYAN}[{bar}]{RESET} {current}/{total} ({pct:.1f}%)  "
        f"{GREEN}✔{ok}{RESET}  {YELLOW}⚠{skip}{RESET}  {RED}✖{err}{RESET}  "
        f"⏱{t_el}  ETA {t_et}   "
    )
    sys.stdout.flush()


# ============================================================
#  Process a single image (executed in a Thread)
# ============================================================

def _process_image(image_path: str, storage_dir: str) -> dict:
    """
    Returns:
        {"status": "ok"|"skip"|"error", "count": int, "msg": str}
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Extract all faces
        faces = face_engine.extract_all_faces(image_bytes)

        if not faces:
            return {"status": "skip", "count": 0,
                    "msg": "No clear face detected"}

        ext = os.path.splitext(image_path)[1] or ".jpg"

        for face_result in faces:
            # Save cropped face
            unique_name = f"{uuid.uuid4().hex}{ext}"
            crop_path   = os.path.join(storage_dir, unique_name)
            with open(crop_path, "wb") as f:
                f.write(face_result["crop_bytes"])

            # Add to database
            person_id = database.add_person(name="", image_path=crop_path)

            # Add to FAISS
            vector_store.add_embedding(
                person_id=person_id,
                embedding=face_result["embedding"]
            )

        # Delete original image after success
        os.remove(image_path)

        return {"status": "ok", "count": len(faces), "msg": ""}

    except Exception as e:
        return {"status": "error", "count": 0, "msg": str(e)}


# ============================================================
#  Main function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch face indexing from the input_images folder"
    )
    parser.add_argument("--input",   default=DEFAULT_INPUT,   help="Input images folder")
    parser.add_argument("--storage", default=DEFAULT_STORAGE, help="Storage folder for cropped faces")
    parser.add_argument("--workers", type=int, default=4,     help="Number of parallel workers")
    args = parser.parse_args()

    input_dir   = args.input
    storage_dir = args.storage
    workers     = args.workers

    if not os.path.isdir(input_dir):
        print(f"{RED}✖ Folder '{input_dir}' does not exist.{RESET}")
        sys.exit(1)

    os.makedirs(storage_dir, exist_ok=True)
    os.makedirs("models",    exist_ok=True)

    # Collect images
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ]
    total = len(files)

    if not total:
        print(f"{YELLOW}⚠ No images found in '{input_dir}'.{RESET}")
        sys.exit(0)

    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"  {BOLD}Face Search Engine — Batch Indexer{RESET}")
    print(f"  Images found    : {BOLD}{total}{RESET}")
    print(f"  Folder          : {input_dir}")
    print(f"  Workers         : {workers}")
    print(f"{BOLD}{'='*55}{RESET}\n")

    # Initialize model and database
    print("⏳  Loading InsightFace…", end="", flush=True)
    face_engine.init_model()
    print(f" {GREEN}✔{RESET}")

    print("⏳  Initializing SQLite…", end="", flush=True)
    database.init_db()
    print(f" {GREEN}✔{RESET}")

    print("⏳  Loading FAISS Index…", end="", flush=True)
    vector_store.init_index()
    print(f" {GREEN}✔{RESET}\n")

    # ──────────── Parallel processing ────────────
    ok_count   = 0
    skip_count = 0
    err_count  = 0
    errors     = []
    skips      = []
    start      = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_process_image, fp, storage_dir): fp
            for fp in files
        }
        done = 0
        for future in as_completed(future_map):
            done += 1
            fp  = future_map[future]
            res = future.result()

            if res["status"] == "ok":
                ok_count += 1
            elif res["status"] == "skip":
                skip_count += 1
                skips.append((os.path.basename(fp), res["msg"]))
            else:
                err_count += 1
                errors.append((os.path.basename(fp), res["msg"]))

            _progress(done, total, ok_count, skip_count, err_count, time.time() - start)

    elapsed = time.time() - start
    print(f"\n\n{BOLD}{'='*55}{RESET}")
    print(f"  {GREEN}✔ Indexed successfully : {ok_count}{RESET}")
    print(f"  {YELLOW}⚠ Skipped             : {skip_count}{RESET}  (no clear face)")
    print(f"  {RED}✖ Errors              : {err_count}{RESET}")
    print(f"  Total indexed        : {vector_store.get_total()}")
    print(f"  Time                 : {elapsed:.1f}s  ({ok_count/elapsed:.1f} faces/s)")
    print(f"{BOLD}{'='*55}{RESET}\n")

    if skips:
        print(f"\n{YELLOW}Skipped images:{RESET}")
        for name, msg in skips[:20]:
            print(f"  ⚠  {name}: {msg}")
        if len(skips) > 20:
            print(f"  ... and {len(skips)-20} more")

    if errors:
        print(f"\n{RED}Errors:{RESET}")
        for name, msg in errors[:20]:
            print(f"  ✖  {name}: {msg}")


if __name__ == "__main__":
    main()
