"""
face_engine.py — InsightFace model wrapper for extracting face embeddings.

Provides two main functions:
- init_model(): Initialize the model once at server startup.
- extract_face(image_bytes): Extract the best face and its embedding from an uploaded image.
"""
import io
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ------- Settings -------
MODEL_NAME    = "antelopev2"
DET_SIZE      = (640, 640)
DET_SCORE_MIN = 0.70
MIN_FACE_PX   = 40          # Minimum face size in pixels
BLUR_LAP_THR  = 10.0        # Laplacian threshold for rejecting blurry images

_app: FaceAnalysis | None = None


# ============================================================
#  Model initialization
# ============================================================

def init_model():
    """
    Load the InsightFace model.
    Called only once at server startup within Lifespan.
    """
    global _app
    _app = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _app.prepare(ctx_id=0, det_size=DET_SIZE)


# ============================================================
#  Helper functions
# ============================================================

def _is_blurry(face_img: np.ndarray, threshold: float = BLUR_LAP_THR) -> bool:
    """True if the image is too blurry."""
    if face_img is None or face_img.size == 0:
        return True
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to a BGR array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to read the image — make sure the file format is valid.")
    return img


def _crop_face(img_bgr: np.ndarray, bbox: np.ndarray, margin: float = 0.40) -> np.ndarray:
    """Crop the face with an additional margin."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return img_bgr[y1:y2, x1:x2].copy()


# ============================================================
#  Main function
# ============================================================

def extract_face(image_bytes: bytes) -> dict:
    """
    Extract the best face from the uploaded image.

    Returns dict containing:
        - embedding (np.ndarray 512D, L2-normalized)
        - crop_bytes (bytes) cropped face image in JPEG format
        - bbox (tuple)
        - det_score (float)

    Raises:
        ValueError: If no clear face is found or the image is blurry.
    """
    if _app is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")

    img_bgr = _bytes_to_bgr(image_bytes)
    faces = _app.get(img_bgr)

    # --- Filter faces ---
    valid_faces = []
    for face in faces:
        if face.det_score < DET_SCORE_MIN:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
            continue
        crop = _crop_face(img_bgr, face.bbox)
        if _is_blurry(crop):
            continue
        valid_faces.append((face, crop))

    if not valid_faces:
        raise ValueError(
            "No clear face detected in the image. "
            "Make sure the image is clear and contains a human face."
        )

    # Select the clearest face (highest det_score)
    best_face, best_crop = max(valid_faces, key=lambda t: t[0].det_score)

    # L2-normalize the embedding
    emb = best_face.embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    # Convert crop to JPEG bytes for storage
    _, jpg_buf = cv2.imencode(
        ".jpg", best_crop, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    x1, y1, x2, y2 = best_face.bbox.astype(int)
    return {
        "embedding":   emb,
        "crop_bytes":  jpg_buf.tobytes(),
        "bbox":        (x1, y1, x2, y2),
        "det_score":   float(best_face.det_score),
    }


def extract_all_faces(image_bytes: bytes) -> list[dict]:
    """
    Extract all faces from the image.
    Useful for batch indexing of group photos.

    Returns:
        List of dicts, each containing:
        embedding, crop_bytes, bbox, det_score
    """
    if _app is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")

    img_bgr = _bytes_to_bgr(image_bytes)
    faces = _app.get(img_bgr)

    results = []
    for face in faces:
        if face.det_score < DET_SCORE_MIN:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
            continue
        crop = _crop_face(img_bgr, face.bbox)
        if _is_blurry(crop):
            continue

        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        _, jpg_buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        results.append({
            "embedding":  emb,
            "crop_bytes": jpg_buf.tobytes(),
            "bbox":       (x1, y1, x2, y2),
            "det_score":  float(face.det_score),
        })

    return results
