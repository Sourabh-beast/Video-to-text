# =============================================================================
# Flask Web Application
# Real-Time OCR System for Video-Based Text Recognition
# in Traffic and Surveillance
# =============================================================================
#
# OCR Engine: EasyOCR (CRAFT detector + CRNN recognizer)
#   - CRAFT: text area detection
#   - CRNN: text recognition
#   - Works on CPU, no GPU required
#   - Supports manual rectangle selection
#
# Technologies: Python, Flask, OpenCV, EasyOCR, NumPy
# =============================================================================

import os

from flask import Flask, render_template, request, jsonify, send_from_directory  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
import easyocr  # type: ignore
try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    YOLO = None
    YOLO_IMPORT_ERROR = str(exc)
else:
    YOLO_IMPORT_ERROR = None
import re
import time
import base64
import uuid
import hashlib

app = Flask(__name__)

# =============================================================================
# EasyOCR Initialization (loaded ONCE at startup)
# =============================================================================
# EasyOCR uses CRAFT for detection + CRNN for recognition.
# First run downloads model files (~1-2GB) — one-time download.
# =============================================================================

print("=" * 60)
print("  Loading EasyOCR Models (CRAFT + CRNN)...")
print("  First run downloads models — one-time ~1-2GB")
print("=" * 60)

# Initialize EasyOCR
# - lang_list = ['en']
# - gpu=False for CPU-only
ocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)

print("[INFO] EasyOCR loaded successfully!")

# ─── Upload folder ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# ─── OCR Quality Constants ────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.12
MANUAL_CONFIDENCE_MIN = 0.45
MANUAL_MAX_RESULTS = 16
MANUAL_OCR_MAX_SIDE = 1600
MANUAL_OCR_UPSCALE_MIN = 800
MIN_MANUAL_LINES = 4
EASYOCR_TEXT_THRESHOLD = 0.65
EASYOCR_LOW_TEXT = 0.35
EASYOCR_LINK_THRESHOLD = 0.40
EASYOCR_MAG_RATIO = 1.2
EASYOCR_CANVAS_SIZE = 1600
EASYOCR_CONTRAST_THS = 0.08
EASYOCR_ADJUST_CONTRAST = 0.55
EASYOCR_DECODER = "greedy"
EASYOCR_FALLBACK = True
EASYOCR_FALLBACK_TEXT_THRESHOLD = 0.55
EASYOCR_FALLBACK_LOW_TEXT = 0.25
EASYOCR_FALLBACK_LINK_THRESHOLD = 0.30
EASYOCR_FALLBACK_MAG_RATIO = 1.4
EASYOCR_FALLBACK_CANVAS_SIZE = 2000
EASYOCR_FALLBACK_DECODER = "beamsearch"
OCR_DET_LIMIT_SIDE = 1920
OCR_DET_THRESH = 0.10
OCR_DET_BOX_THRESH = 0.20
OCR_DET_UNCLIP = 1.8
OCR_REC_SCORE_THRESH = 0.10
OCR_UPSCALE_MAX = 1920
OCR_TILE_MODE = True
OCR_TILE_COLS = 2
OCR_TILE_ROWS = 2
OCR_TILE_OVERLAP = 0.12
OCR_TILE_MIN_W = 900
OCR_TILE_MIN_H = 600
MAX_RETURN_DETECTIONS = 20
FORCE_MAX_OCR = True

PLATE_CACHE_ENABLED = True
PLATE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plate_cache')
PLATE_CACHE_MAX = 12
PLATE_CACHE_FRAME_LIMIT = 4
PLATE_CACHE_EXPAND = 0.06
PLATE_CACHE_SAVE = True

QUALITY_BLUR_MIN = 80.0
QUALITY_BRIGHTNESS_MIN = 40.0
QUALITY_BRIGHTNESS_MAX = 220.0
BEST_FRAME_WINDOW_MS = 1200
OCR_MIN_INTERVAL_MS = 800
OCR_FORCE_EVERY_MS = 2400

PLATE_ONLY_MODE = False
YOLO_PLATE_MODE = False
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
YOLO_MAX_VEHICLES = 10
YOLO_VEHICLE_CLASSES = {2, 3, 5, 7}
YOLO_PLATE_REGION = (0.55, 1.0)
YOLO_PLATE_SIDE_TRIM = 0.08
YOLO_PLATE_EXPAND = 0.06
PLATE_HAAR_MODE = False
PLATE_HAAR_SCALE = 1.1
PLATE_HAAR_MIN_NEIGHBORS = 3
PLATE_HAAR_MIN_SIZE = (40, 12)
PLATE_FALLBACK_FULL = True
PLATE_ROI_FALLBACK = True
PLATE_TILE_FALLBACK = True
PLATE_MAX_CANDIDATES = 20
PLATE_ASPECT_MIN = 1.8
PLATE_ASPECT_MAX = 8.5
PLATE_MIN_AREA_RATIO = 0.0002
PLATE_MAX_AREA_RATIO = 0.12
PLATE_MIN_W = 40
PLATE_MIN_H = 12
PLATE_BOX_EXPAND = 0.12
PLATE_TILE_SIZE = 960
PLATE_TILE_OVERLAP = 0.2
PLATE_NMS_IOU = 0.45
RAW_TEXT_FALLBACK = True

plate_cache = []
if PLATE_CACHE_SAVE:
    os.makedirs(PLATE_CACHE_DIR, exist_ok=True)

best_frame = None
best_frame_score = 0.0
best_frame_time = 0.0
last_ocr_detections = []
last_ocr_zoomed_image = None
last_ocr_zoomed_text = None
last_ocr_time = 0.0

NOISE_PATTERN = re.compile(r'^[\W_]+$')
PLATE_TEXT_PATTERN = re.compile(r'^[A-Z0-9\- ]+$', re.IGNORECASE)

yolo_model = None
if YOLO_PLATE_MODE:
    if YOLO is None:
        print(f"[WARNING] YOLO not available: {YOLO_IMPORT_ERROR}")
    else:
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[INFO] YOLO model loaded: {YOLO_MODEL_PATH}")
        except Exception as exc:
            print(f"[WARNING] YOLO init failed: {exc}")
            yolo_model = None

plate_cascade = None
if PLATE_HAAR_MODE:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
    if os.path.exists(cascade_path):
        plate_cascade = cv2.CascadeClassifier(cascade_path)
        if plate_cascade.empty():
            plate_cascade = None
            print("[WARNING] Haar cascade failed to load.")
        else:
            print("[INFO] Haar cascade loaded for plates.")
    else:
        print("[WARNING] Haar cascade not found for plates.")


def _is_valid_text(text):
    """Check if detected text is real (not noise)."""
    text = text.strip()
    if len(text) < 1:
        return False
    if NOISE_PATTERN.match(text):
        return False
    if not re.search(r'[a-zA-Z0-9]', text):
        return False
    return True


def _is_plate_text(text):
    text = text.strip()
    if len(text) < 3:
        return False
    if not re.search(r'[0-9]', text):
        return False
    if not PLATE_TEXT_PATTERN.match(text):
        return False
    return True


def _hash_image(image) -> str:
    small = cv2.resize(image, (32, 12), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


def _push_plate_cache(crop, text, confidence):
    if crop is None or crop.size == 0:
        return
    key = _hash_image(crop)
    for item in plate_cache:
        if item['hash'] == key:
            if text:
                item['text'] = text
            item['confidence'] = max(item['confidence'], float(confidence))
            item['last_seen'] = time.time()
            return

    filename = None
    if PLATE_CACHE_SAVE:
        filename = f"plate_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
        path = os.path.join(PLATE_CACHE_DIR, filename)
        cv2.imwrite(path, crop)

    plate_cache.insert(0, {
        'hash': key,
        'text': text,
        'confidence': float(confidence),
        'filename': filename,
        'last_seen': time.time(),
    })

    while len(plate_cache) > PLATE_CACHE_MAX:
        removed = plate_cache.pop()
        if PLATE_CACHE_SAVE and removed.get('filename'):
            try:
                os.remove(os.path.join(PLATE_CACHE_DIR, removed['filename']))
            except OSError:
                pass


def _update_plate_cache(frame, detections):
    if not PLATE_CACHE_ENABLED or not detections:
        return
    plate_like = [d for d in detections if _is_plate_text(d.get('text', ''))]
    source = plate_like if plate_like else detections
    source = sorted(source, key=lambda d: d.get('confidence', 0), reverse=True)
    source = source[:PLATE_CACHE_FRAME_LIMIT]

    h, w = frame.shape[:2]
    for det in source:
        x1, y1, x2, y2 = det['bbox']
        x1, y1, x2, y2 = _expand_box((x1, y1, x2, y2), w, h, PLATE_CACHE_EXPAND)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        _push_plate_cache(crop, det.get('text', ''), det.get('confidence', 0.0))


def _serialize_plate_cache():
    items = []
    for item in plate_cache:
        url = f"/plate_cache/{item['filename']}" if item.get('filename') else None
        items.append({
            'url': url,
            'text': item.get('text', ''),
            'confidence': item.get('confidence', 0.0),
        })
    return items


def _frame_quality_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    return blur_score, brightness


def _quality_ok(blur_score, brightness):
    return (
        blur_score >= QUALITY_BLUR_MIN and
        QUALITY_BRIGHTNESS_MIN <= brightness <= QUALITY_BRIGHTNESS_MAX
    )


def _update_best_frame(frame, blur_score, brightness):
    global best_frame, best_frame_score, best_frame_time
    if not _quality_ok(blur_score, brightness):
        return
    if best_frame is None or blur_score >= best_frame_score:
        best_frame = frame.copy()
        best_frame_score = float(blur_score)
        best_frame_time = time.time()


def _prepare_frame_for_ocr(frame):
    h, w = frame.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    if max_side < OCR_UPSCALE_MAX:
        scale = OCR_UPSCALE_MAX / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return frame, scale


def _enhance_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _resize_for_manual_ocr(frame):
    h, w = frame.shape[:2]
    max_side = max(h, w)
    scale = 1.0
    if max_side > MANUAL_OCR_MAX_SIDE:
        scale = MANUAL_OCR_MAX_SIDE / max_side
    elif max_side < MANUAL_OCR_UPSCALE_MIN:
        scale = MANUAL_OCR_UPSCALE_MIN / max_side
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return frame


def _read_easyocr_texts(frame):
    results = ocr_engine.readtext(
        frame,
        text_threshold=EASYOCR_TEXT_THRESHOLD,
        low_text=EASYOCR_LOW_TEXT,
        link_threshold=EASYOCR_LINK_THRESHOLD,
        mag_ratio=EASYOCR_MAG_RATIO,
        canvas_size=EASYOCR_CANVAS_SIZE,
        contrast_ths=EASYOCR_CONTRAST_THS,
        adjust_contrast=EASYOCR_ADJUST_CONTRAST,
        decoder=EASYOCR_DECODER,
    )
    detections = []
    for entry in results:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        text = str(entry[1]).strip()
        conf = float(entry[2]) if entry[2] is not None else 0.0
        if not text:
            continue
        detections.append({
            'text': text,
            'confidence': conf,
        })
    return detections


def _read_easyocr_texts_fallback(frame):
    if not EASYOCR_FALLBACK:
        return []
    results = ocr_engine.readtext(
        frame,
        text_threshold=EASYOCR_FALLBACK_TEXT_THRESHOLD,
        low_text=EASYOCR_FALLBACK_LOW_TEXT,
        link_threshold=EASYOCR_FALLBACK_LINK_THRESHOLD,
        mag_ratio=EASYOCR_FALLBACK_MAG_RATIO,
        canvas_size=EASYOCR_FALLBACK_CANVAS_SIZE,
        contrast_ths=EASYOCR_CONTRAST_THS,
        adjust_contrast=EASYOCR_ADJUST_CONTRAST,
        decoder=EASYOCR_FALLBACK_DECODER,
    )
    detections = []
    for entry in results:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        text = str(entry[1]).strip()
        conf = float(entry[2]) if entry[2] is not None else 0.0
        if not text:
            continue
        detections.append({
            'text': text,
            'confidence': conf,
        })
    return detections


def _build_tiles(frame):
    h, w = frame.shape[:2]
    tile_w = int(np.ceil(w / OCR_TILE_COLS))
    tile_h = int(np.ceil(h / OCR_TILE_ROWS))
    stride_w = max(1, int(tile_w * (1 - OCR_TILE_OVERLAP)))
    stride_h = max(1, int(tile_h * (1 - OCR_TILE_OVERLAP)))
    tiles = []
    for row in range(OCR_TILE_ROWS):
        for col in range(OCR_TILE_COLS):
            x1 = col * stride_w
            y1 = row * stride_h
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            if x2 > w:
                x2 = w
                x1 = max(0, w - tile_w)
            if y2 > h:
                y2 = h
                y1 = max(0, h - tile_h)
            if x2 <= x1 or y2 <= y1:
                continue
            tiles.append((frame[y1:y2, x1:x2], x1, y1))
    return tiles


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter_area / float(area_a + area_b - inter_area + 1e-6)


def _merge_overlaps(boxes, iou_thresh=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    merged = []
    for box in boxes:
        merged_box = None
        for idx, existing in enumerate(merged):
            if _iou(box, existing) >= iou_thresh:
                merged_box = (
                    min(box[0], existing[0]),
                    min(box[1], existing[1]),
                    max(box[2], existing[2]),
                    max(box[3], existing[3]),
                )
                merged[idx] = merged_box
                break
        if merged_box is None:
            merged.append(box)
    return merged


def _nms_detections(detections, iou_thresh):
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    keep = []
    for det in ordered:
        duplicate = False
        for kept in keep:
            if _iou(det['bbox'], kept['bbox']) >= iou_thresh and det['text'] == kept['text']:
                duplicate = True
                break
        if not duplicate:
            keep.append(det)
    return keep


def _expand_box(box, frame_w, frame_h, expand_ratio):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * expand_ratio)
    pad_h = int(bh * expand_ratio)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(frame_w, x2 + pad_w)
    ny2 = min(frame_h, y2 + pad_h)
    return nx1, ny1, nx2, ny2


def _find_plate_candidates(frame):
    h, w = frame.shape[:2]
    min_area = int(h * w * PLATE_MIN_AREA_RATIO)
    max_area = int(h * w * PLATE_MAX_AREA_RATIO)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 15, 15)
    edges = cv2.Canny(gray, 40, 120)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        9,
    )
    mask = cv2.bitwise_or(edges, thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        if bw < PLATE_MIN_W or bh < PLATE_MIN_H:
            continue
        aspect = bw / float(bh + 1e-6)
        if aspect < PLATE_ASPECT_MIN or aspect > PLATE_ASPECT_MAX:
            continue
        candidates.append((x, y, x + bw, y + bh))

    candidates = _merge_overlaps(candidates)
    candidates = sorted(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return candidates[:PLATE_MAX_CANDIDATES]


def _get_plate_rois(frame):
    h, w = frame.shape[:2]
    rois = []
    rois.append((0, int(h * 0.45), w, h))
    rois.append((0, int(h * 0.55), w, h))
    rois.append((int(w * 0.1), int(h * 0.45), int(w * 0.9), h))
    rois.append((int(w * 0.2), int(h * 0.5), int(w * 0.8), h))
    normalized = []
    for x1, y1, x2, y2 in rois:
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        normalized.append((x1, y1, x2, y2))
    return normalized


def _vehicle_plate_roi(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(frame_w - 1, int(x1)))
    y1 = max(0, min(frame_h - 1, int(y1)))
    x2 = max(x1 + 1, min(frame_w, int(x2)))
    y2 = max(y1 + 1, min(frame_h, int(y2)))

    bw = x2 - x1
    bh = y2 - y1
    trim = int(bw * YOLO_PLATE_SIDE_TRIM)
    rx1 = x1 + trim
    rx2 = x2 - trim
    ry1 = y1 + int(bh * YOLO_PLATE_REGION[0])
    ry2 = y1 + int(bh * YOLO_PLATE_REGION[1])
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(frame_w, rx2), min(frame_h, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return _expand_box((rx1, ry1, rx2, ry2), frame_w, frame_h, YOLO_PLATE_EXPAND)


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _is_bbox_in_vehicle_plate_region(bbox, vehicle_box):
    vx1, vy1, vx2, vy2 = vehicle_box
    cx, cy = _bbox_center(bbox)
    if cx < vx1 or cx > vx2 or cy < vy1 or cy > vy2:
        return False
    vh = vy2 - vy1
    return cy >= vy1 + vh * YOLO_PLATE_REGION[0]


def _tile_boxes(frame_h, frame_w, tile_size, overlap):
    if frame_h <= tile_size and frame_w <= tile_size:
        return [(0, 0, frame_w, frame_h)]
    stride = max(int(tile_size * (1.0 - overlap)), 1)
    boxes = []
    y = 0
    while y < frame_h:
        x = 0
        y2 = min(frame_h, y + tile_size)
        if y2 - y < int(tile_size * 0.6):
            y = max(0, frame_h - tile_size)
            y2 = frame_h
        while x < frame_w:
            x2 = min(frame_w, x + tile_size)
            if x2 - x < int(tile_size * 0.6):
                x = max(0, frame_w - tile_size)
                x2 = frame_w
            boxes.append((x, y, x2, y2))
            if x2 == frame_w:
                break
            x += stride
        if y2 == frame_h:
            break
        y += stride
    return boxes


# =============================================================================
# EasyOCR Runner — Automatic Text Detection + Recognition
# =============================================================================

def _run_ocr_on_image(image, offset_x=0, offset_y=0, filter_text=True):
    """
    Run EasyOCR on an image — automatically detects and reads ALL text.

    EasyOCR uses CRAFT (detector) + CRNN (recognizer).

    Returns list of dicts:
      [{'bbox': [x1,y1,x2,y2], 'text': str, 'confidence': float}]
    """
    try:
        ocr_frame, scale = _prepare_frame_for_ocr(image)
        results = ocr_engine.readtext(ocr_frame)
        detections = []

        if not results:
            return detections

        for entry in results:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            bbox, text, conf = entry[0], entry[1], entry[2]
            text = str(text).strip()
            conf = float(conf) if conf is not None else 0.0

            if not text:
                continue
            if filter_text and (conf < MIN_CONFIDENCE or not _is_valid_text(text)):
                continue

            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            if scale != 1.0:
                x1 = int(x1 / scale)
                x2 = int(x2 / scale)
                y1 = int(y1 / scale)
                y2 = int(y2 / scale)

            detections.append({
                'bbox': [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                'text': text,
                'confidence': round(float(conf), 3)
            })

        return detections

    except Exception as e:
        print(f"[WARNING] EasyOCR error: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_paddleocr(image, filter_text=True):
    h, w = image.shape[:2]
    if OCR_TILE_MODE and (w >= OCR_TILE_MIN_W or h >= OCR_TILE_MIN_H):
        detections = []
        for tile, ox, oy in _build_tiles(image):
            detections.extend(_run_ocr_on_image(tile, ox, oy, filter_text=filter_text))
        if not detections:
            detections = _run_ocr_on_image(image, filter_text=filter_text)
        return detections

    return _run_ocr_on_image(image, filter_text=filter_text)


def run_paddleocr_tiled(frame, filter_text=True):
    h, w = frame.shape[:2]
    detections = []
    for x1, y1, x2, y2 in _tile_boxes(h, w, PLATE_TILE_SIZE, PLATE_TILE_OVERLAP):
        tile = frame[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        for det in run_paddleocr(tile, filter_text=filter_text):
            bx1, by1, bx2, by2 = det['bbox']
            detections.append({
                'bbox': [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                'text': det['text'],
                'confidence': det['confidence'],
            })
    return _nms_detections(detections, PLATE_NMS_IOU)


def run_yolo_plate_ocr(frame):
    if yolo_model is None:
        return []
    try:
        results = yolo_model.predict(
            frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False,
        )
    except Exception as exc:
        print(f"[WARNING] YOLO predict failed: {exc}")
        return []

    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    order = np.argsort(confs)[::-1]
    detections = []
    vehicle_boxes = []
    frame_h, frame_w = frame.shape[:2]
    count = 0
    for idx in order:
        if count >= YOLO_MAX_VEHICLES:
            break
        cls_id = int(cls_ids[idx])
        if cls_id not in YOLO_VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = xyxy[idx]
        vehicle_boxes.append((x1, y1, x2, y2))
        count += 1

    if not vehicle_boxes:
        return []

    ocr_dets = run_paddleocr_tiled(frame)
    for det in ocr_dets:
        if not _is_plate_text(det['text']):
            continue
        for vbox in vehicle_boxes:
            if _is_bbox_in_vehicle_plate_region(det['bbox'], vbox):
                detections.append(det)
                break

    if detections:
        return _nms_detections(detections, PLATE_NMS_IOU)

    # Relaxed pass: allow any valid text within vehicle plate region
    for det in ocr_dets:
        if not _is_valid_text(det['text']):
            continue
        for vbox in vehicle_boxes:
            if _is_bbox_in_vehicle_plate_region(det['bbox'], vbox):
                detections.append(det)
                break

    if detections:
        return _nms_detections(detections, PLATE_NMS_IOU)

    # Fallback to per-vehicle cropped OCR if needed
    for vbox in vehicle_boxes:
        roi_box = _vehicle_plate_roi(vbox, frame_w, frame_h)
        if roi_box is None:
            continue
        rx1, ry1, rx2, ry2 = roi_box
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        for det in run_paddleocr(roi):
            if not _is_plate_text(det['text']):
                continue
            bx1, by1, bx2, by2 = det['bbox']
            detections.append({
                'bbox': [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1],
                'text': det['text'],
                'confidence': det['confidence'],
            })

    return _nms_detections(detections, PLATE_NMS_IOU)


def run_haar_plate_ocr(frame):
    if plate_cascade is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=PLATE_HAAR_SCALE,
        minNeighbors=PLATE_HAAR_MIN_NEIGHBORS,
        minSize=PLATE_HAAR_MIN_SIZE,
    )
    detections = []
    h, w = frame.shape[:2]
    for (x, y, bw, bh) in plates:
        x1, y1, x2, y2 = _expand_box((x, y, x + bw, y + bh), w, h, 0.10)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_dets = run_paddleocr(crop)
        for det in crop_dets:
            if not _is_plate_text(det['text']):
                continue
            bx1, by1, bx2, by2 = det['bbox']
            detections.append({
                'bbox': [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                'text': det['text'],
                'confidence': det['confidence'],
            })
    return _nms_detections(detections, PLATE_NMS_IOU)


def run_plate_ocr(frame):
    h, w = frame.shape[:2]
    detections = []
    if FORCE_MAX_OCR:
        raw_results = run_paddleocr_tiled(frame, filter_text=False)
        raw_sorted = sorted(raw_results, key=lambda d: d['confidence'], reverse=True)
        if raw_sorted:
            return raw_sorted[:MAX_RETURN_DETECTIONS]
    if YOLO_PLATE_MODE:
        detections.extend(run_yolo_plate_ocr(frame))
    if detections:
        return _nms_detections(detections, PLATE_NMS_IOU)

    if PLATE_HAAR_MODE:
        detections.extend(run_haar_plate_ocr(frame))
    if detections:
        return _nms_detections(detections, PLATE_NMS_IOU)

    candidates = _find_plate_candidates(frame)

    for box in candidates:
        x1, y1, x2, y2 = _expand_box(box, w, h, PLATE_BOX_EXPAND)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_dets = run_paddleocr(crop)
        for det in crop_dets:
            if not _is_plate_text(det['text']):
                continue

            bx1, by1, bx2, by2 = det['bbox']
            detections.append({
                'bbox': [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                'text': det['text'],
                'confidence': det['confidence'],
            })

    if not detections and PLATE_ROI_FALLBACK:
        for x1, y1, x2, y2 in _get_plate_rois(frame):
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_dets = run_paddleocr(roi)
            for det in roi_dets:
                if not _is_plate_text(det['text']):
                    continue
                bx1, by1, bx2, by2 = det['bbox']
                detections.append({
                    'bbox': [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                    'text': det['text'],
                    'confidence': det['confidence'],
                })

    if not detections and PLATE_FALLBACK_FULL:
        fallback_results = run_paddleocr_tiled(frame) if PLATE_TILE_FALLBACK else run_paddleocr(frame)
        for det in fallback_results:
            if _is_plate_text(det['text']):
                detections.append(det)

    if not detections and RAW_TEXT_FALLBACK:
        raw_results = run_paddleocr_tiled(frame, filter_text=False) if PLATE_TILE_FALLBACK else run_paddleocr(frame, filter_text=False)
        raw_sorted = sorted(raw_results, key=lambda d: d['confidence'], reverse=True)
        return raw_sorted[:MAX_RETURN_DETECTIONS]

    return _nms_detections(detections, PLATE_NMS_IOU)


def run_ocr_on_region(image):
    """Run EasyOCR on a cropped region (manual selection mode)."""
    resized = _resize_for_manual_ocr(image)
    enhanced = _enhance_for_ocr(resized)
    detections = _read_easyocr_texts(enhanced)
    if not detections:
        detections = _read_easyocr_texts(resized)
    if len(detections) < MIN_MANUAL_LINES:
        fallback_dets = _read_easyocr_texts_fallback(enhanced)
        if not fallback_dets:
            fallback_dets = _read_easyocr_texts_fallback(resized)
        if fallback_dets:
            detections = detections + fallback_dets

    if detections:
        merged = {}
        for det in detections:
            text = det['text']
            conf = det['confidence']
            if text not in merged or conf > merged[text]:
                merged[text] = conf
        detections = [{'text': text, 'confidence': conf} for text, conf in merged.items()]
        strong = [d for d in detections if d['confidence'] >= MANUAL_CONFIDENCE_MIN and _is_valid_text(d['text'])]
        others = [d for d in detections if _is_valid_text(d['text'])]

        if not strong:
            strong = others

        if strong:
            strong = sorted(strong, key=lambda d: d['confidence'], reverse=True)
            texts = [d['text'] for d in strong[:MANUAL_MAX_RESULTS]]
            if len(texts) < MANUAL_MAX_RESULTS:
                seen = set(texts)
                extras = [d for d in others if d['text'] not in seen]
                extras = sorted(extras, key=lambda d: d['confidence'], reverse=True)
                for det in extras:
                    if len(texts) >= MANUAL_MAX_RESULTS:
                        break
                    texts.append(det['text'])
            return " ".join(texts)
    return "No text detected"


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload."""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file"}), 400

        file = request.files['video']
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        filename_str = str(file.filename)
        ext = filename_str.rsplit('.', 1)[1].lower() if '.' in filename_str else 'mp4'
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f".{ext} not supported"}), 400

        safe_name = f"video_{uuid.uuid4().hex[:8]}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        file.save(filepath)

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({"error": "File save failed"}), 500

        print(f"[OK] Uploaded: {file.filename} -> {safe_name}")
        return jsonify({"status": "uploaded", "filename": safe_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploads/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/plate_cache/<filename>')
def serve_plate_cache(filename):
    return send_from_directory(PLATE_CACHE_DIR, filename)


@app.route('/ocr_region', methods=['POST'])
def ocr_region():
    """Manual selection: crop region, run OCR, return zoomed + text."""
    if 'frame' not in request.files:
        return jsonify({"error": "No frame data"}), 400

    file = request.files['frame']
    coords = request.form

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        x1 = int(float(coords.get('x1', '0')))
        y1 = int(float(coords.get('y1', '0')))
        x2 = int(float(coords.get('x2', '0')))
        y2 = int(float(coords.get('y2', '0')))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid coordinates"}), 400

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return jsonify({"error": "Invalid region"}), 400

    cropped = frame[y1:y2, x1:x2]
    text = run_ocr_on_region(cropped)

    # Create zoomed version
    crop_h, crop_w = cropped.shape[:2]
    zoom_w = 500
    zoom_scale = zoom_w / crop_w if crop_w > 0 else 1
    zoom_h = max(int(crop_h * zoom_scale), 50)
    zoomed = cv2.resize(cropped, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

    _, buffer = cv2.imencode('.jpg', zoomed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    zoomed_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "text": text if text else "No text detected",
        "zoomed_image": f"data:image/jpeg;base64,{zoomed_b64}"
    })


# =============================================================================
# AUTO-DETECT ENDPOINT — Automatic Text Detection from Running Video
# =============================================================================
# This is the KEY feature: the browser sends a frame every ~1 second,
# EasyOCR finds ALL text automatically, draws boxes, shows results in sidebar,
# and auto-zooms the best detection.
# =============================================================================

@app.route('/ocr_auto_detect', methods=['POST'])
def ocr_auto_detect():
    """
    Automatically detect and recognize ALL text in a video frame.
    No manual rectangle needed — EasyOCR does it all.
    """
    if 'frame' not in request.files:
        return jsonify({"error": "No frame data"}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    global best_frame, best_frame_score, best_frame_time
    global last_ocr_detections, last_ocr_zoomed_image, last_ocr_zoomed_text, last_ocr_time

    now = time.time()
    blur_score, brightness = _frame_quality_metrics(frame)
    quality_ok = _quality_ok(blur_score, brightness)
    _update_best_frame(frame, blur_score, brightness)

    time_since_last = (now - last_ocr_time) * 1000
    time_since_best = (now - best_frame_time) * 1000 if best_frame is not None else None

    use_best = best_frame is not None and time_since_best is not None and time_since_best >= BEST_FRAME_WINDOW_MS
    should_run = False
    ocr_frame = frame

    if use_best:
        should_run = True
        ocr_frame = best_frame
    elif not last_ocr_detections:
        should_run = True
    elif quality_ok and time_since_last >= OCR_MIN_INTERVAL_MS:
        should_run = True
    elif time_since_last >= OCR_FORCE_EVERY_MS:
        should_run = True

    skipped = False
    zoomed_image = None
    zoomed_text = None

    if should_run:
        start_time = time.time()
        if PLATE_ONLY_MODE:
            detections = run_plate_ocr(ocr_frame)
        else:
            detections = run_paddleocr(ocr_frame)
        process_time = (time.time() - start_time) * 1000  # ms

        if PLATE_CACHE_ENABLED:
            _update_plate_cache(ocr_frame, detections)

        best_det = None
        for det in detections:
            if best_det is None or det['confidence'] > best_det['confidence']:
                best_det = det

        if best_det:
            bx1, by1, bx2, by2 = best_det['bbox']
            h, w = ocr_frame.shape[:2]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)

            if bx2 > bx1 and by2 > by1:
                cropped = ocr_frame[by1:by2, bx1:bx2]
                crop_h, crop_w = cropped.shape[:2]
                zoom_w = 400
                zoom_scale = zoom_w / crop_w if crop_w > 0 else 1
                zoom_h = max(int(crop_h * zoom_scale), 50)
                if zoom_w > 0 and zoom_h > 0:
                    zoomed = cv2.resize(cropped, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
                    _, buffer = cv2.imencode('.jpg', zoomed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    zoomed_b64 = base64.b64encode(buffer).decode('utf-8')
                    zoomed_image = f"data:image/jpeg;base64,{zoomed_b64}"
                    zoomed_text = best_det['text']

        last_ocr_detections = detections
        last_ocr_zoomed_image = zoomed_image
        last_ocr_zoomed_text = zoomed_text
        last_ocr_time = now
        best_frame = None
        best_frame_score = 0.0
        best_frame_time = 0.0
    else:
        skipped = True
        detections = list(last_ocr_detections)
        zoomed_image = last_ocr_zoomed_image
        zoomed_text = last_ocr_zoomed_text
        process_time = 0.0

    return jsonify({
        "detections": detections,
        "process_time_ms": round(process_time, 1),
        "zoomed_image": zoomed_image,
        "zoomed_text": zoomed_text,
        "plate_cache": _serialize_plate_cache() if PLATE_CACHE_ENABLED else [],
        "quality": {
            "blur": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "quality_ok": quality_ok,
            "used_best": use_best,
            "skipped": skipped,
        }
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 500MB."}), 413


# =============================================================================
# Run the Flask Server
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  Real-Time OCR System for Video-Based Text Recognition")
    print("  in Traffic and Surveillance")
    print("")
    print("  Engine: EasyOCR (CRAFT detection + CRNN recognition)")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
