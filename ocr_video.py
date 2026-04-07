"""
Real-Time OCR System for Video-Based Text Recognition
Standalone OpenCV app using PaddleOCR.

Controls:
  - Press 'a' to toggle auto-detect mode
  - Click and drag to select a region (manual mode)
  - Press 'c' to clear manual selection
  - Press 'q' to quit
"""

import os
os.environ.setdefault("FLAGS_use_pir", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2  # type: ignore
import numpy as np  # type: ignore
import re
import sys
import time
import hashlib

from paddleocr import PaddleOCR  # type: ignore
try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    YOLO = None
    YOLO_IMPORT_ERROR = str(exc)
else:
    YOLO_IMPORT_ERROR = None

MIN_CONFIDENCE = 0.12
MIN_TEXT_LENGTH = 1
FRAME_SKIP = 1
AUTO_DETECT_DEFAULT = True
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
DISPLAY_W = 960

PLATE_CACHE_ENABLED = True
PLATE_CACHE_DIR = "plate_cache"
PLATE_CACHE_MAX = 12
PLATE_CACHE_PANEL_W = 360
PLATE_CACHE_THUMB_H = 110
PLATE_CACHE_PADDING = 10
PLATE_CACHE_TEXT_SCALE = 0.5
PLATE_CACHE_TEXT_THICKNESS = 1
PLATE_CACHE_BG = (18, 18, 18)
PLATE_CACHE_TEXT_COLOR = (240, 240, 240)
PLATE_CACHE_SAVE = True

PLATE_ONLY_MODE = True
YOLO_PLATE_MODE = True
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
YOLO_MAX_VEHICLES = 10
YOLO_VEHICLE_CLASSES = {2, 3, 5, 7}
YOLO_PLATE_REGION = (0.55, 1.0)
YOLO_PLATE_SIDE_TRIM = 0.08
YOLO_PLATE_EXPAND = 0.06
PLATE_HAAR_MODE = True
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

USE_WEBCAM = False
VIDEO_FILE = "sample_video.mp4"


def _is_valid_text(text: str) -> bool:
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False
    if NOISE_PATTERN.match(text):
        return False
    if not re.search(r'[a-zA-Z0-9]', text):
        return False
    return True


def _is_plate_text(text: str) -> bool:
    text = text.strip()
    if len(text) < 3:
        return False
    if not re.search(r'[0-9]', text):
        return False
    if not PLATE_TEXT_PATTERN.match(text):
        return False
    return True


def open_video_source():
    if USE_WEBCAM:
        print("[INFO] Opening webcam (camera index 0)...")
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.exists(VIDEO_FILE):
            print(f"[ERROR] Video file not found: {VIDEO_FILE}")
            print("[ERROR] Place your video file in the project folder, or change VIDEO_FILE variable.")
            sys.exit(1)
        print(f"[INFO] Opening video file: {VIDEO_FILE}")
        cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print("[ERROR] Could not open video source. Check your camera or file path.")
        sys.exit(1)

    return cap


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


def _get_confidence_color(confidence: float) -> tuple:
    if confidence >= 0.70:
        return (0, 255, 0)
    if confidence >= 0.40:
        return (0, 255, 255)
    return (0, 100, 255)


def _ensure_cache_dir():
    if PLATE_CACHE_SAVE:
        os.makedirs(PLATE_CACHE_DIR, exist_ok=True)


def _hash_image(image) -> str:
    small = cv2.resize(image, (32, 12), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


def _fit_image(image, max_w, max_h):
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _push_plate_cache(crop, text, confidence):
    if crop is None or crop.size == 0:
        return
    key = _hash_image(crop)
    for item in plate_cache:
        if item["hash"] == key:
            item["image"] = crop
            if text:
                item["text"] = text
            item["confidence"] = max(item["confidence"], confidence)
            item["last_seen"] = time.time()
            return

    path = None
    if PLATE_CACHE_SAVE:
        _ensure_cache_dir()
        filename = f"plate_{int(time.time() * 1000)}_{len(plate_cache):02d}.jpg"
        path = os.path.join(PLATE_CACHE_DIR, filename)
        cv2.imwrite(path, crop)

    plate_cache.insert(0, {
        "hash": key,
        "image": crop,
        "text": text,
        "confidence": confidence,
        "last_seen": time.time(),
        "path": path,
    })

    if len(plate_cache) > PLATE_CACHE_MAX:
        plate_cache.pop()


def _update_plate_cache(frame, detections, ocr_engine):
    if not detections:
        return
    plate_like = [d for d in detections if _is_plate_text(d["text"])]
    source = plate_like if plate_like else detections
    frame_h, frame_w = frame.shape[:2]

    for det in source:
        x1, y1, x2, y2 = det["bbox"]
        x1, y1, x2, y2 = _expand_box((x1, y1, x2, y2), frame_w, frame_h, 0.05)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text = det["text"]
        conf = det["confidence"]
        crop_dets = run_paddleocr(crop, ocr_engine, filter_text=False)
        if crop_dets:
            best = max(crop_dets, key=lambda d: d["confidence"])
            if best["text"]:
                text = best["text"]
                conf = best["confidence"]

        _push_plate_cache(crop, text, conf)
        if len(plate_cache) >= PLATE_CACHE_MAX:
            break


def _render_cache_panel(panel_h):
    panel = np.full((panel_h, PLATE_CACHE_PANEL_W, 3), PLATE_CACHE_BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = PLATE_CACHE_PADDING + 16

    cv2.putText(
        panel,
        "Plate Cache",
        (PLATE_CACHE_PADDING, y),
        font,
        0.6,
        PLATE_CACHE_TEXT_COLOR,
        1,
    )
    y += 14 + PLATE_CACHE_PADDING

    if not plate_cache:
        cv2.putText(
            panel,
            "No cached plates yet",
            (PLATE_CACHE_PADDING, y + 18),
            font,
            0.5,
            (180, 180, 180),
            1,
        )
        return panel

    for item in plate_cache:
        if y + PLATE_CACHE_THUMB_H + 30 > panel_h:
            break
        thumb = _fit_image(
            item["image"],
            PLATE_CACHE_PANEL_W - 2 * PLATE_CACHE_PADDING,
            PLATE_CACHE_THUMB_H,
        )
        if thumb is None:
            continue
        th, tw = thumb.shape[:2]
        panel[y:y + th, PLATE_CACHE_PADDING:PLATE_CACHE_PADDING + tw] = thumb

        label = item["text"] or "?"
        conf = int(item["confidence"] * 100)
        text_line = f"{label} ({conf}%)"
        cv2.putText(
            panel,
            text_line,
            (PLATE_CACHE_PADDING, y + th + 18),
            font,
            PLATE_CACHE_TEXT_SCALE,
            PLATE_CACHE_TEXT_COLOR,
            PLATE_CACHE_TEXT_THICKNESS,
        )
        y += th + 28

    return panel


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


def _run_ocr_on_image(image, ocr_engine, offset_x=0, offset_y=0, filter_text=True):
    try:
        ocr_frame, scale = _prepare_frame_for_ocr(image)
        results = ocr_engine.predict(
            ocr_frame,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=OCR_DET_LIMIT_SIDE,
            text_det_limit_type="min",
            text_det_thresh=OCR_DET_THRESH,
            text_det_box_thresh=OCR_DET_BOX_THRESH,
            text_det_unclip_ratio=OCR_DET_UNCLIP,
            text_rec_score_thresh=OCR_REC_SCORE_THRESH,
        )
    except Exception as exc:
        print(f"[WARNING] PaddleOCR error: {exc}")
        return []

    detections = []
    if not results:
        return detections

    for res in results:
        if not isinstance(res, dict):
            continue

        polys = res.get("rec_polys") or []
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []

        for poly, text_item, conf in zip(polys, texts, scores):
            text = text_item[0] if isinstance(text_item, (list, tuple)) else text_item
            text = str(text).strip()
            conf = float(conf)

            if not text:
                continue
            if filter_text and (conf < MIN_CONFIDENCE or not _is_valid_text(text)):
                continue

            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            if scale != 1.0:
                x1 = int(x1 / scale)
                x2 = int(x2 / scale)
                y1 = int(y1 / scale)
                y2 = int(y2 / scale)

            detections.append({
                "bbox": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                "text": text,
                "confidence": conf,
            })

    return detections


def run_paddleocr(image, ocr_engine, filter_text=True):
    h, w = image.shape[:2]
    if OCR_TILE_MODE and (w >= OCR_TILE_MIN_W or h >= OCR_TILE_MIN_H):
        detections = []
        for tile, ox, oy in _build_tiles(image):
            detections.extend(_run_ocr_on_image(tile, ocr_engine, ox, oy, filter_text=filter_text))
        if not detections:
            detections = _run_ocr_on_image(image, ocr_engine, filter_text=filter_text)
        return detections

    return _run_ocr_on_image(image, ocr_engine, filter_text=filter_text)


def run_paddleocr_tiled(frame, ocr_engine, filter_text=True):
    h, w = frame.shape[:2]
    detections = []
    for x1, y1, x2, y2 in _tile_boxes(h, w, PLATE_TILE_SIZE, PLATE_TILE_OVERLAP):
        tile = frame[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        for det in run_paddleocr(tile, ocr_engine, filter_text=filter_text):
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
            roi_dets = run_paddleocr(roi, ocr_engine, filter_text=filter_text)
            for det in roi_dets:
                if filter_text and not _is_plate_text(det["text"]):
                    continue
                bx1, by1, bx2, by2 = det["bbox"]
                detections.append({
                    "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                    "text": det["text"],
                    "confidence": det["confidence"],
                })
    return _nms_detections(detections, PLATE_NMS_IOU)


def run_yolo_plate_ocr(frame, ocr_engine):
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

    ocr_dets = run_paddleocr_tiled(frame, ocr_engine)
    for det in ocr_dets:
        if not _is_plate_text(det["text"]):
            continue
        for vbox in vehicle_boxes:
            if _is_bbox_in_vehicle_plate_region(det["bbox"], vbox):
                detections.append(det)
                break

    if detections:
        return _nms_detections(detections, PLATE_NMS_IOU)

    for det in ocr_dets:
        if not _is_valid_text(det["text"]):
            continue
        for vbox in vehicle_boxes:
            if _is_bbox_in_vehicle_plate_region(det["bbox"], vbox):
                detections.append(det)
                break

    if detections:
        return _nms_detections(detections, PLATE_NMS_IOU)

    for vbox in vehicle_boxes:
        roi_box = _vehicle_plate_roi(vbox, frame_w, frame_h)
        if roi_box is None:
            continue
        rx1, ry1, rx2, ry2 = roi_box
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        for det in run_paddleocr(roi, ocr_engine):
            if not _is_plate_text(det["text"]):
                continue
            bx1, by1, bx2, by2 = det["bbox"]
            detections.append({
                "bbox": [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1],
                "text": det["text"],
                "confidence": det["confidence"],
            })

    return _nms_detections(detections, PLATE_NMS_IOU)


def run_haar_plate_ocr(frame, ocr_engine):
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
        crop_dets = run_paddleocr(crop, ocr_engine)
        for det in crop_dets:
            if not _is_plate_text(det["text"]):
                continue
            bx1, by1, bx2, by2 = det["bbox"]
            detections.append({
                "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                "text": det["text"],
                "confidence": det["confidence"],
            })
    return _nms_detections(detections, PLATE_NMS_IOU)


def run_plate_ocr(frame, ocr_engine):
    h, w = frame.shape[:2]
    detections = []
    if FORCE_MAX_OCR:
        raw_results = run_paddleocr_tiled(frame, ocr_engine, filter_text=False) if PLATE_TILE_FALLBACK else run_paddleocr(frame, ocr_engine, filter_text=False)
        raw_sorted = sorted(raw_results, key=lambda d: d['confidence'], reverse=True)
        if raw_sorted:
            return raw_sorted[:MAX_RETURN_DETECTIONS]
    if YOLO_PLATE_MODE:
        detections.extend(run_yolo_plate_ocr(frame, ocr_engine))
    if detections:
        return _nms_detections(detections, PLATE_NMS_IOU)

    if PLATE_HAAR_MODE:
        detections.extend(run_haar_plate_ocr(frame, ocr_engine))
    if detections:
        return _nms_detections(detections, PLATE_NMS_IOU)

    candidates = _find_plate_candidates(frame)

    for box in candidates:
        x1, y1, x2, y2 = _expand_box(box, w, h, PLATE_BOX_EXPAND)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_dets = run_paddleocr(crop, ocr_engine)
        for det in crop_dets:
            if not _is_plate_text(det["text"]):
                continue
            bx1, by1, bx2, by2 = det["bbox"]
            detections.append({
                "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                "text": det["text"],
                "confidence": det["confidence"],
            })

    if not detections and PLATE_FALLBACK_FULL:
        fallback_results = run_paddleocr_tiled(frame, ocr_engine) if PLATE_TILE_FALLBACK else run_paddleocr(frame, ocr_engine)
        for det in fallback_results:
            if _is_plate_text(det["text"]):
                detections.append(det)

    if not detections and RAW_TEXT_FALLBACK:
        raw_results = run_paddleocr_tiled(frame, ocr_engine, filter_text=False) if PLATE_TILE_FALLBACK else run_paddleocr(frame, ocr_engine, filter_text=False)
        raw_sorted = sorted(raw_results, key=lambda d: d['confidence'], reverse=True)
        return raw_sorted[:MAX_RETURN_DETECTIONS]

    return _nms_detections(detections, PLATE_NMS_IOU)


mouse = {
    "drawing": False,
    "start_x": 0,
    "start_y": 0,
    "end_x": 0,
    "end_y": 0,
    "rect_ready": False,
}

plate_cache = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["drawing"] = True
        mouse["start_x"] = x
        mouse["start_y"] = y
        mouse["end_x"] = x
        mouse["end_y"] = y
        mouse["rect_ready"] = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
        mouse["end_x"] = x
        mouse["end_y"] = y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse["drawing"] = False
        mouse["end_x"] = x
        mouse["end_y"] = y

        dx = abs(mouse["end_x"] - mouse["start_x"])
        dy = abs(mouse["end_y"] - mouse["start_y"])
        if dx > 10 and dy > 10:
            mouse["rect_ready"] = True


def main():
    print("=" * 60)
    print("  Loading PaddleOCR (PP-OCR)...")
    print("  First run downloads models - one-time ~100MB")
    print("=" * 60)

    ocr_engine = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        device='cpu',
        enable_mkldnn=False,
        enable_cinn=False,
    )

    cap = open_video_source()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Video FPS: {fps:.0f}")

    cv2.namedWindow("Real-Time OCR Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "Real-Time OCR Video",
        DISPLAY_W + (PLATE_CACHE_PANEL_W if PLATE_CACHE_ENABLED else 0),
        540,
    )
    cv2.setMouseCallback("Real-Time OCR Video", mouse_callback)

    print("=" * 60)
    print("  REAL-TIME OCR SYSTEM - RUNNING")
    print("  Engine: PaddleOCR (PP-OCR detection + recognition)")
    print("")
    print("  Press 'a' to toggle auto-detect mode")
    print("  Draw a rectangle with your mouse for manual OCR")
    print("  Press 'c' to clear, 'q' to quit")
    print("=" * 60)

    auto_detect_mode = AUTO_DETECT_DEFAULT
    auto_detections = []
    last_ocr_text = ""
    last_rect = []
    frame_count = 0
    fps_display = 0.0
    fps_timer = time.time()
    fps_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if not USE_WEBCAM:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[INFO] Camera disconnected.")
            break

        frame_count += 1

        display_w = DISPLAY_W
        h, w = frame.shape[:2]
        scale = display_w / w
        display_h = int(h * scale)
        display = cv2.resize(frame, (display_w, display_h))

        fps_frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_timer = time.time()

        if auto_detect_mode and frame_count % FRAME_SKIP == 0:
            if PLATE_ONLY_MODE:
                auto_detections = run_plate_ocr(display, ocr_engine)
            else:
                auto_detections = run_paddleocr(display, ocr_engine)

        if auto_detect_mode:
            for det in auto_detections:
                x1, y1, x2, y2 = det["bbox"]
                text = det["text"]
                conf = det["confidence"]
                color = _get_confidence_color(conf)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{text} ({int(conf * 100)}%)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                tw, th = cv2.getTextSize(label, font, font_scale, thickness)[0]
                label_y = y1 - 8
                if label_y < 20:
                    label_y = y2 + th + 8

                cv2.rectangle(
                    display,
                    (x1, label_y - th - 4),
                    (x1 + tw + 8, label_y + 4),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(display, label, (x1 + 4, label_y - 2), font, font_scale, color, thickness)

        if mouse["drawing"] or mouse["rect_ready"]:
            x1, y1 = mouse["start_x"], mouse["start_y"]
            x2, y2 = mouse["end_x"], mouse["end_y"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if mouse["rect_ready"]:
            x1, y1 = mouse["start_x"], mouse["start_y"]
            x2, y2 = mouse["end_x"], mouse["end_y"]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            if x2 > x1 and y2 > y1:
                crop = display[y1:y2, x1:x2]
                detections = run_paddleocr(crop, ocr_engine)
                last_ocr_text = " ".join([d["text"] for d in detections]) or "No text detected"
                last_rect = [x1, y1, x2, y2]

                zoom_w = 520
                crop_h, crop_w = crop.shape[:2]
                zoom_scale = zoom_w / crop_w if crop_w > 0 else 1
                zoom_h = max(int(crop_h * zoom_scale), 50)
                zoomed = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Zoomed Region", zoomed)

            mouse["rect_ready"] = False

        if last_ocr_text and last_rect:
            cv2.putText(
                display,
                f"Manual: {last_ocr_text}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            display,
            f"FPS: {fps_display:.1f}",
            (10, display_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        status_text = "AUTO: ON (press A)" if auto_detect_mode else "AUTO: OFF (press A)"
        cv2.putText(
            display,
            status_text,
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        if PLATE_CACHE_ENABLED and auto_detect_mode and auto_detections:
            _update_plate_cache(display, auto_detections, ocr_engine)

        if PLATE_CACHE_ENABLED:
            panel = _render_cache_panel(display_h)
            display = np.concatenate([display, panel], axis=1)
            cv2.resizeWindow("Real-Time OCR Video", display.shape[1], display.shape[0])

        cv2.imshow("Real-Time OCR Video", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            auto_detect_mode = not auto_detect_mode
            auto_detections = []
            print(f"[INFO] Auto-detect {'ON' if auto_detect_mode else 'OFF'}")
        if key == ord('c'):
            last_ocr_text = ""
            last_rect = []
            if cv2.getWindowProperty("Zoomed Region", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Zoomed Region")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
