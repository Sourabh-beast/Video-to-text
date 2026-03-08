# =============================================================================
# Flask Web Application
# Real-Time OCR System for Video-Based Text Recognition
# in Traffic and Surveillance
# =============================================================================

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import pytesseract
import numpy as np
import os
import sys
import base64
import uuid

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# ─── Tesseract Setup ─────────────────────────────────────────────────────────
def setup_tesseract():
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

setup_tesseract()

# ─── OCR Processing (Multi-Strategy for Best Accuracy) ───────────────────────
import re
import string

def get_preprocessed_versions(image):
    """
    Generate multiple preprocessed versions of the image.
    Different preprocessing works better for different text types:
    - License plates need high contrast
    - Road signs need color-to-gray clarity
    - CCTV text needs denoising
    """
    h, w = image.shape[:2]

    # Scale up small images for better OCR (Tesseract works best at ~300 DPI)
    scale = 1
    if w < 200 or h < 100:
        scale = 3
    elif w < 400 or h < 200:
        scale = 2

    if scale > 1:
        image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    versions = []

    # 1. Simple grayscale (good for clean printed text)
    versions.append(gray)

    # 2. Otsu thresholding (good for high-contrast text like license plates)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(otsu)

    # 3. Adaptive threshold (good for uneven lighting like outdoor signs)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    versions.append(adaptive)

    # 4. Inverted Otsu (for white text on dark background — common in CCTV)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    versions.append(otsu_inv)

    # 5. Denoised + sharpened (for noisy surveillance footage)
    denoised = cv2.fastNlMeansDenoising(gray, h=12)
    sharp_kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, sharp_kernel)
    _, sharp_bin = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(sharp_bin)

    return versions


def score_text(text):
    """
    Score OCR result for English readability.
    Higher score = more likely to be real English text.
    """
    if not text or not text.strip():
        return -1

    clean = text.strip()
    total_chars = len(clean)
    if total_chars == 0:
        return -1

    # Count alphanumeric characters (English letters + digits)
    alnum = sum(1 for c in clean if c.isalnum())

    # Count actual English letters
    letters = sum(1 for c in clean if c.isalpha())

    # Count common punctuation and spaces
    normal = sum(1 for c in clean if c in string.ascii_letters + string.digits + ' .,;:!?-/()\n')

    # Ratio of "normal readable" characters
    normal_ratio = normal / total_chars

    # Penalize very short results (likely noise)
    length_bonus = min(total_chars / 3, 10)

    # Penalize results that are mostly special characters
    if normal_ratio < 0.4:
        return -1

    score = (alnum * 2) + (letters * 1) + (length_bonus * 3) + (normal_ratio * 20)
    return score


def run_ocr(image):
    """
    Run OCR with multiple preprocessing strategies and PSM modes.
    Return the best (most readable English) result.
    """
    try:
        versions = get_preprocessed_versions(image)
        best_text = ""
        best_score = -1

        # Try different PSM modes:
        # 3 = Fully automatic page segmentation
        # 6 = Assume a single uniform block of text
        # 7 = Treat the image as a single text line
        # 8 = Treat the image as a single word
        psm_modes = [6, 3, 7, 8]

        for preprocessed in versions:
            for psm in psm_modes:
                try:
                    config = f"--oem 3 --psm {psm} -l eng"
                    raw = pytesseract.image_to_string(preprocessed, config=config)
                    lines = [line.strip() for line in raw.splitlines() if line.strip()]
                    text = "\n".join(lines)

                    sc = score_text(text)
                    if sc > best_score:
                        best_score = sc
                        best_text = text
                except Exception:
                    continue

        if best_text:
            return best_text
        return "No readable text found"

    except Exception as e:
        return f"OCR Error: {str(e)}"

# ─── Flask Routes ────────────────────────────────────────────────────────────
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
            return jsonify({"error": "No video file in request"}), 400

        file = request.files['video']
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        # Get extension
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'mp4'
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"File type .{ext} not supported"}), 400

        # Save with safe UUID name
        safe_name = f"video_{uuid.uuid4().hex[:8]}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        file.save(filepath)

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({"error": "File save failed"}), 500

        print(f"[OK] Uploaded: {file.filename} -> {safe_name}")
        return jsonify({"status": "uploaded", "filename": safe_name})

    except Exception as e:
        print(f"[ERROR] Upload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def serve_video(filename):
    """Serve uploaded video files directly for HTML5 video element."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/ocr_region', methods=['POST'])
def ocr_region():
    """Receive a frame image + crop coordinates, run OCR, return results."""
    if 'frame' not in request.files:
        return jsonify({"error": "No frame data"}), 400

    file = request.files['frame']
    coords = request.form

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        x1 = int(float(coords.get('x1', 0)))
        y1 = int(float(coords.get('y1', 0)))
        x2 = int(float(coords.get('x2', 0)))
        y2 = int(float(coords.get('y2', 0)))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid coordinates"}), 400

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return jsonify({"error": "Invalid region"}), 400

    # Crop and run OCR
    cropped = frame[y1:y2, x1:x2]
    text = run_ocr(cropped)

    # Create zoomed version
    crop_h, crop_w = cropped.shape[:2]
    zoom_w = 500
    zoom_scale = zoom_w / crop_w if crop_w > 0 else 1
    zoom_h = max(int(crop_h * zoom_scale), 50)
    zoomed = cv2.resize(cropped, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

    _, buffer = cv2.imencode('.jpg', zoomed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    zoomed_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "text": text if text else "No text detected in this region",
        "zoomed_image": f"data:image/jpeg;base64,{zoomed_b64}"
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 500MB."}), 413


if __name__ == '__main__':
    print("=" * 70)
    print("  Real-Time OCR System for Video-Based Text Recognition")
    print("  in Traffic and Surveillance")
    print("")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
