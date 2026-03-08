# =============================================================================
# Real-Time OCR System for Video-Based Text Recognition
# in Traffic and Surveillance
# =============================================================================
# 
# HOW TO USE:
#   1. Run the script: python ocr_video.py
#   2. Video will play in "Real-Time OCR Video" window
#   3. Click and DRAG your mouse to draw a rectangle on any text area
#   4. The selected region will be ZOOMED in "Zoomed Text Region" window
#   5. OCR text will appear in GREEN on the zoomed view
#   6. Press 'c' to clear selection, press 'q' to quit
# =============================================================================

import cv2
import pytesseract
import numpy as np
import os
import sys

# ─── Tesseract Setup ─────────────────────────────────────────────────────────
def setup_tesseract():
    """Find and configure Tesseract OCR executable."""
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"[INFO] Tesseract found at: {path}")
            return True
    # Check system PATH
    try:
        pytesseract.get_tesseract_version()
        print("[INFO] Tesseract found on system PATH.")
        return True
    except Exception:
        pass
    print("[ERROR] Tesseract not found! Install from: https://github.com/tesseract-ocr/tesseract")
    return False


# ─── Video Input ──────────────────────────────────────────────────────────────
USE_WEBCAM = False
VIDEO_FILE = "sample_video.mp4"

def open_video_source():
    """Open webcam or video file based on USE_WEBCAM flag."""
    if USE_WEBCAM:
        print("[INFO] Opening webcam...")
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.exists(VIDEO_FILE):
            print(f"[ERROR] Video file not found: {VIDEO_FILE}")
            sys.exit(1)
        print(f"[INFO] Opening video: {VIDEO_FILE}")
        cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        sys.exit(1)
    return cap


# ─── Preprocessing for OCR ───────────────────────────────────────────────────
def preprocess_for_ocr(image):
    """
    Preprocess a cropped region for better OCR accuracy.
    Steps: Resize(2x) → Grayscale → Gaussian Blur → Adaptive Threshold
    """
    h, w = image.shape[:2]
    # Step 1: Resize 2x — makes small text larger for Tesseract
    resized = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    # Step 2: Grayscale — OCR doesn't need colour info
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Step 3: Gaussian Blur — reduces noise/grain
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Step 4: Adaptive Threshold — crisp black/white text
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return binary


# ─── Multi-Strategy OCR for Best Accuracy ────────────────────────────────────
import string

def get_preprocessed_versions(image):
    """Generate multiple preprocessed versions for different text types."""
    h, w = image.shape[:2]
    scale = 1
    if w < 200 or h < 100:
        scale = 3
    elif w < 400 or h < 200:
        scale = 2
    if scale > 1:
        image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    versions = []

    # 1. Grayscale
    versions.append(gray)
    # 2. Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(otsu)
    # 3. Adaptive threshold
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    versions.append(adaptive)
    # 4. Inverted (for white-on-dark text)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    versions.append(otsu_inv)
    # 5. Denoised + sharpened
    denoised = cv2.fastNlMeansDenoising(gray, h=12)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    _, sharp_bin = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(sharp_bin)
    return versions

def score_text(text):
    """Score OCR result for English readability."""
    if not text or not text.strip():
        return -1
    clean = text.strip()
    total = len(clean)
    if total == 0:
        return -1
    alnum = sum(1 for c in clean if c.isalnum())
    letters = sum(1 for c in clean if c.isalpha())
    normal = sum(1 for c in clean if c in string.ascii_letters + string.digits + ' .,;:!?-/()\n')
    ratio = normal / total
    if ratio < 0.4:
        return -1
    return (alnum * 2) + (letters * 1) + (min(total / 3, 10) * 3) + (ratio * 20)

def run_ocr(image):
    """Run OCR with multiple strategies, return best readable English result."""
    try:
        versions = get_preprocessed_versions(image)
        best_text = ""
        best_score = -1
        for prep in versions:
            for psm in [6, 3, 7, 8]:
                try:
                    config = f"--oem 3 --psm {psm} -l eng"
                    raw = pytesseract.image_to_string(prep, config=config)
                    lines = [line.strip() for line in raw.splitlines() if line.strip()]
                    text = "\n".join(lines)
                    sc = score_text(text)
                    if sc > best_score:
                        best_score = sc
                        best_text = text
                except Exception:
                    continue
        return best_text if best_text else "No readable text found"
    except Exception as e:
        print(f"[WARNING] OCR error: {e}")
        return ""


# ─── Mouse Drawing State ─────────────────────────────────────────────────────
# This dictionary tracks the mouse state for drawing rectangles
mouse = {
    "drawing": False,    # True while mouse button is held down
    "start_x": 0,       # Starting X of rectangle
    "start_y": 0,       # Starting Y of rectangle
    "end_x": 0,         # Current/ending X
    "end_y": 0,         # Current/ending Y
    "rect_ready": False, # True when a rectangle has been completed
}


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for drawing rectangles on the video frame.
    - Left button DOWN  → start drawing
    - Mouse MOVE        → update rectangle end point
    - Left button UP    → finish drawing, mark rectangle as ready
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a new rectangle
        mouse["drawing"] = True
        mouse["start_x"] = x
        mouse["start_y"] = y
        mouse["end_x"] = x
        mouse["end_y"] = y
        mouse["rect_ready"] = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse["drawing"]:
            # Update the end point as mouse moves
            mouse["end_x"] = x
            mouse["end_y"] = y

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing
        mouse["drawing"] = False
        mouse["end_x"] = x
        mouse["end_y"] = y
        # Only mark ready if the rectangle has some size
        dx = abs(mouse["end_x"] - mouse["start_x"])
        dy = abs(mouse["end_y"] - mouse["start_y"])
        if dx > 10 and dy > 10:
            mouse["rect_ready"] = True


# ─── Main Loop ───────────────────────────────────────────────────────────────
def main():
    if not setup_tesseract():
        sys.exit(1)

    cap = open_video_source()

    # Create the main video window and attach mouse callback
    cv2.namedWindow("Real-Time OCR Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time OCR Video", 960, 540)
    cv2.setMouseCallback("Real-Time OCR Video", mouse_callback)

    print("[INFO] Video playing. Draw a rectangle with your mouse to OCR a region.")
    print("[INFO] Press 'c' to clear selection, 'q' to quit.")

    # Store the last OCR result so it persists until cleared
    last_ocr_text = ""
    last_rect = None  # (x1, y1, x2, y2) of the completed rectangle

    while True:
        ret, frame = cap.read()
        if not ret:
            # If video file ended, loop back to start
            if not USE_WEBCAM:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[INFO] Camera disconnected.")
                break

        # Resize frame for consistent display
        display_w = 960
        h, w = frame.shape[:2]
        scale = display_w / w
        display_h = int(h * scale)
        display = cv2.resize(frame, (display_w, display_h))

        # ── Draw the rectangle the user is currently dragging ──
        if mouse["drawing"]:
            # Draw a CYAN rectangle while dragging (so user can see it)
            cv2.rectangle(
                display,
                (mouse["start_x"], mouse["start_y"]),
                (mouse["end_x"], mouse["end_y"]),
                (255, 255, 0),  # Cyan color
                2
            )

        # ── When user finishes drawing a rectangle ──
        if mouse["rect_ready"]:
            # Calculate the rectangle coordinates (handle any drag direction)
            x1 = min(mouse["start_x"], mouse["end_x"])
            y1 = min(mouse["start_y"], mouse["end_y"])
            x2 = max(mouse["start_x"], mouse["end_x"])
            y2 = max(mouse["start_y"], mouse["end_y"])

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(display_w, x2)
            y2 = min(display_h, y2)

            # Store the rectangle
            last_rect = (x1, y1, x2, y2)

            # Crop the selected region from the display frame
            cropped = display[y1:y2, x1:x2]

            if cropped.size > 0:
                # Preprocess and run OCR on the cropped region
                preprocessed = preprocess_for_ocr(cropped)
                last_ocr_text = run_ocr(preprocessed)

                if last_ocr_text:
                    print(f"[OCR RESULT] {last_ocr_text}")

            # Mark as processed (don't re-run OCR every frame)
            mouse["rect_ready"] = False

        # ── Draw the persistent GREEN rectangle on the frame ──
        if last_rect is not None:
            x1, y1, x2, y2 = last_rect
            # Draw GREEN rectangle around selected area
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ── Show OCR text in GREEN above the rectangle ──
            if last_ocr_text:
                # Split text into lines and draw each above the box
                lines = last_ocr_text.split("\n")
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                line_gap = 30

                for i, line in enumerate(lines):
                    text_y = y1 - 10 - (len(lines) - 1 - i) * line_gap
                    if text_y < 20:
                        text_y = y2 + 25 + i * line_gap  # Put below if no space above

                    # Black background for readability
                    (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
                    cv2.rectangle(display, (x1, text_y - th - 5), (x1 + tw + 10, text_y + 5), (0, 0, 0), -1)
                    # Green text
                    cv2.putText(display, line, (x1 + 5, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            # ── Show ZOOMED view in separate window ──
            cropped_region = display[y1:y2, x1:x2]
            if cropped_region.size > 0:
                # Zoom the cropped region to 500px wide
                crop_h, crop_w = cropped_region.shape[:2]
                zoom_w = 500
                zoom_scale = zoom_w / crop_w if crop_w > 0 else 1
                zoom_h = int(crop_h * zoom_scale)
                zoomed = cv2.resize(cropped_region, (zoom_w, max(zoom_h, 50)), interpolation=cv2.INTER_CUBIC)

                # Add green border
                cv2.rectangle(zoomed, (0, 0), (zoomed.shape[1] - 1, zoomed.shape[0] - 1), (0, 255, 0), 3)

                # Add OCR text at the bottom of zoomed view
                if last_ocr_text:
                    text_lines = last_ocr_text.split("\n")
                    # Add black bar at bottom for text
                    bar_height = 35 * len(text_lines) + 10
                    text_bar = np.zeros((bar_height, zoom_w, 3), dtype=np.uint8)
                    for i, line in enumerate(text_lines):
                        cv2.putText(
                            text_bar, line, (10, 30 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
                        )
                    # Stack zoomed image with text bar
                    zoomed = np.vstack([zoomed, text_bar])

                cv2.imshow("Zoomed Text Region", zoomed)

        # ── Draw instructions on the frame ──
        cv2.rectangle(display, (0, 0), (display_w, 30), (0, 0, 0), -1)
        cv2.putText(
            display, "Draw rectangle on text to OCR  |  'c' = clear  |  'q' = quit",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA
        )

        # ── Show main video window ──
        cv2.imshow("Real-Time OCR Video", display)

        # ── Keyboard Controls ──
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to quit
        if key == ord("q") or key == ord("Q"):
            print("[INFO] Quitting...")
            break

        # Press 'c' to clear the selection
        if key == ord("c") or key == ord("C"):
            last_rect = None
            last_ocr_text = ""
            mouse["rect_ready"] = False
            try:
                cv2.destroyWindow("Zoomed Text Region")
            except Exception:
                pass
            print("[INFO] Selection cleared.")

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program exited cleanly.")


if __name__ == "__main__":
    main()