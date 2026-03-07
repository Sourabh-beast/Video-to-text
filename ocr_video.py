# =============================================================================
# Real-Time OCR System for Video-Based Text Recognition
# in Traffic and Surveillance
# =============================================================================
# Final Year Project
#
# Technologies: Python, OpenCV, pytesseract (Tesseract OCR), NumPy
#
# This program reads a video file (or webcam feed) frame-by-frame,
# preprocesses each frame for better OCR accuracy, detects text regions,
# draws green rectangles around them, displays recognized text on the frame,
# and shows a zoomed view of detected text in a separate window.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import cv2              # OpenCV – video capture, image processing, drawing
import pytesseract      # Python wrapper for Google's Tesseract OCR engine
import numpy as np      # NumPy – numerical array operations on images
import os               # OS utilities – file/path checks
import sys              # System utilities – for clean exit

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TESSERACT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Tesseract must be installed on your system.
# On Windows, it is typically installed at one of these paths.
# If your installation is elsewhere, update the path below.

def setup_tesseract():
    """
    Locate the Tesseract executable and configure pytesseract.
    Checks common Windows installation paths automatically.
    """
    # List of common Tesseract installation paths on Windows
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(
            os.getenv("USERNAME", "")
        ),
    ]

    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"[INFO] Tesseract found at: {path}")
            return True

    # If none of the common paths work, check if tesseract is on the PATH
    # (e.g., on Linux/macOS or if it was added to Windows PATH)
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[INFO] Tesseract version {version} found on system PATH.")
        return True
    except Exception:
        pass

    print("[ERROR] Tesseract OCR engine not found!")
    print("        Please install Tesseract from:")
    print("        https://github.com/tesseract-ocr/tesseract")
    print("        Then update the path in setup_tesseract() if needed.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: VIDEO INPUT
# ─────────────────────────────────────────────────────────────────────────────
# The program supports two input modes:
#   1. Video file  – e.g., "sample_video.mp4"
#   2. Webcam feed – using cv2.VideoCapture(0)
#
# Change the variable USE_WEBCAM to True if you want to use your webcam.
# Otherwise, set VIDEO_FILE to the path of your video file.

USE_WEBCAM = False                # Set True to use webcam instead of file
VIDEO_FILE = "sample_video.mp4"   # Path to your video file


def open_video_source():
    """
    Open and return a cv2.VideoCapture object.
    Uses webcam if USE_WEBCAM is True, otherwise uses VIDEO_FILE.
    """
    if USE_WEBCAM:
        print("[INFO] Opening webcam (device 0)...")
        cap = cv2.VideoCapture(0)
    else:
        # Check if the video file exists before trying to open it
        if not os.path.exists(VIDEO_FILE):
            print(f"[ERROR] Video file not found: {VIDEO_FILE}")
            print("        Place your video file in the same folder as this script,")
            print("        or update the VIDEO_FILE variable with the correct path.")
            sys.exit(1)
        print(f"[INFO] Opening video file: {VIDEO_FILE}")
        cap = cv2.VideoCapture(VIDEO_FILE)

    # Verify the source opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        sys.exit(1)

    return cap


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
# Before we feed a frame to the OCR engine, we preprocess it to improve
# text recognition accuracy.  Each step has a specific purpose:
#
#   1. RESIZING  – Enlarge the frame so that small text (like license plates)
#                  becomes large enough for Tesseract to recognize.
#
#   2. GRAYSCALE – Convert from 3-channel BGR colour to a single-channel
#                  grayscale image.  Tesseract works better on grayscale
#                  because colour information is irrelevant for text shape.
#
#   3. GAUSSIAN BLUR – Slightly smooth the image to reduce high-frequency
#                      noise (camera grain, compression artifacts).  This
#                      prevents Tesseract from confusing noise with text.
#
#   4. THRESHOLDING – Convert the grayscale image to pure black-and-white.
#                     This maximizes contrast between text and background,
#                     making character boundaries crisp and clear for OCR.

def preprocess_frame(frame):
    """
    Apply a series of preprocessing steps to improve OCR accuracy.

    Parameters
    ----------
    frame : numpy.ndarray
        The original BGR video frame.

    Returns
    -------
    processed : numpy.ndarray
        A binary (black & white) image optimized for OCR.
    gray : numpy.ndarray
        The intermediate grayscale image (used for display).
    """
    # Step 1: RESIZE – scale the frame up by 2x for better OCR on small text
    height, width = frame.shape[:2]
    resized = cv2.resize(
        frame,
        (width * 2, height * 2),
        interpolation=cv2.INTER_CUBIC   # high-quality upscaling
    )

    # Step 2: GRAYSCALE – convert BGR → single-channel gray
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 3: GAUSSIAN BLUR – reduce noise with a 5×5 kernel
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: THRESHOLDING – adaptive threshold handles uneven lighting
    #         (e.g., shadows on road signs, headlamp glare at night)
    processed = cv2.adaptiveThreshold(
        blurred,
        255,                              # maximum pixel value
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # weighted mean of neighbourhood
        cv2.THRESH_BINARY,                # output binary image
        11,                               # block size (neighbourhood)
        2                                 # constant subtracted from mean
    )

    return processed, gray


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TEXT DETECTION (pytesseract)
# ─────────────────────────────────────────────────────────────────────────────
# pytesseract.image_to_data() returns detailed information about every
# detected word: its bounding box (x, y, width, height), confidence, and
# the recognized text string.
#
# We filter results by a minimum confidence threshold to discard random
# noise that Tesseract might mis-identify as text.

# Minimum OCR confidence (0-100) to accept a detection
MIN_CONFIDENCE = 40


def detect_text_regions(processed_frame):
    """
    Use Tesseract to detect text in the preprocessed frame.

    Parameters
    ----------
    processed_frame : numpy.ndarray
        A preprocessed (binary/thresholded) image.

    Returns
    -------
    detections : list of dict
        Each dict has keys: 'x', 'y', 'w', 'h', 'text', 'conf'
        representing one detected text region.
    """
    # Tesseract configuration:
    #   --oem 3  : Use the default OCR Engine Mode (LSTM neural net)
    #   --psm 6  : Assume a uniform block of text (good for mixed content)
    custom_config = r"--oem 3 --psm 6"

    try:
        # image_to_data returns a tab-separated table of detections
        data = pytesseract.image_to_data(
            processed_frame,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        print(f"[WARNING] Tesseract error: {e}")
        return []

    detections = []
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        # Extract the recognized text and strip whitespace
        text = data["text"][i].strip()

        # Skip empty detections
        if not text:
            continue

        # Get the confidence score (how sure Tesseract is)
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0

        # Only keep detections above our confidence threshold
        if conf < MIN_CONFIDENCE:
            continue

        # Bounding box coordinates (in the preprocessed image's coordinate system)
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        detections.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "text": text,
            "conf": conf,
        })

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: DRAWING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
# These functions draw visual overlays on the video frame:
#   • Green rectangles around detected text areas
#   • Recognized text labels above each rectangle
#   • An information banner at the top of the frame

def draw_detections(display_frame, detections, scale_x, scale_y):
    """
    Draw GREEN RECTANGLES and text labels on the display frame.

    Because preprocessing scaled the image by 2x, we must divide the
    coordinates by 2 to map them back to the original frame size.

    Parameters
    ----------
    display_frame : numpy.ndarray
        The original (colour) frame to draw on.
    detections : list of dict
        Text regions returned by detect_text_regions().
    scale_x, scale_y : float
        Scale factors to convert preprocessed coords → display coords.

    Returns
    -------
    display_frame : numpy.ndarray
        The frame with overlays drawn on it.
    best_region : dict or None
        The detection with the highest confidence (for zooming).
    """
    best_region = None
    best_conf = -1

    for det in detections:
        # Convert coordinates from preprocessed image → display image
        x = int(det["x"] / scale_x)
        y = int(det["y"] / scale_y)
        w = int(det["w"] / scale_x)
        h = int(det["h"] / scale_y)
        text = det["text"]
        conf = det["conf"]

        # ── Draw a GREEN RECTANGLE around the text area ──
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(
            display_frame,
            top_left,
            bottom_right,
            color=(0, 255, 0),   # Green in BGR
            thickness=2
        )

        # ── Draw the recognized text ABOVE the rectangle ──
        # Create a label string with confidence percentage
        label = f"{text} ({conf}%)"

        # Calculate text size for the background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Draw a filled black rectangle as background for readability
        cv2.rectangle(
            display_frame,
            (x, y - text_h - 10),
            (x + text_w + 6, y - 2),
            (0, 0, 0),          # Black background
            -1                   # Filled
        )

        # Draw the text in bright GREEN
        cv2.putText(
            display_frame,
            label,
            (x + 3, y - 6),
            font,
            font_scale,
            (0, 255, 0),        # Green text
            thickness,
            cv2.LINE_AA          # Anti-aliased for smooth rendering
        )

        # Track the highest-confidence detection for the zoomed view
        if conf > best_conf:
            best_conf = conf
            best_region = {"x": x, "y": y, "w": w, "h": h, "text": text}

    return display_frame, best_region


def draw_info_banner(frame, text_count, fps):
    """
    Draw an information banner at the top of the frame showing
    the number of detected text regions and current FPS.
    """
    banner_text = f"Detected Texts: {text_count}  |  FPS: {fps:.1f}  |  Press 'q' to quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        frame, banner_text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
    )
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: ZOOMED TEXT VIEW
# ─────────────────────────────────────────────────────────────────────────────
# When text is detected, we crop the region from the original frame,
# enlarge it, and display it in a separate window called "Zoomed Text Region".
# This is especially useful for reading small text like license plates
# or distant road signs.

# Desired width for the zoomed display window (pixels)
ZOOM_DISPLAY_WIDTH = 400


def show_zoomed_region(frame, region):
    """
    Crop, zoom/enlarge, and display the detected text region in a
    separate window called 'Zoomed Text Region'.

    Parameters
    ----------
    frame : numpy.ndarray
        The original display frame.
    region : dict
        Must have keys 'x', 'y', 'w', 'h', 'text'.
    """
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    text = region["text"]

    # Add padding around the region for context
    pad = 15
    frame_h, frame_w = frame.shape[:2]

    # Calculate padded coordinates, clamped to frame boundaries
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_w, x + w + pad)
    y2 = min(frame_h, y + h + pad)

    # Crop the region from the original frame
    cropped = frame[y1:y2, x1:x2]

    # Safety check: skip if the crop is empty
    if cropped.size == 0:
        return

    # Calculate zoom scale to make the cropped region fill the display width
    crop_h, crop_w = cropped.shape[:2]
    zoom_scale = ZOOM_DISPLAY_WIDTH / crop_w if crop_w > 0 else 1
    zoomed_w = int(crop_w * zoom_scale)
    zoomed_h = int(crop_h * zoom_scale)

    # Resize (zoom/enlarge) the cropped region
    zoomed = cv2.resize(
        cropped,
        (zoomed_w, zoomed_h),
        interpolation=cv2.INTER_CUBIC
    )

    # Draw a green border around the zoomed image
    cv2.rectangle(zoomed, (0, 0), (zoomed_w - 1, zoomed_h - 1), (0, 255, 0), 3)

    # Draw the recognized text at the bottom of the zoomed image
    label = f"OCR: {text}"
    cv2.rectangle(zoomed, (0, zoomed_h - 40), (zoomed_w, zoomed_h), (0, 0, 0), -1)
    cv2.putText(
        zoomed, label, (10, zoomed_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
    )

    # Display in the "Zoomed Text Region" window
    cv2.imshow("Zoomed Text Region", zoomed)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────
# This is the heart of the program.  It reads frames one-by-one from the
# video source, preprocesses them, detects text, draws overlays, and
# shows both the annotated video and the zoomed text region.

def main():
    """Main function — entry point of the Real-Time OCR system."""

    # ── Step 1: Setup Tesseract ──
    if not setup_tesseract():
        print("[FATAL] Cannot proceed without Tesseract. Exiting.")
        sys.exit(1)

    # ── Step 2: Open video source ──
    cap = open_video_source()

    # Get video properties for display
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Video resolution: {frame_width}x{frame_height}, FPS: {video_fps:.1f}")

    # ── Step 3: Create display windows ──
    # Window 1: Main video with overlays
    cv2.namedWindow("Real-Time OCR Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time OCR Video", 960, 540)

    # Window 2: Zoomed text region (created dynamically when text is found)

    print("[INFO] Processing started. Press 'q' to quit.")
    print("=" * 60)

    # Variables for FPS calculation
    fps = 0.0
    frame_count = 0
    start_time = cv2.getTickCount()

    # We process OCR every N frames to keep the display smooth
    # (OCR is computationally expensive)
    OCR_EVERY_N_FRAMES = 3
    last_detections = []       # Store last OCR results for drawing
    last_best_region = None    # Store last best region for zoom window

    # ── Step 4: Frame-by-frame processing loop ──
    while True:
        # Read one frame from the video source
        ret, frame = cap.read()

        # If no frame was read, the video has ended (or camera disconnected)
        if not ret:
            print("[INFO] End of video or camera disconnected.")
            break

        frame_count += 1

        # Resize frame for consistent display (960px wide)
        display_width = 960
        h, w = frame.shape[:2]
        display_scale = display_width / w
        display_h = int(h * display_scale)
        display_frame = cv2.resize(frame, (display_width, display_h))

        # ── Run OCR every N frames for performance ──
        if frame_count % OCR_EVERY_N_FRAMES == 0:
            # Preprocess the display frame for OCR
            processed, gray = preprocess_frame(display_frame)

            # Calculate the scale factor between preprocessed and display
            # (preprocessing scales by 2x)
            scale_x = 2.0
            scale_y = 2.0

            # Detect text regions using Tesseract
            last_detections = detect_text_regions(processed)
            last_best_region = None  # Will be set by draw_detections

        # ── Draw detections on every frame (even between OCR runs) ──
        annotated_frame = display_frame.copy()

        if last_detections:
            annotated_frame, best_region = draw_detections(
                annotated_frame, last_detections, scale_x=2.0, scale_y=2.0
            )
            if best_region is not None:
                last_best_region = best_region

        # ── Calculate FPS ──
        elapsed_ticks = cv2.getTickCount() - start_time
        elapsed_seconds = elapsed_ticks / cv2.getTickFrequency()
        if elapsed_seconds > 0:
            fps = frame_count / elapsed_seconds

        # ── Draw info banner ──
        annotated_frame = draw_info_banner(
            annotated_frame, len(last_detections), fps
        )

        # ── Display Window 1: "Real-Time OCR Video" ──
        cv2.imshow("Real-Time OCR Video", annotated_frame)

        # ── Display Window 2: "Zoomed Text Region" ──
        if last_best_region is not None:
            show_zoomed_region(display_frame, last_best_region)
        else:
            # Show a placeholder when no text is detected
            placeholder = np.zeros((100, ZOOM_DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "No text detected...",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Zoomed Text Region", placeholder)

        # ── USER CONTROL: Press 'q' to quit ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("[INFO] User pressed 'q'. Stopping...")
            break

    # ── Step 5: Cleanup ──
    # Release the video capture object (frees the camera or file handle)
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("[INFO] Cleanup complete. Program exited.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: PROGRAM ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()