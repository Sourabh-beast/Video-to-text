# Real-Time OCR System for Video-Based Text Recognition in Traffic and Surveillance

A web-based OCR system for traffic and surveillance footage. It supports both webcam and uploaded videos, with manual region selection. OCR is powered by EasyOCR (CRAFT detector + CRNN recognizer) and runs on CPU.

## Features

- Webcam mode and video upload mode
- Manual rectangle selection for OCR (no auto-detect)
- Zoomed region preview shown for 5 seconds
- Detected text shown for 10 seconds
- OCR preprocessing (contrast + sharpening) for better accuracy

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Flask | Web server |
| OpenCV | Video capture and image processing |
| EasyOCR | OCR engine (CRAFT + CRNN) |
| NumPy | Array operations |
| HTML/CSS/JavaScript | UI and client logic |

## Project Structure

```
Real-Time-OCR-System/
├── app.py                 # Flask web server (main backend)
├── requirements.txt       # Python dependencies
├── README.md              # This documentation
├── templates/
│   ├── index.html          # Home page
│   ├── webcam.html         # Live webcam page
│   └── upload.html         # Video upload page
├── static/
│   └── css/
│       └── style.css       # Website styling
└── uploads/                # Uploaded videos (auto-created)
```

## Installation

Recommended Python version: 3.10

```bash
python -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
```

Note: First install will download EasyOCR model weights (about 1-2GB). This is a one-time download.

## Usage

Start the web app:

```bash
python app.py
```

Open http://localhost:5000 in your browser. The home page provides two options:

- Live Webcam
- Upload Video

### Webcam Flow

1) Open the Webcam page.
2) Click Start Camera.
3) Draw a rectangle on the text area.
4) The zoomed region appears for 5 seconds and the detected text for 10 seconds.

### Upload Flow

1) Open the Upload Video page.
2) Choose a video file and wait for it to play.
3) Draw a rectangle on the text area.
4) The zoomed region appears for 5 seconds and the detected text for 10 seconds.

## Tuning Accuracy and Speed

Adjust these in app.py:

- MANUAL_CONFIDENCE_MIN: higher value means stricter results
- MANUAL_OCR_MAX_SIDE: lower value speeds up OCR
- MANUAL_OCR_UPSCALE_MIN: higher value can improve small text detection

Tips for best results:

- Draw a tight rectangle around the text
- Use clear frames (pause the video if needed)
- Ensure good lighting and contrast

## License

This project is for educational use.
