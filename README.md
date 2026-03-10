# Real-Time OCR System for Video-Based Text Recognition in Traffic and Surveillance

## 📌 Project Overview

This project develops a **real-time OCR (Optical Character Recognition) system** that extracts and recognizes text from traffic and surveillance video feeds. It processes live video frames to detect and read text such as:

- 🚗 **Vehicle license plates**
- 🛣️ **Road signboards**
- 📹 **Text appearing in CCTV/surveillance recordings**

The system handles real-world challenges like motion blur, low resolution, varying lighting conditions, and complex backgrounds.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **OpenCV** | Video capture, frame extraction, image preprocessing (CLAHE, resize), display |
| **EasyOCR** | Deep learning-based OCR engine — uses CRAFT model for text detection + recognition network for text reading |
| **NumPy** | Numerical operations on image arrays (pixels are number matrices) |
| **Flask** | Lightweight Python web framework for serving the web application |
| **HTML/CSS/JavaScript** | Frontend — UI design, video player, rectangle drawing, results display |

---

## 📋 Features

### Web Application (`app.py`)
1. ✅ **Home page** — Professional dark theme with project title and navigation
2. ✅ **Live Webcam mode** — Real-time text detection from camera feed
3. ✅ **Video Upload mode** — Upload MP4/AVI/MOV/MKV traffic footage and detect text
4. ✅ **Interactive region selection** — Click & drag to draw rectangle on any text area
5. ✅ **Zoomed view** — Selected region shown enlarged in sidebar (auto-clears after 5s)
6. ✅ **OCR results display** — Detected text shown in green (auto-clears after 10s)
7. ✅ **Drag & drop upload** — Drag video files directly into the browser
8. ✅ **Responsive design** — Works on different screen sizes

### Desktop Application (`ocr_video.py`)
1. ✅ **Dual video input** — Video file (`sample_video.mp4`) or webcam
2. ✅ **Green rectangle overlay** — Highlights selected text areas
3. ✅ **Zoomed text window** — Separate OpenCV window with enlarged text region + OCR result
4. ✅ **Text display on frame** — Recognized text shown in green font above/below the rectangle
5. ✅ **Keyboard controls** — Press `c` to clear selection, `q` to quit

---

## 🔄 How It Works (OCR Pipeline)```
Video (file or webcam)
    ↓
OpenCV extracts frames
    ↓
User draws rectangle on text area
    ↓
Selected region is cropped
    ↓
Preprocessing (CLAHE contrast enhancement + upscaling)
    ↓
EasyOCR detects + reads text (CRAFT + Recognition Network)
    ↓
Result displayed on screen (green text overlay)
```

### Key Preprocessing Steps:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — Enhances contrast in images with uneven lighting, making dim/shadowed text clearly visible
- **Upscaling** — Small text regions are scaled up 3-5x for better OCR accuracy (OCR needs ~300 DPI)

### Why EasyOCR?
EasyOCR uses deep learning models that are specifically trained for **scene text** (text in natural images like photos, traffic footage, CCTV). Unlike traditional OCR engines that work well only on scanned documents, EasyOCR handles:
- Low resolution and motion blur
- Complex backgrounds and varying fonts
- Different lighting conditions (shadows, night, reflections)

---

## 🚀 Installation & Setup

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — Video and image processing
- `numpy` — Array operations
- `flask` — Web server
- `easyocr` — Deep learning OCR (automatically installs PyTorch)

> **Note:** First install may download ~1-2GB (PyTorch + EasyOCR models). This is a one-time download.

### Step 2: Run the Web Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### Step 3 (Optional): Run the Desktop Version

```bash
python ocr_video.py
```

> **Note:** For the desktop version, place your video file as `sample_video.mp4` in the project folder, or change the `VIDEO_FILE` variable in `ocr_video.py`.

---

## 🎮 How to Use

### Web Application
1. Open **http://localhost:5000** in your browser
2. Choose **Live Webcam** or **Upload Video**
3. For Upload: drag & drop a video file or click "Choose Video"
4. Once video is playing, **click and drag** to draw a rectangle on any text
5. The **zoomed region** and **detected text** appear in the sidebar
6. Zoom auto-clears after 5 seconds, text after 10 seconds

### Desktop Application
1. Run `python ocr_video.py`
2. Video will play in the "Real-Time OCR Video" window
3. **Click and drag** to draw a rectangle on any text area
4. The zoomed view appears in a separate "Zoomed Text Region" window
5. Press **`c`** to clear selection, **`q`** to quit

### To use Webcam (Desktop version):
Change this line in `ocr_video.py`:
```python
USE_WEBCAM = True
```

---

---

## 📁 Project Structure

```
Real-Time-OCR-System/
├── app.py                  # Flask web server (main backend)
├── ocr_video.py            # Desktop version (OpenCV windows)
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
├── templates/
│   ├── index.html          # Home page
│   ├── webcam.html         # Live webcam page
│   └── upload.html         # Video upload page
├── static/
│   └── css/
│       └── style.css       # Website styling (dark professional theme)
└── uploads/                # Uploaded videos stored here (auto-created)
```

---

## 🌍 Real-World Applications

- **Traffic Monitoring** — Read bus/vehicle numbers from traffic cameras
- **Toll Booth Automation** — Automatic license plate recognition
- **CCTV Surveillance** — Extract text from surveillance footage
- **Smart City Systems** — Road sign detection for navigation/autonomous vehicles
- **Parking Management** — Automatic number plate logging

---

## 📝 License

This project is created for educational purposes as a Final Year Major Project.
