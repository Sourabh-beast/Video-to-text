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
| **OpenCV** | Video capture, image processing, display |
| **pytesseract** | Python wrapper for Tesseract OCR engine |
| **Tesseract OCR** | Google's open-source OCR engine (LSTM neural network) |
| **NumPy** | Numerical operations on image arrays |

---

## 📋 Features

1. ✅ **Dual video input** – Video file (`sample_video.mp4`) or webcam
2. ✅ **Image preprocessing** – Resize, grayscale, Gaussian blur, thresholding
3. ✅ **Text detection** – Locates text regions using Tesseract
4. ✅ **Green rectangle overlay** – Highlights detected text areas
5. ✅ **Zoomed text view** – Separate window with enlarged text region
6. ✅ **Text display** – Recognized text shown in green font on the frame
7. ✅ **User controls** – Press `q` to quit safely
8. ✅ **Dual output windows** – "Real-Time OCR Video" + "Zoomed Text Region"
9. ✅ **FPS counter** – Live performance monitoring
10. ✅ **Confidence filtering** – Only shows high-confidence detections

---

## 🚀 Installation & Setup

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Tesseract OCR Engine

**Windows:**
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR\`)
3. The program auto-detects common installation paths

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Step 3: Run the Program

```bash
python ocr_video.py
```

---

## 🎮 How to Use

1. Place your video file as `sample_video.mp4` in the project folder (or change `VIDEO_FILE` in the code)
2. Run `python ocr_video.py`
3. Two windows will appear:
   - **"Real-Time OCR Video"** — The main video with green rectangles and text
   - **"Zoomed Text Region"** — Enlarged view of the detected text area
4. Press **`q`** to stop and exit

### To use Webcam:
Change this line in `ocr_video.py`:
```python
USE_WEBCAM = True
```

---

## 📁 Project Structure

```
Real-Time-OCR-System/
├── ocr_video.py          # Main program (complete OCR system)
├── sample_video.mp4      # Sample traffic/surveillance video
├── requirements.txt      # Python dependencies
└── README.md             # This documentation file
```

---

## 📝 License

This project is created for educational purposes as a Final Year Project.
