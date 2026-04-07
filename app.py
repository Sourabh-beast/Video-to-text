import os

from flask import Flask, render_template  # type: ignore


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


if __name__ == "__main__":
    print("=" * 64)
    print("  Real-Time OCR (Browser-side Tesseract.js)")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 64)
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))