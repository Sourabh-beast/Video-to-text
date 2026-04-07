import os

from flask import Flask, render_template  # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/health")
def health():
    return {"ok": True}, 200


if __name__ == "__main__":
    print("=" * 64)
    print("  Real-Time OCR (Browser-side Tesseract.js)")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 64)
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))