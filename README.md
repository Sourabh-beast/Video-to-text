# Real-Time OCR for Traffic and Surveillance Video

This project is optimized for free deployment on Render.

OCR runs in the browser using Tesseract.js, and Flask only serves the web pages. This removes heavy server-side ML dependencies and makes free-tier hosting practical.

## Features

- Webcam OCR with manual region selection
- Local video OCR with manual region selection
- In-browser OCR (no heavy OCR model on server)
- Free-tier friendly deployment on Render

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Backend runtime |
| Flask | Web app server |
| Gunicorn | Production WSGI server |
| HTML/CSS/JavaScript | UI and interaction |
| Tesseract.js | Browser-side OCR engine |

## Project Structure

```
Video-to-text/
|-- app.py
|-- requirements.txt
|-- runtime.txt
|-- .render.yaml
|-- templates/
|   |-- index.html
|   |-- webcam.html
|   `-- upload.html
`-- static/
    `-- css/
        `-- style.css
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

## Render Free Deployment

1. Push this project to GitHub.
2. In Render, create a new Blueprint and select your repo.
3. Render will use `.render.yaml` automatically.
4. Deploy.

Current `.render.yaml` is already free-tier friendly:

- `plan: free`
- lightweight Python web service
- no paid persistent disk needed

## Usage

### Webcam

1. Open Webcam page.
2. Click Start Camera.
3. Draw a box around text.
4. OCR result appears in the right panel.

### Upload Video

1. Open Upload Video page.
2. Select or drop a local video file.
3. Pause on a clear frame (optional but recommended).
4. Draw a box around text.
5. OCR result appears in the right panel.

## Important Notes

- First OCR run in browser can take longer while language data loads.
- OCR speed depends on end-user device performance.
- Webcam access requires HTTPS on deployed site (Render provides HTTPS).

## License

Educational use.
