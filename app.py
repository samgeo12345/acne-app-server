# paste the Flask code I gave you here
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

CLASS_NAMES = {
    0: 'acne',
    1: 'blackhead',
    2: 'whitehead',
    3: 'pimple',
    4: 'spot',
    5: 'scar'
}

def estimate_severity(total):
    if total == 0:
        return "none"
    if total <= 5:
        return "mild"
    if total <= 20:
        return "moderate"
    return "severe"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    results = model(image, conf=0.25)[0]

    counts = {}
    total = 0
    avg_conf = 0

    if results.boxes is not None:
        cls = results.boxes.cls.cpu().numpy().astype(int)
        conf = results.boxes.conf.cpu().numpy()

        for c in cls:
            counts[CLASS_NAMES[c]] = counts.get(CLASS_NAMES[c], 0) + 1

        total = len(cls)
        avg_conf = float(conf.mean())

    severity = estimate_severity(total)

    return jsonify({
        "detections": counts,
        "total": total,
        "average_confidence": round(avg_conf, 3),
        "severity": severity
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
