from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2, numpy as np, tensorflow as tf, base64, os
from utils import preprocess_image

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "emnist_cnn.h5")

model = tf.keras.models.load_model(MODEL_PATH)

labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image"}), 400

    img_data = base64.b64decode(data["image"].split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed = preprocess_image(img)
    preds = model.predict(processed, verbose=0)[0]

    top3 = preds.argsort()[-3:][::-1]
    results = [
        {"char": labels[i], "confidence": float(preds[i])}
        for i in top3
    ]

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)

