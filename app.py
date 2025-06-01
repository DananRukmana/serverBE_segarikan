from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import os
import gdown

app = Flask(__name__)
CORS(app)

# === Download model dari Google Drive jika belum ada ===
def download_model_if_needed():
    os.makedirs("model", exist_ok=True)

    files = {
        "model_cek_ikan.h5": "1MSliz9qsueL4YO5sgc7A69FcpsG7uMQn",
        "model_cek_kepala_ikan.h5": "1bUNramVQApA1_cKkbD-5DWOrk_a0AthY",
        "model_last_ikan.h5": "1BsCNherYcG7I_DPsYwFJFcrczpbhJSvx"
    }

    for filename, file_id in files.items():
        path = os.path.join("model", filename)
        if not os.path.exists(path):
            print(f"🔽 Downloading {filename} ...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
        else:
            print(f"✅ {filename} already exists.")

download_model_if_needed()

# === Load Models ===
model_ikan = tf.keras.models.load_model("model/model_cek_ikan.h5", compile=False)
model_kepala = tf.keras.models.load_model("model/model_cek_kepala_ikan.h5", compile=False)
model_kesegaran = tf.keras.models.load_model("model/model_last_ikan.h5", compile=False)

# === Helper ===
def decode_image(base64_str):
    header, data = base64_str.split(",", 1)
    img_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return image.resize((224, 224))

def preprocess_image(img):
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 224, 224, 3)

def prediksi_model(img_tensor, model, threshold, kelas_utama, kelas_tidak):
    skor = model.predict(img_tensor)[0][0]
    hasil = kelas_utama if skor > threshold else kelas_tidak
    return hasil, float(skor)

# === Routing ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image = decode_image(data["image"])
        img_tensor = preprocess_image(image)

        hasil1, skor1 = prediksi_model(img_tensor, model_ikan, 0.8, "ikan", "bukan_ikan")
        if hasil1 == "bukan_ikan":
            return jsonify({"step": 1, "result": hasil1, "score": skor1})

        hasil2, skor2 = prediksi_model(img_tensor, model_kepala, 0.8, "kepala_ikan", "bukan_kepala")
        if hasil2 == "bukan_kepala":
            return jsonify({"step": 2, "result": hasil2, "score": skor2})

        hasil3, skor3 = prediksi_model(img_tensor, model_kesegaran, 0.5, "Tidak Segar", "Segar")
        return jsonify({
            "step": 3,
            "result": hasil3,
            "score": skor3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
