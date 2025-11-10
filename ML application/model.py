import numpy as np
import tensorflow as tf
from PIL import Image
import io
from huggingface_hub import hf_hub_download

# Download pre-trained MobileNetV2 pneumonia model (h5) from Hugging Face
# Model card: ayushirathour/chest-xray-pneumonia-detection
MODEL_LOCAL_PATH = hf_hub_download(
    repo_id="ayushirathour/chest-xray-pneumonia-detection",
    filename="best_chest_xray_model.h5",
)

model = tf.keras.models.load_model(MODEL_LOCAL_PATH)

def _preprocess_bytes(file_bytes, target=(224, 224)):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(target)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_pneumonia(file_bytes):
    x = _preprocess_bytes(file_bytes)
    prob = float(model.predict(x, verbose=0)[0][0])
    is_pneu = prob >= 0.5
    result = "Pneumonia Detected" if is_pneu else "Normal"
    confidence = prob if is_pneu else 1 - prob
    return {"result": result, "confidence": confidence}
