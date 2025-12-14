from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import cv2

# ==========================================================
# Flask App Configuration
# ==========================================================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Load model with placeholder metrics (for compatibility)
# ==========================================================
def dice_coef(y_true, y_pred): return tf.constant(0.0)
def precision(y_true, y_pred): return tf.constant(0.0)
def sensitivity(y_true, y_pred): return tf.constant(0.0)
def specificity(y_true, y_pred): return tf.constant(0.0)
def dice_coef_necrotic(y_true, y_pred): return tf.constant(0.0)
def dice_coef_edema(y_true, y_pred): return tf.constant(0.0)
def dice_coef_enhancing(y_true, y_pred): return tf.constant(0.0)

custom_objects = {
    'dice_coef': dice_coef,
    'precision': precision,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'dice_coef_necrotic': dice_coef_necrotic,
    'dice_coef_edema': dice_coef_edema,
    'dice_coef_enhancing': dice_coef_enhancing
}

MODEL_PATH = r"D:\Projects\Brain-Tumor-Segmentation-main\brain_tumor_model.keras"
model = load_model(MODEL_PATH, custom_objects=custom_objects)

# ==========================================================
# Preprocessing Function (Middle Slice)
# ==========================================================
def preprocess_nifti_middle_slice(file_path, target_shape=(128, 128), num_channels=2):
    nii = nib.load(file_path)
    img = nii.get_fdata()
    z_mid = img.shape[2] // 2
    original_slice = img[:, :, z_mid]

    # Resize for model input
    slice_resized = zoom(original_slice,
                         (target_shape[0] / original_slice.shape[0],
                          target_shape[1] / original_slice.shape[1]),
                         order=1)
    slice_resized = slice_resized[..., np.newaxis]
    if num_channels == 2:
        slice_resized = np.concatenate([slice_resized, slice_resized], axis=-1)
    slice_resized = slice_resized / np.max(slice_resized)
    slice_input = np.expand_dims(slice_resized, axis=0)
    return slice_input, original_slice

# ==========================================================
# Overlay Segmentation Function
# ==========================================================
def overlay_segmentation(original_slice, prediction):
    # Convert prediction to label map
    if prediction.ndim == 4 and prediction.shape[-1] > 1:
        seg = np.argmax(prediction, axis=-1)[0]
    else:
        seg = prediction[0, :, :, 0]

    seg_resized = cv2.resize(seg.astype(np.uint8),
                             (original_slice.shape[1], original_slice.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    original_rgb = np.stack([original_slice] * 3, axis=-1)
    original_rgb = (original_rgb / np.max(original_rgb) * 255).astype(np.uint8)

    color_map = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}  # Red, Green, Blue
    alpha = 0.4
    overlay = original_rgb.copy()

    for label, color in color_map.items():
        mask = seg_resized == label
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.uint8)

    return overlay

# ==========================================================
# Improved Confidence Score Function
# ==========================================================
def calculate_confidence(prediction):
    """
    Calculates tumor confidence based on predicted tumor voxel probabilities.
    Only considers regions with high tumor likelihood.
    """
    if prediction.shape[-1] > 1:
        # Assume class index 0 = background, class 1 = tumor
        tumor_probs = prediction[0, :, :, 1]
    else:
        tumor_probs = prediction[0, :, :, 0]

    # Focus only on pixels where tumor prob > 0.5
    tumor_pixels = tumor_probs[tumor_probs > 0.5]
    if tumor_pixels.size > 0:
        confidence = float(np.mean(tumor_pixels))
    else:
        confidence = 0.0
    return confidence

# ==========================================================
# Prediction Route
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Preprocess and predict
        slice_input, original_slice = preprocess_nifti_middle_slice(file_path)
        pred = model.predict(slice_input)

        # Generate overlay and confidence
        overlay_img = overlay_segmentation(original_slice, pred)
        confidence_score = calculate_confidence(pred)

        # Save overlay
        output_filename = f"{os.path.splitext(filename)[0]}_overlay.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, overlay_img)

        return jsonify({
            "image_url": f"/output/{output_filename}",
            "confidence": round(confidence_score * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================================
# Serve Output Images
# ==========================================================
@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/png")

# ==========================================================
# Run Flask App
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)
