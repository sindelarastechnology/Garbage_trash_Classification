import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'content/garbage_classifier_mobilenetv2.h5'
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASSES_ID = ['Kardus', 'Kaca', 'Logam', 'Kertas', 'Plastik', 'Sampah Organik']

model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224))
    img_in = np.expand_dims(img_res.astype('float32') / 255.0, axis=0)
    return img_in, img_rgb

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diupload'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            img_in, img_rgb = preprocess_image(filepath)
            preds = model.predict(img_in, verbose=0)[0]
            pred_class = CLASSES[np.argmax(preds)]
            confidence = float(np.max(preds) * 100)

            predictions = {}
            for i, cls in enumerate(CLASSES):
                predictions[cls] = round(float(preds[i] * 100), 2)

            os.remove(filepath)

            return jsonify({
                'success': True,
                'pred_class': pred_class,
                'pred_class_id': CLASSES_ID[CLASSES.index(pred_class)],
                'confidence': round(confidence, 2),
                'predictions': predictions
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050)
