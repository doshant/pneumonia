from flask import Flask, request, jsonify , send_file
from PIL import Image
import numpy as np
import io
from keras import backend as K
from keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import os
from flask_cors import CORS
import matplotlib
from PIL import Image
matplotlib.use('Agg') 
app = Flask(__name__)

CORS(app)
# Load your pre-trained model
model = load_model('CNN_model.h5')  # Ensure you load your model weights or entire model here

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = np.array(image)
    image = (image / 255.0).astype(np.float32)
    return np.expand_dims(image, axis=0)

def get_lime_explanation(image_array, model, true_label):
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_array[0].astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
    pred_label = np.argmax(model.predict(image_array))
    temp, mask = explanation.get_image_and_mask(pred_label, positive_only=False, num_features=10, hide_rest=False)
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.axis('off')
    plt.title('LIME Explanation')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    image = Image.open(file.stream)
    prepared_image = prepare_image(image, (256, 256))
    prediction = model.predict(prepared_image)
    pred_label = np.argmax(prediction, axis=1)[0]
    label_names = {0: 'Normal', 1: 'Pneumonia'}
    pred_class = label_names[pred_label]

    # Generate LIME explanation
    
    response = {
        'prediction': pred_class,
        'confidence': float(np.max(prediction)),
    }
    return jsonify(response), 200

@app.route('/explain', methods=['POST'])
def explain():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image file with Pillow
        image = Image.open(file.stream)
        # Depending on your model's requirements, you might need to convert the image
        # to grayscale, resize it, etc. Here's an example of resizing and converting to a numpy array
        image = image.resize((224, 224))  # Resize to the input size required by the model
        prepared_image = prepare_image(image, (256, 256)) # Convert image to numpy array
         # Model expects a batch of images

        # Use LIME explainer
        explainer = lime_image.LimeImageExplainer()
        pred_label = np.argmax(model.predict(prepared_image), axis=1)[0]
        explanation = explainer.explain_instance(prepared_image[0].astype('double'), model.predict, top_labels=5, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(pred_label, positive_only=False, num_features=10, hide_rest=False)

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.axis('off')

        # Save the explanation to a byte stream and send as a response
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
