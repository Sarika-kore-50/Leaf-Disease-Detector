from flask import Flask, render_template, request, redirect
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploadimages'

# Load your trained model
model = tf.keras.models.load_model("E:\Plant_diasesas\plant_disease_recog_model_pwp.keras")

# List of class labels, matching the order of the model's outputs
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___Healthy',
    'Blueberry___Healthy', 'Cherry___Powdery_mildew', 'Cherry___Healthy', 'Corn___Northern_Leaf_Blight',
    'Corn___Cercospora_Leaf_Spot', 'Corn___Healthy', 'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight',
    'Grape___Healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy',
    'Pepper___Bacterial_spot', 'Pepper___Healthy', 'Strawberry___Leaf_scorch', 'Strawberry___Healthy',
    'Wheat___Leaf_rust', 'Wheat___Healthy', 'Cotton___Healthy', 'Cucumber___Downy_mildew',
    'Cucumber___Healthy', 'Squash___Powdery_mildew', 'Squash___Healthy', 'Carrot___Alternaria_leaf_spot',
    'Carrot___Healthy', 'Cabbage___Black_rot', 'Cabbage___Healthy', 'Onion___Downy_mildew', 'Onion___Healthy',
    'Lettuce___Downy_mildew', 'Lettuce___Healthy', 'Peach___Leaf_curl', 'Peach___Healthy',
    'Pear___Fire_blight', 'Pear___Healthy', 'Almond___Brown_rot', 'Almond___Healthy', 'Pine___Needle_cast',
    'Pine___Healthy', 'Tomato___Early_blight', 'Tomato___Healthy'
]

# Corresponding treatments or cures
cures = {
    'Apple___Apple_scab': "Use fungicides and resistant apple varieties.",
    'Apple___Black_rot': "Prune and remove infected areas. Apply fungicides during the growing season.",
    'Apple___Cedar_apple_rust': "Remove nearby juniper hosts. Use resistant varieties and fungicides.",
    'Apple___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Blueberry___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Cherry___Powdery_mildew': "Prune infected areas, and apply fungicides.",
    'Cherry___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Corn___Northern_Leaf_Blight': "Use resistant varieties and apply fungicides.",
    'Corn___Cercospora_Leaf_Spot': "Use resistant varieties and apply fungicides.",
    'Corn___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Grape___Black_rot': "Prune infected areas and apply fungicides.",
    'Grape___Esca': "Remove infected vines and apply fungicides.",
    'Grape___Leaf_blight': "Use fungicides and remove infected leaves.",
    'Grape___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Tomato___Late_blight': "Use resistant varieties and apply recommended fungicides.",
    'Tomato___Leaf_Mold': "Ensure good ventilation and apply fungicides.",
    'Tomato___Septoria_leaf_spot': "Prune affected leaves, and apply fungicides.",
    'Tomato___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Potato___Early_blight': "Use resistant varieties and fungicides.",
    'Potato___Late_blight': "Remove infected areas and apply fungicides.",
    'Potato___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Pepper___Bacterial_spot': "Remove infected leaves, and apply copper-based fungicides.",
    'Pepper___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Strawberry___Leaf_scorch': "Prune infected leaves and apply fungicides.",
    'Strawberry___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Wheat___Leaf_rust': "Apply appropriate fungicides and use resistant varieties.",
    'Wheat___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Cotton___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Cucumber___Downy_mildew': "Apply fungicides and remove infected leaves.",
    'Cucumber___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Squash___Powdery_mildew': "Apply fungicides and prune infected areas.",
    'Squash___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Carrot___Alternaria_leaf_spot': "Remove infected leaves and apply fungicides.",
    'Carrot___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Cabbage___Black_rot': "Prune and remove infected areas. Use resistant varieties.",
    'Cabbage___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Onion___Downy_mildew': "Apply fungicides and ensure proper air circulation.",
    'Onion___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Lettuce___Downy_mildew': "Ensure good ventilation and apply fungicides.",
    'Lettuce___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Peach___Leaf_curl': "Prune infected areas and apply fungicides in the dormant season.",
    'Peach___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Pear___Fire_blight': "Prune infected areas and apply bactericides.",
    'Pear___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Almond___Brown_rot': "Prune infected areas and apply fungicides.",
    'Almond___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Pine___Needle_cast': "Prune infected needles and apply fungicides.",
    'Pine___Healthy': "No disease detected. Keep the plant healthy and well-maintained.",
    'Tomato___Early_blight': "Remove infected leaves and apply fungicides.",
    'Tomato___Healthy': "No disease detected. Keep the plant healthy and well-maintained."
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Image preprocessing
            img = Image.open(filepath).convert('RGB').resize((160, 160))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Prediction
            predictions = model.predict(img)
            pred_idx = np.argmax(predictions)

            # Retrieve the label based on the index
            label = class_labels[pred_idx]
            cure = cures.get(label, "No cure available for this disease.")

            prediction = {
                'name': label,
                'cause': "Cause for " + label,
                'cure': cure
            }

            return render_template('index.html', result=True, prediction=prediction, filename='uploadimages/' + filename)

    return render_template('index.html', result=False)

if __name__ == '__main__':
    app.run(debug=True)
