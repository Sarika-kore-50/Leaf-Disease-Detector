# 🌿 Leaf Disease Detection using Deep Learning
A Flask-based web application to detect diseases in plant leaves using a trained deep learning model.
## 📂 Project Structure
leaf-disease-detector/
│
├── app.py                  # Flask backend server
├── requirements.txt        # Python libraries needed
├── README.md                # Project description (this file)
├── static/
│   └── uploadimages/        # Folder for uploaded images
├── templates/
│   └── index.html           # Frontend HTML file   
└── model training/
    └── plant_disease_recog_model_pwp.keras  # Trained deep learning model

## ⚙️ Setup Instructions
### Clone the Repository
git clone https://github.com/YOUR-USERNAME/leaf-disease-detector.git
cd leaf-disease-detector
### Create a Virtual Environment (Recommended)
python -m venv env
source env/bin/activate  # Linux / Mac
env\Scripts\activate     # Windows
### Install Dependencies
pip install -r requirements.txt
### Run the App
python app.py
Open your browser and go to http://127.0.0.1:5000/
## 📦 requirements.txt
Here’s the requirements.txt you can use:
Flask
tensorflow
Pillow
numpy

