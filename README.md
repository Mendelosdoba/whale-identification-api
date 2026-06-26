Whale Identification API

A two-stage computer vision application that identifies whales, dolphins, and porpoises from uploaded images.

The application first checks whether an image contains a cetacean. If it does, a second model predicts the most likely species.

How It Works
The user uploads an image.
The first model predicts:
Whale, dolphin, or porpoise
Not a whale or dolphin
If the image contains a cetacean, the second model predicts the species.
The result and confidence scores are displayed on a web page.
Features
Simple image-upload webpage
Whale or dolphin detection
Species classification
Top three species predictions
Confidence percentages
FastAPI backend
TensorFlow and ResNet50 models
Project Files
whale-identification-api/
├── app.py
├── whale_vs_not.keras
├── whale_vs_not_class_names.json
├── whale_species_model.keras
├── whale_species_class_names.json
├── .gitignore
├── .gitattributes
└── README.md
Installation

Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

Install the required packages:

pip install fastapi uvicorn tensorflow pillow python-multipart
Run the Application

Start the FastAPI server:

python -m uvicorn app:app --reload

Open this address in your browser:

http://127.0.0.1:8000

Upload an image and click Identify Animal.

Models

The application uses two TensorFlow models:

whale_vs_not.keras determines whether an image contains a whale, dolphin, or porpoise.
whale_species_model.keras predicts the most likely cetacean species.

The model files are stored using Git Large File Storage.

Important Note

Machine-learning predictions may occasionally be incorrect, especially when images are distant, blurry, partially obstructed, or show species that were not included in the training dataset.

Technologies
Python
TensorFlow
Keras
ResNet50
FastAPI
HTML
CSS
JavaScript
Git LFS
