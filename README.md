Whale & Dolphin Species Identification App

A computer vision application that analyzes an uploaded image and identifies whether it contains a whale, dolphin, or porpoise.

The application uses a two-stage prediction system. The first model determines whether the uploaded image contains a cetacean. If a cetacean is detected, the second model predicts the most likely species.

Features:

Upload an image through a simple web interface.

Detect whether the image contains a whale, dolphin, or porpoise.

Reject images that do not appear to contain a cetacean.

Predict the most likely cetacean species.

Display prediction confidence scores.

Technologies:
Python
TensorFlow / Keras
ResNet50
FastAPI
Pillow
HTML and CSS
How It Works?

The application uses two convolutional neural network models:

Cetacean detection model:

Classifies the uploaded image as cetacean or non-cetacean.

Species identification model:

Identifies the most likely whale, dolphin, or porpoise species.

Running the Application
1. Install the required packages
pip install fastapi uvicorn tensorflow pillow python-multipart
2. Start the application
python -m uvicorn app:app --reload
3. Open the application

Open the following address in your web browser:

http://127.0.0.1:8000

Upload an image and click Identify Animal to receive a prediction and confidence score.

Limitations

Predictions may be less accurate when images are blurry, distant, partially obstructed, or taken from unusual angles.

This project was developed for educational and portfolio purposes and should not be used as a substitute for professional wildlife identification.
