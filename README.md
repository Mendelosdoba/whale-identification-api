Whale Identification App

A computer-vision app that analyzes an uploaded image.

It first checks whether the image contains a whale, dolphin, or porpoise. If it does, a second model tries to identify the species.

Features
Upload an image through a simple webpage
Detect whether the image contains a cetacean
Predict the most likely species
Show confidence percentages

Technologies:
Python
TensorFlow
Keras
ResNet50
FastAPI
Run the App

Install the required packages:

pip install fastapi uvicorn tensorflow pillow python-multipart

Start the application:

python -m uvicorn app:app --reload

Then open:

http://127.0.0.1:8000

Upload an image and click Identify Animal.

Note

The predictions may not always be correct, especially when an image is blurry, distant, or shows an animal that was not included in the training data.
