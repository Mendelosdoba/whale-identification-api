from io import BytesIO
from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet50_preprocess_input
)


# ============================================================
# 1. FILE NAMES
# ============================================================

BASE_FOLDER = Path(__file__).resolve().parent

CETACEAN_MODEL_PATH = (
    BASE_FOLDER / "whale_vs_not.keras"
)

CETACEAN_CLASSES_PATH = (
    BASE_FOLDER / "whale_vs_not_class_names.json"
)

SPECIES_MODEL_PATH = (
    BASE_FOLDER / "whale_species_model.keras"
)

SPECIES_CLASSES_PATH = (
    BASE_FOLDER / "whale_species_class_names.json"
)


IMAGE_SIZE = (224, 224)

# Change later after testing if needed
CETACEAN_THRESHOLD = 0.50
SPECIES_THRESHOLD = 0.45


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

required_files = [
    CETACEAN_MODEL_PATH,
    CETACEAN_CLASSES_PATH,
    SPECIES_MODEL_PATH,
    SPECIES_CLASSES_PATH
]

missing_files = [
    file_path.name
    for file_path in required_files
    if not file_path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "\nMissing required files:\n"
        + "\n".join(missing_files)
        + "\n\nAll four files must be in the same folder "
          "as app.py."
    )


# ============================================================
# 3. FIX FOR SAVED LAMBDA PREPROCESSING LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="WhaleAPI",
    name="preprocess_input"
)
def preprocess_input(image):
    """
    Function required when loading models that contain a saved
    Lambda layer named preprocess_input.
    """

    return resnet50_preprocess_input(image)


CUSTOM_OBJECTS = {
    "preprocess_input": preprocess_input
}

tf.keras.utils.get_custom_objects().update(
    CUSTOM_OBJECTS
)


# ============================================================
# 4. LOAD MODELS
# ============================================================

print("\nLoading whale-vs-not model...")

with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
    cetacean_model = tf.keras.models.load_model(
        CETACEAN_MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
        safe_mode=False
    )

print("Whale-vs-not model loaded successfully.")


print("\nLoading whale species model...")

with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
    species_model = tf.keras.models.load_model(
        SPECIES_MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
        safe_mode=False
    )

print("Whale species model loaded successfully.")


# ============================================================
# 5. LOAD CLASS NAMES
# ============================================================

with open(
    CETACEAN_CLASSES_PATH,
    "r",
    encoding="utf-8"
) as file:
    cetacean_class_names = json.load(file)


with open(
    SPECIES_CLASSES_PATH,
    "r",
    encoding="utf-8"
) as file:
    species_class_names = json.load(file)


if not isinstance(cetacean_class_names, list):
    raise ValueError(
        "whale_vs_not_class_names.json must contain a list."
    )

if len(cetacean_class_names) != 2:
    raise ValueError(
        "whale_vs_not_class_names.json must contain "
        "exactly two class names."
    )

if not isinstance(species_class_names, list):
    raise ValueError(
        "whale_species_class_names.json must contain a list."
    )

if len(species_class_names) == 0:
    raise ValueError(
        "whale_species_class_names.json is empty."
    )


print("\nWhale-vs-not classes:")
print(cetacean_class_names)

print("\nSpecies classes:")
print(species_class_names)


# ============================================================
# 6. FIND WHICH CLASS MEANS CETACEAN
# ============================================================

def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


cetacean_class_name = None

accepted_cetacean_names = {
    "cetacean",
    "whale",
    "whales",
    "whale_dolphin",
    "whale_or_dolphin",
    "whale_and_dolphin"
}


for class_name in cetacean_class_names:

    normalized_name = normalize_name(
        class_name
    )

    if normalized_name in accepted_cetacean_names:
        cetacean_class_name = class_name
        break


if cetacean_class_name is None:
    raise ValueError(
        "\nCould not determine which class means cetacean.\n"
        f"Classes found: {cetacean_class_names}\n\n"
        "The positive folder should be named 'cetacean'."
    )


not_cetacean_class_name = next(
    class_name
    for class_name in cetacean_class_names
    if class_name != cetacean_class_name
)


print("\nCetacean class:")
print(cetacean_class_name)

print("\nNot-cetacean class:")
print(not_cetacean_class_name)


# ============================================================
# 7. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Whale Identification API",
    description=(
        "Checks whether an image contains a whale, dolphin, "
        "or porpoise. If it does, the API predicts its species."
    ),
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# 8. PREPARE IMAGE
# ============================================================

def prepare_image(
    image_bytes: bytes
) -> np.ndarray:

    try:
        image = Image.open(
            BytesIO(image_bytes)
        )

        image = image.convert("RGB")

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        ) from error


    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )


    image_array = np.asarray(
        image,
        dtype=np.float32
    )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Do not call preprocess_input here.
    # The saved models already contain their preprocessing.
    return image_array


# ============================================================
# 9. WHALE/DOLPHIN VS NOT-WHALE PREDICTION
# ============================================================

def predict_cetacean(
    image_array: np.ndarray
) -> dict:

    raw_prediction = cetacean_model.predict(
        image_array,
        verbose=0
    )


    prediction = np.asarray(
        raw_prediction
    ).flatten()


    # Binary sigmoid model
    if len(prediction) == 1:

        class_1_probability = float(
            prediction[0]
        )

        class_probabilities = {
            cetacean_class_names[0]:
                1.0 - class_1_probability,

            cetacean_class_names[1]:
                class_1_probability
        }


    # Two-output softmax model
    elif len(prediction) == 2:

        class_probabilities = {
            cetacean_class_names[0]:
                float(prediction[0]),

            cetacean_class_names[1]:
                float(prediction[1])
        }


    else:
        raise RuntimeError(
            "The whale-vs-not model returned an unexpected "
            f"number of outputs: {len(prediction)}"
        )


    cetacean_probability = float(
        class_probabilities[
            cetacean_class_name
        ]
    )


    not_cetacean_probability = float(
        class_probabilities[
            not_cetacean_class_name
        ]
    )


    is_cetacean = (
        cetacean_probability >= CETACEAN_THRESHOLD
        and
        cetacean_probability >= not_cetacean_probability
    )


    return {
        "is_cetacean": is_cetacean,
        "cetacean_probability":
            cetacean_probability,
        "not_cetacean_probability":
            not_cetacean_probability
    }


# ============================================================
# 10. SPECIES PREDICTION
# ============================================================

def predict_species(
    image_array: np.ndarray
) -> dict:

    raw_prediction = species_model.predict(
        image_array,
        verbose=0
    )


    probabilities = np.asarray(
        raw_prediction
    ).flatten()


    if len(probabilities) != len(species_class_names):
        raise RuntimeError(
            "\nSpecies model output does not match the "
            "species class-names file.\n"
            f"Model outputs: {len(probabilities)}\n"
            f"Class names: {len(species_class_names)}"
        )


    best_index = int(
        np.argmax(probabilities)
    )


    best_species = species_class_names[
        best_index
    ]


    best_confidence = float(
        probabilities[best_index]
    )


    top_count = min(
        3,
        len(species_class_names)
    )


    top_indices = np.argsort(
        probabilities
    )[-top_count:][::-1]


    top_predictions = []


    for index in top_indices:

        index = int(index)

        confidence = float(
            probabilities[index]
        )

        readable_name = (
            species_class_names[index]
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

        top_predictions.append(
            {
                "species":
                    species_class_names[index],

                "readable_species":
                    readable_name,

                "confidence":
                    round(confidence, 4),

                "confidence_percent":
                    round(confidence * 100, 2)
            }
        )


    return {
        "species": best_species,
        "confidence": best_confidence,
        "top_predictions": top_predictions
    }


# ============================================================
# 11. HOME PAGE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Whale Identification API is running.",
        "instructions": (
            "Open http://127.0.0.1:8000/docs "
            "to upload a picture."
        )
    }


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "whale_vs_not_model_loaded": True,
        "species_model_loaded": True,
        "cetacean_classes":
            cetacean_class_names,
        "species_count":
            len(species_class_names),
        "species":
            species_class_names
    }


# ============================================================
# 13. IMAGE PREDICTION
# ============================================================

@app.post("/predict")
async def predict_whale(
    file: UploadFile = File(...)
):

    allowed_content_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    }


    if (
        file.content_type
        and file.content_type not in allowed_content_types
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a JPG, JPEG, PNG, "
                "or WEBP image."
            )
        )


    image_bytes = await file.read()


    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )


    image_array = prepare_image(
        image_bytes
    )


    # --------------------------------------------------------
    # MODEL 1: WHALE/DOLPHIN OR NOT
    # --------------------------------------------------------

    try:
        cetacean_result = predict_cetacean(
            image_array
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The whale-vs-not model could not "
                f"process the image: {error}"
            )
        ) from error


    cetacean_confidence = cetacean_result[
        "cetacean_probability"
    ]


    if not cetacean_result["is_cetacean"]:

        return {
            "result":
                "not_whale_or_dolphin",

            "message":
                "Not a whale or dolphin.",

            "is_cetacean":
                False,

            "cetacean_confidence":
                round(
                    cetacean_confidence,
                    4
                ),

            "cetacean_confidence_percent":
                round(
                    cetacean_confidence * 100,
                    2
                ),

            "not_cetacean_confidence_percent":
                round(
                    cetacean_result[
                        "not_cetacean_probability"
                    ] * 100,
                    2
                )
        }


    # --------------------------------------------------------
    # MODEL 2: WHICH SPECIES
    # --------------------------------------------------------

    try:
        species_result = predict_species(
            image_array
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The species model could not process "
                f"the image: {error}"
            )
        ) from error


    predicted_species = species_result[
        "species"
    ]


    species_confidence = species_result[
        "confidence"
    ]


    readable_species = (
        predicted_species
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


    if species_confidence >= SPECIES_THRESHOLD:

        result = "species_identified"

        message = (
            f"This appears to be a "
            f"{readable_species}."
        )

    else:

        result = "species_uncertain"

        message = (
            "This appears to be a whale or dolphin, "
            "but the species is uncertain. "
            f"The model's best guess is "
            f"{readable_species}."
        )


    return {
        "result":
            result,

        "message":
            message,

        "is_cetacean":
            True,

        "cetacean_confidence":
            round(
                cetacean_confidence,
                4
            ),

        "cetacean_confidence_percent":
            round(
                cetacean_confidence * 100,
                2
            ),

        "predicted_species":
            predicted_species,

        "readable_species":
            readable_species,

        "species_confidence":
            round(
                species_confidence,
                4
            ),

        "species_confidence_percent":
            round(
                species_confidence * 100,
                2
            ),

        "top_species_predictions":
            species_result[
                "top_predictions"
            ]
    }