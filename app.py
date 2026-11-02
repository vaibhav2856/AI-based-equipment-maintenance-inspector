from __future__ import annotations

import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Equipment Image Classifier",
    page_icon="🔍",
    layout="wide",
)

APP_DIR = Path(r"C:\Users\dilip\Downloads\vaibhav2856\equipment dataset\train").resolve().parent
IMAGE_SIZE = (64, 64)  # Kept consistent with the supplied training code.
BATCH_SIZE_DEFAULT = 16
MODEL_FILE = APP_DIR / "equipment_model.keras"
CLASS_FILE = APP_DIR / "classes.json"

# Compatibility with the model filename used in the original script.
MODEL_CANDIDATES = [
    MODEL_FILE,
    APP_DIR / "CAT.keras",
    APP_DIR / "cat.keras",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -----------------------------------------------------------------------------
# Model and class-name helpers
# -----------------------------------------------------------------------------
def find_model_file() -> Path | None:
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str, modified_time_ns: int) -> tf.keras.Model:
    """Load a model once and refresh the cache when its file changes."""
    del modified_time_ns  # Used as part of Streamlit's cache key.
    return tf.keras.models.load_model(model_path)


def read_class_names(output_count: int) -> list[str]:
    if CLASS_FILE.is_file():
        try:
            with CLASS_FILE.open("r", encoding="utf-8") as file:
                names = json.load(file)
            if isinstance(names, list) and len(names) == output_count:
                return [str(name) for name in names]
            st.warning(
                "The number of names in classes.json does not match the model output. "
                "Generic class names will be shown."
            )
        except (OSError, json.JSONDecodeError) as error:
            st.warning(f"Could not read classes.json: {error}")

    return [f"Class {index + 1}" for index in range(output_count)]


def get_loaded_model() -> tuple[tf.keras.Model | None, list[str], str | None]:
    model_path = find_model_file()
    if model_path is None:
        return None, [], None

    try:
        model = load_model_cached(str(model_path), model_path.stat().st_mtime_ns)
        output_count = int(model.output_shape[-1])
        class_names = read_class_names(output_count)
        return model, class_names, model_path.name
    except Exception as error:
        return None, [], f"Could not load {model_path.name}: {error}"


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def contains_image_files(folder: Path) -> bool:
    try:
        return any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in folder.iterdir()
        )
    except OSError:
        return False


def find_dataset_root(extracted_folder: Path) -> Path:
    """Find a folder whose immediate subfolders are image classes."""
    candidates = [extracted_folder]
    candidates.extend(path for path in extracted_folder.rglob("*") if path.is_dir())

    for candidate in candidates:
        try:
            class_folders = [
                folder
                for folder in candidate.iterdir()
                if folder.is_dir() and not folder.name.startswith(".")
                and contains_image_files(folder)
            ]
        except OSError:
            continue

        if len(class_folders) >= 2:
            return candidate

    raise ValueError(
        "Could not find the dataset folder. The ZIP must contain at least two "
        "class folders, with image files directly inside each class folder."
    )


def extract_dataset_zip(zip_bytes: bytes, destination: Path) -> Path:
    """Safely extract an uploaded ZIP and identify its class-folder directory."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            for member in archive.infolist():
                relative_path = PurePosixPath(member.filename)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError("The ZIP contains an unsafe file path.")
                if not relative_path.parts:
                    continue

                target = destination.joinpath(*relative_path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                # Copy file contents instead of using extractall().
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as error:
        raise ValueError("The uploaded file is not a valid ZIP archive.") from error

    return find_dataset_root(destination)


def validate_dataset(dataset_dir: Path) -> list[str]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(f"Dataset folder does not exist: {dataset_dir}")

    class_folders = sorted(
        folder
        for folder in dataset_dir.iterdir()
        if folder.is_dir() and not folder.name.startswith(".")
        and contains_image_files(folder)
    )
    if len(class_folders) < 2:
        raise ValueError(
            "The dataset must contain at least two class folders, such as "
            "train/Normal/ and train/Cracks/."
        )

    too_small = [
        folder.name
        for folder in class_folders
        if sum(
            1 for item in folder.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ) < 2
    ]
    if too_small:
        raise ValueError(
            "Each class needs at least two image files. Too few images in: "
            + ", ".join(too_small)
        )

    return [folder.name for folder in class_folders]


# -----------------------------------------------------------------------------
# CNN training
# -----------------------------------------------------------------------------
def build_cnn(number_of_classes: int) -> tf.keras.Model:
    """Build a CNN with an output neuron for every dataset class."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(number_of_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class StreamlitTrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar: Any, status_text: Any, total_epochs: int):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        self.progress_bar.progress(min((epoch + 1) / self.total_epochs, 1.0))
        accuracy = logs.get("accuracy")
        val_accuracy = logs.get("val_accuracy")
        message = f"Epoch {epoch + 1}/{self.total_epochs}"
        if accuracy is not None:
            message += f" · Training accuracy: {accuracy:.1%}"
        if val_accuracy is not None:
            message += f" · Validation accuracy: {val_accuracy:.1%}"
        self.status_text.text(message)


def train_model(
    dataset_dir: Path,
    epochs: int,
    batch_size: int,
    progress_bar: Any,
    status_text: Any,
) -> tuple[tf.keras.Model, list[str], dict[str, list[float]]]:
    class_names = validate_dataset(dataset_dir)

    train_data = tf.keras.utils.image_dataset_from_directory(
        str(dataset_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=42,
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        str(dataset_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=42,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_data = train_data.prefetch(buffer_size=autotune)
    validation_data = validation_data.prefetch(buffer_size=autotune)

    model = build_cnn(len(class_names))
    callbacks = [
        StreamlitTrainingProgress(progress_bar, status_text, epochs),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=0.00001
        ),
    ]

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )

    # Save the model and the class names together in the app directory.
    model.save(MODEL_FILE)
    temporary_class_file = CLASS_FILE.with_suffix(".json.tmp")
    with temporary_class_file.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    temporary_class_file.replace(CLASS_FILE)

    load_model_cached.clear()
    return model, class_names, history.history


# -----------------------------------------------------------------------------
# Image prediction
# -----------------------------------------------------------------------------
def prepare_image(uploaded_image: Any) -> Image.Image:
    image = Image.open(uploaded_image)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def predict_image(
    model: tf.keras.Model, image: Image.Image, class_names: list[str]
) -> tuple[str, float, dict[str, float]]:
    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    prediction = np.asarray(model.predict(image_batch, verbose=0))
    if prediction.ndim != 2 or prediction.shape[0] != 1:
        raise ValueError("The model returned an unexpected prediction format.")
    if prediction.shape[1] != len(class_names):
        raise ValueError(
            "The model output does not match the class names. Check classes.json."
        )

    probabilities = prediction[0]
    best_index = int(np.argmax(probabilities))
    scores = {
        class_names[index]: float(probabilities[index])
        for index in range(len(class_names))
    }
    return class_names[best_index], float(probabilities[best_index]), scores


# -----------------------------------------------------------------------------
# Streamlit interface
# -----------------------------------------------------------------------------
st.title("🔍 Equipment Image Classifier")
st.write(
    "Upload a photograph of an equipment surface to classify its visible condition "
    "using your trained CNN model."
)

classify_tab, train_tab = st.tabs(["Classify an image", "Train the model"])

with classify_tab:
    model, class_names, model_status = get_loaded_model()

    if model is None:
        st.warning(
            "No trained model was found. Train a model in the **Train the model** "
            "tab, or place your existing CAT.keras and classes.json files beside app.py."
        )
        if model_status:
            st.error(model_status)
    else:
        st.caption(f"Loaded model: {model_status}")
        uploaded_image = st.file_uploader(
            "Choose an equipment image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="prediction_image",
        )

        if uploaded_image is not None:
            try:
                image = prepare_image(uploaded_image)
                left, right = st.columns([1, 1])
                with left:
                    st.image(image, caption="Uploaded image", use_container_width=True)

                with st.spinner("Analysing image..."):
                    predicted_class, confidence, scores = predict_image(
                        model, image, class_names
                    )

                with right:
                    st.subheader("Prediction")
                    st.success(predicted_class)
                    st.metric("Model confidence", f"{confidence:.2%}")
                    st.caption(
                        "Confidence is the model's predicted score, not a guarantee "
                        "that the classification is correct."
                    )

                st.subheader("Class-wise scores")
                st.bar_chart(
                    {"Class": list(scores.keys()), "Score": list(scores.values())},
                    x="Class",
                    y="Score",
                    x_label="Class",
                    y_label="Predicted score",
                )
            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"Could not process this image: {error}")
            except Exception as error:
                st.error(f"Prediction failed: {error}")

with train_tab:
    st.write(
        "Train a CNN using a folder of images. Each class must have its own folder. "
        "For example, `train/Normal/`, `train/Cracks/`, and `train/Scratches/`."
    )

    dataset_source = st.radio(
        "Where is your dataset?",
        ["Dataset folder on this computer/server", "Upload dataset as a ZIP file"],
        index=0,
    )

    dataset_path_text = None
    dataset_zip = None
    if dataset_source == "Dataset folder on this computer/server":
        dataset_path_text = st.text_input(
            "Dataset folder path",
            value=str(APP_DIR / "equipment dataset" / "train"),
            help="This folder must contain one subfolder per image class.",
        )
    else:
        dataset_zip = st.file_uploader(
            "Upload a ZIP containing the class folders",
            type=["zip"],
            key="training_dataset_zip",
        )

    epochs = st.slider("Maximum training epochs", min_value=5, max_value=50, value=20)
    batch_size = st.selectbox("Batch size", options=[8, 16, 32], index=1)

    st.info(
        "The app automatically detects the number of classes. It reserves 20% of "
        "the images for validation and may stop early when validation performance "
        "does not improve."
    )

    if st.button("Train and save model", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        try:
            if dataset_source == "Dataset folder on this computer/server":
                if not dataset_path_text or not dataset_path_text.strip():
                    raise ValueError("Enter the dataset folder path.")
                dataset_dir = Path(dataset_path_text.strip()).expanduser()
                if not dataset_dir.is_absolute():
                    dataset_dir = APP_DIR / dataset_dir

                with st.spinner("Training the model. This may take some time..."):
                    trained_model, trained_classes, history = train_model(
                        dataset_dir,
                        epochs,
                        batch_size,
                        progress_bar,
                        status_text,
                    )
            else:
                if dataset_zip is None:
                    raise ValueError("Upload your dataset ZIP file first.")
                with tempfile.TemporaryDirectory(prefix="equipment_dataset_") as temp:
                    extracted_root = Path(temp)
                    dataset_dir = extract_dataset_zip(
                        dataset_zip.getvalue(), extracted_root / "dataset"
                    )
                    with st.spinner("Training the model. This may take some time..."):
                        trained_model, trained_classes, history = train_model(
                            dataset_dir,
                            epochs,
                            batch_size,
                            progress_bar,
                            status_text,
                        )

            final_accuracy = history.get("val_accuracy", [0.0])[-1]
            st.success(f"Training complete. Validation accuracy: {final_accuracy:.2%}")
            st.write("Classes detected:", ", ".join(trained_classes))
            st.write(f"Model saved as `{MODEL_FILE.name}` and class names as `{CLASS_FILE.name}`.")
            if history.get("accuracy"):
                st.line_chart(
                    {
                        "Training accuracy": history["accuracy"],
                        "Validation accuracy": history.get("val_accuracy", []),
                    }
                )
            st.info("Open the **Classify an image** tab to test the newly trained model.")
        except Exception as error:
            st.error(f"Training failed: {error}")
            st.caption(
                "Check that the dataset has at least two class folders, that each "
                "folder contains images, and that the available memory is sufficient."
            )

st.divider()
st.caption(
    "Image size: 64 × 64 pixels · Model: Convolutional Neural Network (CNN) · "
    "The result is an automated screening aid and should be checked by a human inspector."
)



