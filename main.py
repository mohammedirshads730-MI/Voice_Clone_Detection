import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoProcessor, Wav2Vec2Model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MODEL_NAME = "facebook/wav2vec2-base-960h"

TRAIN_REAL_DIR = "dataset/real"
TRAIN_FAKE_DIR = "dataset/cloned"

VAL_REAL_DIR = "dataset/validation/real"
VAL_FAKE_DIR = "dataset/validation/cloned"

TEST_REAL_DIR = "dataset/test/real"
TEST_FAKE_DIR = "dataset/test/cloned"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "voice_clone_detector.pth")

FEATURE_DIR = "features_cache"

TARGET_SAMPLE_RATE = 16000
MAX_DURATION = 4.0
MAX_SAMPLES = int(TARGET_SAMPLE_RATE * MAX_DURATION)

FEATURE_BATCH_SIZE = 8
CLASSIFIER_BATCH_SIZE = 64

EPOCHS = 15
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("VOICE CLONE / DEEPFAKE DETECTION TRAINING")
print("=" * 60)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)


# ============================================================
# FILE COLLECTION
# ============================================================

def get_audio_files(folder):

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".wav", ".flac", ".mp3", ".m4a")
        ):
            files.append(
                os.path.join(folder, filename)
            )

    files.sort()

    return files


def build_split(real_dir, fake_dir):

    real_files = get_audio_files(real_dir)
    fake_files = get_audio_files(fake_dir)

    files = real_files + fake_files

    labels = (
        [0] * len(real_files) +
        [1] * len(fake_files)
    )

    return files, labels


# ============================================================
# DATASET SUMMARY
# ============================================================

train_files, train_labels = build_split(
    TRAIN_REAL_DIR,
    TRAIN_FAKE_DIR
)

val_files, val_labels = build_split(
    VAL_REAL_DIR,
    VAL_FAKE_DIR
)

test_files, test_labels = build_split(
    TEST_REAL_DIR,
    TEST_FAKE_DIR
)

print("\nDATASET SUMMARY")
print("-" * 60)

print(
    f"Training   : {len(train_files)} "
    f"({sum(1 for x in train_labels if x == 0)} real, "
    f"{sum(1 for x in train_labels if x == 1)} cloned)"
)

print(
    f"Validation : {len(val_files)} "
    f"({sum(1 for x in val_labels if x == 0)} real, "
    f"{sum(1 for x in val_labels if x == 1)} cloned)"
)

print(
    f"Test       : {len(test_files)} "
    f"({sum(1 for x in test_labels if x == 0)} real, "
    f"{sum(1 for x in test_labels if x == 1)} cloned)"
)


# ============================================================
# AUDIO LOADING / PREPROCESSING
# ============================================================

def load_audio(path):

    waveform, sample_rate = torchaudio.load(path)

    # Convert stereo/multi-channel → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample → 16 kHz
    if sample_rate != TARGET_SAMPLE_RATE:

        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE
        )(waveform)

    waveform = waveform.squeeze(0)

    # Limit to 4 seconds
    if waveform.shape[0] > MAX_SAMPLES:

        waveform = waveform[:MAX_SAMPLES]

    # Zero-pad shorter audio
    elif waveform.shape[0] < MAX_SAMPLES:

        waveform = torch.nn.functional.pad(
            waveform,
            (0, MAX_SAMPLES - waveform.shape[0])
        )

    return waveform


# ============================================================
# WAV2VEC2 FEATURE EXTRACTION
# ============================================================

def extract_features(
    files,
    labels,
    processor,
    feature_model,
    split_name
):

    cache_path = os.path.join(
        FEATURE_DIR,
        f"{split_name}_features.pt"
    )

    # --------------------------------------------------------
    # Load cached features if available
    # --------------------------------------------------------

    if os.path.exists(cache_path):

        print(
            f"\nLoading cached {split_name} features..."
        )

        cached = torch.load(
            cache_path,
            map_location="cpu"
        )

        return cached["features"], cached["labels"]

    print(
        f"\nExtracting Wav2Vec2 features for "
        f"{split_name}..."
    )

    all_features = []

    feature_model.eval()

    for start in range(
        0,
        len(files),
        FEATURE_BATCH_SIZE
    ):

        batch_files = files[
            start:start + FEATURE_BATCH_SIZE
        ]

        waveforms = []

        for path in batch_files:

            try:

                waveform = load_audio(path)

                waveforms.append(
                    waveform.numpy()
                )

            except Exception as e:

                print(
                    f"\nSkipping corrupted file: {path}"
                )

                print("Error:", e)

        if not waveforms:
            continue

        # Processor
        inputs = processor(
            waveforms,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to GPU
        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # Wav2Vec2 inference
        with torch.inference_mode():

            outputs = feature_model(
                **inputs
            )

            # Mean pooling over time
            features = (
                outputs.last_hidden_state
                .mean(dim=1)
            )

        # Move features back to CPU
        all_features.append(
            features.cpu()
        )

        processed = min(
            start + FEATURE_BATCH_SIZE,
            len(files)
        )

        print(
            f"\r{split_name}: "
            f"{processed}/{len(files)} files",
            end=""
        )

    print()

    features = torch.cat(
        all_features,
        dim=0
    )

    labels_tensor = torch.tensor(
        labels[:len(features)],
        dtype=torch.float32
    )

    # Save cache
    torch.save(
        {
            "features": features,
            "labels": labels_tensor
        },
        cache_path
    )

    print(
        f"Saved cached features → {cache_path}"
    )

    print(
        f"Feature shape: {features.shape}"
    )

    return features, labels_tensor


# ============================================================
# CLASSIFIER
# ============================================================

class VoiceCloneClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        return self.network(x).squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    model,
    features,
    labels
):

    model.eval()

    dataset = TensorDataset(
        features,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=CLASSIFIER_BATCH_SIZE,
        shuffle=False
    )

    all_probabilities = []
    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for batch_features, batch_labels in loader:

            batch_features = batch_features.to(
                DEVICE
            )

            logits = model(
                batch_features
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).long()

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch_labels.numpy()
            )

    all_labels = np.array(
        all_labels
    ).astype(int)

    all_predictions = np.array(
        all_predictions
    ).astype(int)

    all_probabilities = np.array(
        all_probabilities
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    except ValueError:

        auc = 0.0

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(
    train_features,
    train_labels,
    val_features,
    val_labels
):

    print("\n" + "=" * 60)
    print("TRAINING CLASSIFIER")
    print("=" * 60)

    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CLASSIFIER_BATCH_SIZE,
        shuffle=True
    )

    classifier = VoiceCloneClassifier().to(
        DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE
    )

    best_val_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):

        classifier.train()

        total_loss = 0.0

        for features, labels in train_loader:

            features = features.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = classifier(
                features
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss /
            len(train_loader)
        )

        (
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            val_auc,
            _
        ) = calculate_metrics(
            classifier,
            val_features,
            val_labels
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Loss      : {avg_loss:.4f}"
        )

        print(
            f"Val Acc   : {val_accuracy:.4f}"
        )

        print(
            f"Val Prec  : {val_precision:.4f}"
        )

        print(
            f"Val Recall: {val_recall:.4f}"
        )

        print(
            f"Val F1    : {val_f1:.4f}"
        )

        print(
            f"Val AUC   : {val_auc:.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                {
                    "model_state_dict":
                        classifier.state_dict(),

                    "model_type":
                        "VoiceCloneClassifier",

                    "feature_dimension":
                        768,

                    "sample_rate":
                        TARGET_SAMPLE_RATE,

                    "max_duration":
                        MAX_DURATION,

                    "wav2vec_model":
                        MODEL_NAME,

                    "best_val_f1":
                        best_val_f1
                },
                MODEL_PATH
            )

            print(
                f"✅ Best model saved → {MODEL_PATH}"
            )

    return classifier


# ============================================================
# MAIN
# ============================================================

def train_model():

    print("\nLoading Wav2Vec2 processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Loading Wav2Vec2 model...")

    feature_model = Wav2Vec2Model.from_pretrained(
        MODEL_NAME
    )

    feature_model = feature_model.to(
        DEVICE
    )

    feature_model.eval()

    # Freeze Wav2Vec2
    for parameter in feature_model.parameters():

        parameter.requires_grad = False

    print("Wav2Vec2 ready.")

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    train_features, train_labels_tensor = (
        extract_features(
            train_files,
            train_labels,
            processor,
            feature_model,
            "train"
        )
    )

    val_features, val_labels_tensor = (
        extract_features(
            val_files,
            val_labels,
            processor,
            feature_model,
            "validation"
        )
    )

    test_features, test_labels_tensor = (
        extract_features(
            test_files,
            test_labels,
            processor,
            feature_model,
            "test"
        )
    )

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        "Train features:",
        train_features.shape
    )

    print(
        "Validation features:",
        val_features.shape
    )

    print(
        "Test features:",
        test_features.shape
    )

    # --------------------------------------------------------
    # TRAIN CLASSIFIER
    # --------------------------------------------------------

    train_classifier(
        train_features,
        train_labels_tensor,
        val_features,
        val_labels_tensor
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print("\nLoading best model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    best_model = VoiceCloneClassifier().to(
        DEVICE
    )

    best_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm
    ) = calculate_metrics(
        best_model,
        test_features,
        test_labels_tensor
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\n" + "=" * 60)

    print(
        f"✅ FINAL MODEL SAVED:"
    )

    print(
        MODEL_PATH
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_model()
