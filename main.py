import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

REAL_DIR = "dataset/real"
CLONED_DIR = "dataset/cloned"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "voice_clone_detector.pth")

SAMPLE_RATE = 16000
MAX_DURATION = 4.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_NAME = "facebook/wav2vec2-base-960h"

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(CLONED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Dataset directories ready.")


# ============================================================
# CHECK DATASET
# ============================================================

def get_dataset_files():

    real_files = [
        os.path.join(REAL_DIR, f)
        for f in os.listdir(REAL_DIR)
        if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a"))
    ]

    cloned_files = [
        os.path.join(CLONED_DIR, f)
        for f in os.listdir(CLONED_DIR)
        if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a"))
    ]

    print("\nDataset summary")
    print("-------------------------")
    print(f"Real samples   : {len(real_files)}")
    print(f"Cloned samples : {len(cloned_files)}")
    print("-------------------------")

    if len(real_files) == 0:
        raise RuntimeError(
            f"No real audio files found in {REAL_DIR}"
        )

    if len(cloned_files) == 0:
        raise RuntimeError(
            f"No cloned/fake audio files found in {CLONED_DIR}"
        )

    return real_files, cloned_files


# ============================================================
# AUDIO DATASET
# ============================================================

class AudioDeepfakeDataset(Dataset):

    def __init__(
        self,
        file_paths,
        labels,
        target_sample_rate=SAMPLE_RATE,
        max_duration=MAX_DURATION
    ):

        self.file_paths = file_paths
        self.labels = labels

        self.target_sample_rate = target_sample_rate
        self.max_samples = int(
            target_sample_rate * max_duration
        )

    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, idx):

        path = self.file_paths[idx]

        waveform, sample_rate = torchaudio.load(path)

        # Convert stereo → mono
        if waveform.shape[0] > 1:

            waveform = torch.mean(
                waveform,
                dim=0,
                keepdim=True
            )

        # Resample → 16 kHz
        if sample_rate != self.target_sample_rate:

            waveform = T.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )(waveform)

        waveform = waveform.squeeze(0)

        # Trim / pad to fixed 4 seconds
        if waveform.shape[0] > self.max_samples:

            waveform = waveform[:self.max_samples]

        else:

            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.max_samples - waveform.shape[0])
            )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float
        )

        return waveform, label


# ============================================================
# CLASSIFIER
# ============================================================

class VoiceCloneClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer = nn.Sequential(

            nn.Linear(768, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 1),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.layer(x)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    waveforms,
    processor,
    feature_model
):

    inputs = processor(
        waveforms.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        outputs = feature_model(**inputs)

        features = outputs.last_hidden_state.mean(dim=1)

    return features


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    classifier,
    dataloader,
    processor,
    feature_model
):

    classifier.eval()

    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():

        for waveforms, labels in dataloader:

            features = extract_features(
                waveforms,
                processor,
                feature_model
            )

            scores = classifier(features).squeeze(1)

            predictions = (
                scores >= 0.5
            ).long()

            all_labels.extend(
                labels.long().tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

            all_scores.extend(
                scores.tolist()
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
            all_scores
        )

    except ValueError:

        auc = 0.0

    print("\nEvaluation")
    print("==============================")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("==============================")


# ============================================================
# TRAINING
# ============================================================

def train_model():

    real_files, cloned_files = get_dataset_files()

    # Real = 0
    # Cloned = 1

    file_paths = real_files + cloned_files

    labels = (
        [0] * len(real_files)
        +
        [1] * len(cloned_files)
    )

    # Stratified train/validation split
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    print("\nSplit")
    print("-------------------------")
    print(f"Training   : {len(train_files)}")
    print(f"Validation : {len(val_files)}")
    print("-------------------------")

    train_dataset = AudioDeepfakeDataset(
        train_files,
        train_labels
    )

    val_dataset = AudioDeepfakeDataset(
        val_files,
        val_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Wav2Vec2
    # --------------------------------------------------------

    print("\nLoading Wav2Vec2...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    feature_model = Wav2Vec2Model.from_pretrained(
        MODEL_NAME
    )

    # Freeze Wav2Vec2
    feature_model.eval()

    for param in feature_model.parameters():

        param.requires_grad = False

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    classifier = VoiceCloneClassifier()

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...")

    for epoch in range(EPOCHS):

        classifier.train()

        total_loss = 0

        for batch_idx, (
            waveforms,
            batch_labels
        ) in enumerate(train_loader):

            features = extract_features(
                waveforms,
                processor,
                feature_model
            )

            optimizer.zero_grad()

            predictions = classifier(
                features
            ).squeeze(1)

            loss = criterion(
                predictions,
                batch_labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    evaluate_model(
        classifier,
        val_loader,
        processor,
        feature_model
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    torch.save(
        classifier.state_dict(),
        MODEL_PATH
    )

    print(
        f"\nModel saved successfully:"
        f"\n{MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_directories()

    train_model()
