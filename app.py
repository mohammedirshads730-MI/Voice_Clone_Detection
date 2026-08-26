import os

import streamlit as st
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf

from transformers import (
    AutoProcessor,
    Wav2Vec2Model
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/wav2vec2-base-960h"

MODEL_PATH = "models/voice_clone_detector.pth"

SAMPLE_RATE = 16000

MAX_DURATION = 4.0

MAX_SAMPLES = int(
    SAMPLE_RATE * MAX_DURATION
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Voice Clone & Deepfake Detection",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    .stButton>button {

        width: 100%;

        background-color: #2563eb;

        color: white;

        border-radius: 6px;

        font-weight: 600;

        border: none;

        padding: 0.6rem;

    }

    .stButton>button:hover {

        background-color: #1d4ed8;

    }

    </style>
    """,
    unsafe_allow_html=True
)


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
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    feature_model = Wav2Vec2Model.from_pretrained(
        MODEL_NAME
    )

    feature_model.eval()

    classifier = VoiceCloneClassifier()

    # IMPORTANT:
    # Load the trained classifier weights

    if not os.path.exists(MODEL_PATH):

        return processor, feature_model, None

    classifier.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    classifier.eval()

    return processor, feature_model, classifier


processor, feature_model, classifier = load_models()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Audio Forensics & Deepfake Detection Engine"
)

st.markdown(
    "AI-powered Human vs Synthetic Voice Detection"
)

st.markdown("---")


# ============================================================
# MODEL STATUS
# ============================================================

if classifier is None:

    st.error(
        "⚠️ Trained model not found. "
        "Please train the model using main.py first."
    )

    st.stop()

else:

    st.success(
        "✅ Trained voice detection model loaded."
    )


# ============================================================
# AUDIO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Audio Sample",
    type=[
        "wav",
        "mp3",
        "m4a",
        "flac"
    ]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file
    )

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Size:** "
        f"{uploaded_file.size / 1024:.2f} KB"
    )

    if st.button(
        "🔍 Run Forensic Analysis"
    ):

        with st.spinner(
            "Analyzing acoustic characteristics..."
        ):

            temp_path = "temp_audio.wav"

            with open(
                temp_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            # ------------------------------------------------
            # LOAD AUDIO
            # ------------------------------------------------

            data, sample_rate = sf.read(
                temp_path
            )

            waveform = torch.tensor(
                data,
                dtype=torch.float32
            )

            # ------------------------------------------------
            # MONO
            # ------------------------------------------------

            if waveform.ndim == 1:

                waveform = waveform.unsqueeze(0)

            else:

                waveform = waveform.T

            if waveform.shape[0] > 1:

                waveform = torch.mean(
                    waveform,
                    dim=0,
                    keepdim=True
                )

            # ------------------------------------------------
            # RESAMPLE
            # ------------------------------------------------

            if sample_rate != SAMPLE_RATE:

                waveform = T.Resample(
                    orig_freq=sample_rate,
                    new_freq=SAMPLE_RATE
                )(waveform)

            # ------------------------------------------------
            # FIX LENGTH
            # ------------------------------------------------

            waveform = waveform.squeeze(0)

            if waveform.shape[0] > MAX_SAMPLES:

                waveform = waveform[
                    :MAX_SAMPLES
                ]

            else:

                waveform = torch.nn.functional.pad(
                    waveform,
                    (
                        0,
                        MAX_SAMPLES -
                        waveform.shape[0]
                    )
                )

            # ------------------------------------------------
            # FEATURE EXTRACTION
            # ------------------------------------------------

            with torch.no_grad():

                inputs = processor(
                    waveform.numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True
                )

                outputs = feature_model(
                    **inputs
                )

                features = (
                    outputs
                    .last_hidden_state
                    .mean(dim=1)
                )

                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                prediction = classifier(
                    features
                ).item()

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown(
                "### 🔬 Forensic Assessment"
            )

            if prediction >= 0.5:

                synthetic_confidence = (
                    prediction * 100
                )

                st.error(
                    "🚨 SYNTHETIC / CLONED VOICE"
                )

                st.metric(
                    "Synthetic Voice Score",
                    f"{synthetic_confidence:.2f}%"
                )

            else:

                human_confidence = (
                    (1 - prediction) * 100
                )

                st.success(
                    "✅ AUTHENTIC HUMAN VOICE"
                )

                st.metric(
                    "Human Voice Score",
                    f"{human_confidence:.2f}%"
                )

            # ------------------------------------------------
            # RAW MODEL SCORE
            # ------------------------------------------------

            with st.expander(
                "Technical Details"
            ):

                st.write(
                    f"Model score: "
                    f"{prediction:.6f}"
                )

                st.write(
                    "Decision threshold: 0.50"
                )

                st.write(
                    "Feature extractor: Wav2Vec2"
                )

                st.write(
                    "Analysis window: 4 seconds"
                )

            # ------------------------------------------------
            # CLEANUP
            # ------------------------------------------------

            if os.path.exists(temp_path):

                os.remove(temp_path)
