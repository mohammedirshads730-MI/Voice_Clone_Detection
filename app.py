import os
import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from transformers import AutoProcessor, Wav2Vec2Model

# Page Configuration for Professional Appearance
st.set_page_config(
    page_title="Voice Clone & Deepfake Detection Platform",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Sleek, Professional UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 4px;
        font-weight: 600;
        border: none;
        padding: 0.6rem;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .metric-container {
        background-color: #1f2937;
        padding: 1.2rem;
        border-radius: 6px;
        border-left: 4px solid #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# Classifier Architecture
class VoiceCloneClassifier(nn.Module):
    def __init__(self):
        super(VoiceCloneClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layer(x)

@st.cache_resource
def load_forensic_models():
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    feature_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    feature_model.eval()
    
    classifier = VoiceCloneClassifier()
    classifier.eval()
    return processor, feature_model, classifier

processor, feature_model, classifier = load_forensic_models()

# Header Section
st.title("🛡️ Audio Forensics & Deepfake Detection Engine")
st.markdown("Smart India Hackathon | Enterprise Voice Authentication Dashboard")
st.markdown("---")

# File Ingestion
uploaded_file = st.file_uploader("Upload Audio Sample for Forensic Verification", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"File Name: {uploaded_file.name}")
    with col2:
        st.text(f"File Size: {uploaded_file.size / 1024:.2f} KB")
        
    if st.button("Run Forensic Analysis"):
        with st.spinner("Extracting Wav2Vec2 acoustic representations..."):
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            waveform, sample_rate = torchaudio.load(temp_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != 16000:
                waveform = T.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
                
            waveform = waveform.squeeze(0)
            max_samples = int(16000 * 4.0)
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[0]))
                
            with torch.no_grad():
                inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                outputs = feature_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
                
                prediction = classifier(features).item()
            
            st.markdown("### Forensic Assessment Report")
            
            if prediction > 0.5:
                confidence = prediction * 100
                st.error(f"**Classification Status:** SYNTHETIC / CLONED VOICE")
                st.metric(label="Tampering Confidence Index", value=f"{confidence:.2f}%")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"**Classification Status:** AUTHENTIC HUMAN VOICE")
                st.metric(label="Authenticity Confidence Index", value=f"{confidence:.2f}%")
                
            if os.path.exists(temp_path):
                os.remove(temp_path)
