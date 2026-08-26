import torch
import torchaudio
import torch.nn as nn
from transformers import AutoProcessor, Wav2Vec2Model
import gradio as gr

# Load processor and base model
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# Define Classifier Architecture
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

classifier = VoiceCloneClassifier()
classifier.eval()

def predict_audio(file_path):
    if not file_path:
        return "Please upload or record an audio file first!"
        
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    input_values = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values
    
    with torch.no_grad():
        outputs = model(input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        score = classifier(pooled).item()
    
    if score > 0.5:
        confidence = score * 100
        return f" !!FAKE / Cloned — {confidence:.1f}% confidence"
    else:
        confidence = (1 - score) * 100
        return f" REAL (Bonafide) — {confidence:.1f}% confidence"

demo = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs="text",
    title="AI Voice Clone Detection System (SIH Prototype)",
    description="Upload a voice sample or record directly using your microphone to check if it's real or synthetically cloned."
)

if __name__ == "__main__":
    demo.launch()
