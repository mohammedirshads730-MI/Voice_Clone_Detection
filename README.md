# Audio Deepfake & Voice Clone Detection System 
*Smart India Hackathon (SIH) Prototype*

An end-to-end deep learning framework designed to detect synthetically cloned voices and prevent audio-based social engineering and financial fraud.
## System Architecture
[Raw Audio / Mic] ➔ [16kHz Resampling] ➔ [Wav2Vec2-Base Encoder] ➔ [Mean Pooling (768D)] ➔ [Custom Linear Classifier] ➔ [REAL / FAKE Output]
