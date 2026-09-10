# 🎙️ Synthetic Voice Cloning & Neural TTS Studio

An end-to-end generative AI voice synthesis platform featuring zero-shot voice cloning, real-time Mel-spectrogram processing, and a comprehensive **Responsible AI & Ethical Deployment Framework** compliant with Section 14 standards.

---

## 🌟 Overview & Highlights

* **3-Stage Deep Learning Pipeline:**
  1. **Denoising Autoencoder:** Convolutional filtering ($16 \to 32 \to 64$ channels) to strip noise from raw Mel-spectrograms.
  2. **Speaker VAE (64-D Latent Space):** Disentangles speaker timbre/pitch into a continuous Gaussian latent vector $\mathbf{z}$ with reparameterization sampling ($\tau$).
  3. **Transformer TTS:** Character-level Multi-Head Self-Attention and Cross-Attention generator conditioned on the extracted speaker vector $\mathbf{z}$.
* **Interactive Web GUI:** Modern dark-mode studio featuring live HTML5 Canvas rendering for 80-band Mel-spectrograms, microphone recording, audio playback, and parameter tuning.
* **Alternative Gradio UI:** Pure-Python user interface for quick machine learning experiments.
* **Ethical Governance & Anti-Misuse Framework:**
  * **Dual-Layer Synthetic Watermarking:** Sub-perceptual ~10.2 kHz acoustic carrier tone + cryptographic RIFF header metadata (`AGY-SYNTH-VCTK-VERIFIED`).
  * **Live Deepfake Verifier:** In-browser FFT spectral scanner to detect AI-generated audio with confidence scoring.
  * **Assistive AAC Board:** Restorative communication board designed for individuals with ALS, stroke, or vocal impairments.
  * **Session Rate Limiting:** Anti-abuse protection against automated voice scraping and bot spam.

---

## 🏛️ System Architecture

```
                                  VOICE CLONING PIPELINE
                                  
 [Raw Input Audio]                                              [Input Text]
        │                                                            │
        ▼                                                            ▼
┌─────────────────────────┐                                ┌───────────────────┐
│ MODEL 1:                │                                │ Character         │
│ Denoising Autoencoder   │                                │ Tokenizer (35-dim)│
└─────────────────────────┘                                └─────────┬─────────┘
        │ Clean Mel (80×128)                                         │
        ▼                                                            ▼
┌─────────────────────────┐                                ┌───────────────────┐
│ MODEL 2:                │  64-D Speaker Vector z         │ MODEL 3:          │
│ Speaker VAE             │ ─────────────────────────────► │ Transformer TTS   │
└─────────────────────────┘                                └─────────┬─────────┘
                                                                     │ Predicted Mel
                                                                     ▼
                                                           ┌───────────────────┐
                                                           │ Griffin-Lim +     │
                                                           │ Watermarking      │
                                                           └─────────┬─────────┘
                                                                     │
                                                                     ▼
                                                            [Cloned WAV Audio]
```

---

## 📁 Repository Structure

```
├── app.py                      # FastAPI backend server with full REST APIs
├── models.py                   # PyTorch architectures (AE, VAE, Transformer) & vocoder
├── ethical_safeguards.py       # Watermarking, deepfake verifier, consent & AAC logic
├── gradio_app.py               # Standalone Gradio interface
├── dump_notebook.py            # Utility script to extract notebook cells
├── notebook_extracted_code.py  # Python dump of original training notebook
├── static/
│   ├── index.html              # 5-tab responsive dark-mode Web Studio
│   ├── style.css               # Glassmorphism aesthetic CSS design system
│   ├── app.js                  # Frontend interactivity & Mel Canvas renderer
│   └── samples/                # Demo WAV audio samples (male, female, resonant)
└── Trained/
    └── Trained_models/
        ├── autoencoder_full.pt # Trained weights for Denoising Autoencoder
        ├── speaker_vae_full.pt # Trained weights for Speaker VAE
        ├── transformer.pt      # Trained weights for Transformer TTS
        └── *.ipynb             # Training notebooks on VCTK corpus
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* PyTorch & Torchaudio

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/NexTechANKARA/Synthetic-Voice-Cloning-TTS.git
cd Synthetic-Voice-Cloning-TTS

# Install dependencies
pip install torch torchaudio fastapi uvicorn gradio soundfile scipy numpy
```

### 3. Running the FastAPI Web Studio
```bash
python -m uvicorn app:app --reload --port 8000
```
Open your browser at `http://localhost:8000`.

### 4. Running the Gradio Alternative
```bash
python gradio_app.py
```
Open the provided local URL (typically `http://localhost:7860`).

---

## 🛡️ Responsible AI & Ethical Standards (Section 14)

* **14.1 Informed Consent:** Enforces affirmative authorization before initiating voice synthesis.
* **14.2 Provenance & Verification:** Injects inaudible frequency carriers and cryptographic hashes into every generated WAV to prevent deepfake impersonation. Includes an FFT verifier tool.
* **14.3 Accessibility:** AAC speech board designed for restorative healthcare use (ALS, stroke, laryngectomy).
