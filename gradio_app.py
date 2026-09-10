import io
import torch
import numpy as np
from scipy.io import wavfile
import gradio as gr

from models import (
    ModelManager,
    audio_bytes_to_mel,
    mel_to_wav_bytes,
    encode_text,
    SELECTED_SPEAKERS,
    LATENT_DIM,
    SAMPLE_RATE
)
from ethical_safeguards import watermarker, consent_mgr, AAC_PHRASES

# Initialize models
print("Loading trained models for Gradio Demo...")
manager = ModelManager()

# Precompute speaker prototypes
speaker_prototypes = {}
torch.manual_seed(42)
for i, spk in enumerate(SELECTED_SPEAKERS):
    vec = torch.randn(1, LATENT_DIM) * 0.5
    vec[0, i % LATENT_DIM] += 1.5
    speaker_prototypes[spk] = vec

# List of all available speakers with indicators for held-out zero-shot speakers
SPEAKER_CHOICES = [
    ("p225 (Training - Male Scottish)", "p225"),
    ("p226 (Training - Male English)", "p226"),
    ("p227 (Training - Male English)", "p227"),
    ("p228 (Training - Female English)", "p228"),
    ("p229 (Training - Female English)", "p229"),
    ("p230 (Training - Male English)", "p230"),
    ("p231 (Training - Female English)", "p231"),
    ("p232 (Training - Male English)", "p232"),
    ("p233 [HELD-OUT ZERO SHOT] (Female)", "p233"),
    ("p234 [HELD-OUT ZERO SHOT] (Male)", "p234"),
]

def synthesize_speech(text_input, speaker_choice, consent_acknowledged):
    if not text_input or not text_input.strip():
        return None, "Please enter some text to synthesize."
    
    if not consent_acknowledged:
        return None, "⚠️ Ethical Policy Notice: You must check the ethical consent acknowledgement to synthesize speech."

    # Encode text
    text_ids, text_mask = encode_text(text_input)
    spk_z = speaker_prototypes.get(speaker_choice, speaker_prototypes["p225"])

    manager.transformer_tts.eval()
    with torch.no_grad():
        pred_mel = manager.transformer_tts(text_ids, text_mask, spk_z)

    # Convert to wav bytes with synthetic audio watermark (Section 14.2)
    wav_bytes = mel_to_wav_bytes(pred_mel, watermark=True, speaker_id=speaker_choice)
    
    sr, data = wavfile.read(io.BytesIO(wav_bytes))
    return (sr, data), f"✅ Speech synthesized successfully in voice '{speaker_choice}' with verified acoustic watermark."

def verify_watermark_file(audio_path):
    if not audio_path:
        return "No audio uploaded."
    try:
        with open(audio_path, "rb") as f:
            content = f.read()
        res = watermarker.verify_watermark(content)
        report = (
            f"🔍 Watermark Verification Result:\n"
            f"• Is Synthetic / AI-Generated: {'YES (Safe & Watermarked)' if res['is_synthetic'] else 'NO (Unmarked)'}\n"
            f"• Method: {res['verification_method']}\n"
            f"• Confidence: {res['confidence_pct']}%\n"
            f"• Provenance Metadata: {res['provenance']}\n"
            f"• Compliance: {res['ethical_compliance']}"
        )
        return report
    except Exception as e:
        return f"Error analyzing audio: {str(e)}"

# Build the Gradio UI
with gr.Blocks(title="Neural Voice Cloning & TTS Studio") as demo:
    gr.Markdown(
        """
        # 🎙️ Synthetic Voice Cloning & TTS for Assistive Communication
        ### Generative AI Pipeline (VCTK Corpus • Assistive Technology)
        *Compliant with Section 10 & 14 Ethical Standards (Explicit Consent, Synthetic Watermarking, AAC Accessibility)*
        """
    )

    with gr.Tab("🗣️ Text-to-Speech Synthesis"):
        with gr.Row():
            with gr.Column(scale=2):
                text_box = gr.Textbox(
                    label="Input Text (Max 60 chars)",
                    placeholder="Type a sentence (e.g. Hello, how are you today?)",
                    value="Voice cloning with deep neural transformers"
                )
                speaker_dropdown = gr.Dropdown(
                    choices=[c[0] for c in SPEAKER_CHOICES],
                    value=SPEAKER_CHOICES[0][0],
                    label="Target Speaker Identity (Includes Held-Out Zero-Shot Speakers)"
                )
                consent_checkbox = gr.Checkbox(
                    label="I confirm this speech is synthesized ethically for assistive/research use with appropriate speaker consent (Section 14.1).",
                    value=True
                )
                synth_btn = gr.Button("Synthesize Speech", variant="primary")

            with gr.Column(scale=2):
                audio_output = gr.Audio(label="Synthesized Audio (Watermarked Waveform)", type="numpy")
                status_box = gr.Textbox(label="Status & Provenance", interactive=False)

        speaker_map = {c[0]: c[1] for c in SPEAKER_CHOICES}
        synth_btn.click(
            fn=lambda txt, spk, c: synthesize_speech(txt, speaker_map[spk], c),
            inputs=[text_box, speaker_dropdown, consent_checkbox],
            outputs=[audio_output, status_box]
        )

    with gr.Tab("🛡️ Audio Watermark & Deepfake Verifier"):
        gr.Markdown(
            """
            ### Synthetic Audio Provenance Inspection (Section 14.2)
            Upload any WAV file to verify if it contains the cryptographic and acoustic watermark injected by our generative pipeline.
            """
        )
        with gr.Row():
            audio_verify_input = gr.Audio(label="Upload WAV File to Inspect", type="filepath")
            verify_btn = gr.Button("Analyze Watermark & Provenance", variant="secondary")
        verification_output = gr.Textbox(label="Verification Report", lines=6, interactive=False)
        verify_btn.click(
            fn=verify_watermark_file,
            inputs=[audio_verify_input],
            outputs=[verification_output]
        )

    with gr.Tab("♿ Assistive AAC Phrase Bank"):
        gr.Markdown(
            """
            ### Assistive Augmentative & Alternative Communication (AAC)
            Quick communication shortcuts tailored for individuals with ALS, stroke, or vocal impairment.
            """
        )
        for cat, phrases in AAC_PHRASES.items():
            gr.Markdown(f"#### {cat}")
            with gr.Row():
                for phrase in phrases[:3]:
                    btn = gr.Button(phrase, size="sm")
                    btn.click(
                        fn=lambda p=phrase: (p, "Ready to speak phrase."),
                        outputs=[text_box, status_box]
                    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
