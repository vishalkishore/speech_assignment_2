# Assignment-2, Speech Understanding

## Vishal Kishore,B23CS1078



## Top-level layout:
- `source_clip_10min.wav`: final 10-minute source clip
- `stage1_lid`: frame-level LID
- `stage1_stt`: constrained Whisper decoding
- `stage1_preprocess`: denoising and normalization
- `stage2_ipa`: Hinglish to IPA conversion
- `stage2_translation`: Maithili dictionary and text translation
- `stage3_voice`: 60-second student voice reference and embedding
- `stage3_prosody`: DTW-based prosody transfer
- `stage3_tts`: final Maithili synthesis output
- `stage4_spoof`: LFCC anti-spoof classifier
- `stage4_attack`: FGSM-style adversarial robustness evaluation

## Setup

First, install the required dependencies:
```bash
pip install -e .
```

## How to Run

The pipeline is organized into modular stages. Each stage expects the previous stage's outputs to be available. You can execute the entire pipeline by running the scripts sequentially from the project root:

### Stage 1: Denoising, LID & Constrained STT

1. **Preprocessing (Optional but recommended for noisy inputs):**
   ```bash
   python stage1_preprocess/scripts/run_preprocess_manifest.py
   ```
   *Applies spectral subtraction and level normalization to all audio chunks to improve downstream ASR performance.*

2. **Frame-level Language Identification (LID):**
   ```bash
   python stage1_lid/src/training/finetune_clip_adapt_lid.py
   ```
   *Domain-adapts the pretrained BiLSTM LID model using pseudo-labeled chunks from the 10-minute clip. This generates the language hints used by the Whisper decoder.*

3. **Constrained ASR Transcription:**
   ```bash
   python stage1_stt/scripts/transcribe_whisper_chunked.py
   ```
   *Runs Whisper Large-v3 with a custom n-gram logit-bias to enforce technical vocabulary (e.g., "mel-filterbank"). It uses the LID outputs to intelligently select between constrained and baseline decodes.*

4. **Evaluate Word Error Rate (WER):**
   ```bash
   python stage1_stt/scripts/score_wer.py
   ```
   *Computes the temporally-aligned lenient WER against the reference spans.*

### Stage 2: IPA Conversion & Translation

5. **Hinglish to IPA & Maithili Translation:**
   ```bash
   python stage2_ipa/scripts/convert_transcript_txt_to_ipa.py
   ```
   *Routes text through Unicode-aware script handlers to convert Hinglish text to IPA representations. Then, it uses a 500-entry parallel glossary to translate the content into syntactically valid Maithili chunks.*

### Stage 3: Voice Cloning & Synthesis

6. **Maithili TTS Synthesis:**
   ```bash
   bash stage3_tts/scripts/run_lexmodi_mms_duration_aware.sh
   ```
   *Generates chunk-by-chunk Maithili speech using the Meta MMS model. It explicitly time-stretches the generated chunks to match the exact duration of the source segments to prevent accumulated temporal drift.*

7. **DTW Prosody Transfer:**
   ```bash
   bash stage3_prosody/scripts/run_prosody_pair.sh
   ```
   *Extracts F0, energy, and timing features from both the source audio and the TTS output, aligns them using dynamic time warping (DTW), and applies a PSOLA-based pitch shift to transfer the speaker's original prosody onto the cloned voice.*

### Stage 4: Adversarial Attack & Spoofing Defense

8. **Train LFCC Countermeasure:**
   ```bash
   bash stage4_spoof/scripts/run_lfcc_cm.sh
   ```
   *Trains a convolutional anti-spoofing classifier using Linear Frequency Cepstral Coefficients (LFCC) to distinguish between genuine human speech and the synthesized MMS output.*

9. **Evaluate FGSM Adversarial Robustness:**
   ```bash
   bash stage4_attack/scripts/run_fgsm_lid_attack.sh
   ```
   *Generates an imperceptible adversarial perturbation using the exact log-Mel frontend to flip the LID model's predictions from Hindi to English on a target chunk.*

## Results Summary

The pipeline achieved the following evaluation metrics on the final code-switched segment:

- **LID Macro-F1:** 0.951 (English: 0.867, Hindi: 0.933)
- **English WER:** 0.126 (12.6%)
- **Hindi WER:** 0.043 (4.3%)
- **Macro WER:** 0.084 (8.4%)
- **MCD (MMS Maithili Synthesis):** 5.52
- **Spoof EER:** 0.00%
- **FGSM Attack:** Successful at $\epsilon = 0.000232$ with an SNR of 59.75 dB

## Final submission-facing assets:
- `source_clip_10min.wav`
- `stage3_voice/student_voice_ref_60s.wav`
- `stage3_tts/final_audio/output_lrl_cloned_22050.wav`