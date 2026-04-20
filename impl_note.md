# Implementation Notes
**Speech Understanding — Programming Assignment 2**  
*One non-obvious design choice per question*

---

## Question 1 — Transcription

**Design choice: Balancing LID data splits based on language fraction, instead of just random chunk assignment.**

One big issue I faced was that my LID validation score was great, but testing on the full 10-minute clip was terrible (English recall was ~43%). I spent hours trying different model architectures before I realized the issue was the data split itself. Randomly splitting the 43 chunks put almost all the English-heavy chunks in the training set. The validation set was almost completely Hindi-dominated.

The fix was simple: I sorted the chunks by how much English they contain, and made sure the train, val, and test splits had a proportional mix. After balancing the splits, my held-out English F1 jumped up to 0.83 with the exact same model weights. The takeaway is that evaluating code-switched speech requires careful data splitting, otherwise the metrics are completely misleading.

---

## Question 2 — Phonetic Mapping and Translation

**Design choice: Routing text using Unicode blocks before applying G2P rules, rather than relying on a language classifier.**

Standard G2P tools struggle with Hinglish because they assume the input is monolingual. For example, if you give an English G2P the romanized Hindi word *hai* (meaning "is"), it outputs the phonemes for the English word "hay". The word *jo* becomes "Joe". There were dozens of these cases in the transcript.

Instead of trying to use a language classification model to tag each word (which often fails on short words anyway), I just checked the Unicode script. Devanagari text goes straight to my custom lookup table. For latin script, I check against a small list of common romanized Hindi words. Everything else falls back to English G2P. It's a simple, hardcoded rule-based fix, but it works way better and is much more reliable than an ML model for this specific step.

---

## Question 3 — Voice Cloning and Synthesis

**Design choice: Duration matching for TTS chunks before applying DTW.**

When I first connected the pipeline end-to-end, the DTW alignment between the original 10-min audio and the Maithili output was completely ruined. The prosody transfer gave terrible results because the Maithili TTS was generating audio at a slightly different speed. Over 10 minutes, the TTS output ended up being almost 3 minutes longer than the source. DTW simply cannot handle a 20%+ difference in overall length.

My solution was to generate each TTS chunk separately, time-stretch it so its duration matches the corresponding source chunk exactly, and then stitch them all back together. Stretching short chunks by 5-15% isn't really noticeable to the ear, and it keeps the whole timeline perfectly synced up so DTW can actually map the prosody correctly.

---

## Question 4 — Adversarial Robustness and Spoofing

**Design choice: Using the exact log-Mel frontend for FGSM attacks instead of a basic spectrogram.**

My initial FGSM attack wasn't working well at all. I was taking the gradient through a basic STFT magnitude spectrogram instead of the full log-Mel pipeline. I had to turn epsilon up so high (around 0.001) that the audio sounded completely distorted and the SNR dropped way below the 40dB requirement, just to get a few frames to flip.

I realized that the gradient needs to go through the exact same feature extraction that the model uses. The Mel filterbank and the log compression change the gradients completely. If you use the wrong Jacobian, the attack directions are wrong. Once I swapped in the exact `torchaudio.MelSpectrogram` module I used during training, the attack worked perfectly at a much lower epsilon (0.000250) and kept the SNR over 50 dB.
