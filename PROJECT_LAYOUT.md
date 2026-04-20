# Assignment 2 Workspace Notes

This workspace is organized to support the experiment phase first.

## Dependency flow
- Part I is the main upstream dependency chain.
- Part II depends heavily on the transcript produced in Part I.
- Part III depends on Part II text plus source-audio prosody features.
- Part IV splits in two directions:
  - anti-spoofing can be prototyped once we have real and synthetic samples
  - adversarial robustness depends directly on the LID model from Part I

## Non-hierarchical exceptions
- speaker embedding extraction can begin before the final STT pipeline is complete
- anti-spoofing scaffolding can begin before the full lecture synthesis is complete
- evaluation utilities can be built in parallel

## Practical execution order
1. data ingestion and manifests
2. preprocessing and denoising
3. frame-level LID
4. STT baseline
5. constrained decoding and domain biasing
6. IPA conversion
7. translation / technical lexicon
8. speaker embedding and TTS baseline
9. DTW prosody transfer
10. anti-spoofing
11. adversarial attack
12. integrated evaluation
