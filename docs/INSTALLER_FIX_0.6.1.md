# Installer fix 0.6.1 — MV-Adapter/avatar readiness

## Observed target-machine failure

Stage `08-mv-adapter-environment` downloaded and installed all requested packages, then failed only because the combined one-line readiness probe returned non-zero while discarding stdout and stderr.

The 0.6.0 dependency set also requested `opencv-python` directly while `mediapipe==0.10.21` installed `opencv-contrib-python`. Both distributions provide the same `cv2` namespace. The OpenCV Python packaging guidance requires exactly one OpenCV wheel variant in an environment.

## Correction

- Remove all installed OpenCV wheel variants from the MV-Adapter environment.
- Reinstall only `opencv-contrib-python==4.11.0.86`, which provides both the base and contrib APIs needed by MediaPipe.
- Preserve already downloaded MediaPipe, JAX, Torch, Diffusers, and Transformers packages.
- Replace the opaque one-line probe with `scripts/verify_mv_adapter_env.py`.
- Record per-component results and tracebacks in:
  - `proof/mv-adapter-readiness.json`
  - `install-logs/mv-adapter-readiness.log`
- Verify Torch CUDA, Diffusers, the Transformers BiRefNet auto-model API, a single OpenCV distribution, MediaPipe Pose, and the pinned MV-Adapter checkout independently.

## Resume behavior

The stage fingerprint changes, so only stage 08 reruns. `uv` reuses its cache and existing environment. Stages 01–07 and completed stage 09 remain intact. Stage 10 then retries the already-fixed configuration merge.
