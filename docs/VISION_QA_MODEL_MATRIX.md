# Vision model candidates for the 6 GB pipeline

All footprint values are planning inputs. Local peak VRAM and output quality remain **NOT PROVEN** until measured on the GTX 1660 SUPER.

| Candidate | Role | Published scale / files | Integration decision |
|---|---|---:|---|
| Qwen3.5-2B | Primary structured visual supervisor | 2B; full checkpoint about 4.57 GB | Quantized local-server benchmark only; never full BF16 by default |
| MiniCPM-V 4.6 | Independent second opinion | 1.3B | Preferred secondary; load only after primary exits |
| Florence-2-base-ft | OCR, captions, boxes and region labels | 0.23B; about 463 MB | First specialist to implement |
| EdgeTAM | Prompted masks | checkpoint about 56 MB | Pair with Florence boxes after official API smoke test |
| Depth Anything 3 Small | Independent depth/camera cross-check | 34.3M; about 137 MB | Benchmark 448/518 long edge in separate environment |
| MoGe-2 ViT-S normal | Existing point/depth/normal/FOV baseline | 35M; ONNX about 141 MB | Already measured on target; not an independent judge of MoGe output |
| Qwen3-VL-Embedding-2B | Failure memory and duplicate retry detection | 2B; full checkpoint about 4.26 GB | Phase two, sequential and quantized |
| MoGe-3 | Future fine-detail geometry | code and weights not released as of 2026-08-03 | Watchlist only |

## Source records

- Qwen3.5: `https://huggingface.co/Qwen/Qwen3.5-2B`
- MiniCPM-V: `https://github.com/OpenBMB/MiniCPM-V`
- Florence-2: `https://huggingface.co/microsoft/Florence-2-base-ft`
- EdgeTAM: `https://github.com/facebookresearch/EdgeTAM`
- Depth Anything 3: `https://github.com/ByteDance-Seed/Depth-Anything-3`
- MoGe: `https://github.com/microsoft/MoGe`
- Qwen multimodal embeddings: `https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B`

## Selection rationale

The VLMs are used for visible semantics and failure routing, not numeric geometry. Florence and EdgeTAM provide inspectable boxes and masks. DA3 provides a separate depth-family comparison. MoGe remains the generating baseline and therefore cannot independently validate its own geometric assumptions. Embeddings are deferred because retrieval memory is useful only after the pipeline has a library of correctly classified successes and failures.
