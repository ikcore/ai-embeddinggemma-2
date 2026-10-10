# ai-gemini-embeddings-2

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)
![Backends](https://img.shields.io/badge/backends-CPU%20%7C%20CUDA%20%7C%20Metal%20%7C%20MLX-6f42c1.svg)

A Rust SDK and server for Google's open-weights multimodal embedding model, [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2). It embeds **text, images, video, PDF pages and audio** into one 768-d vector space, so a text query can find a photo, a slide or a sound clip.

- **Runs anywhere:** CPU, NVIDIA CUDA and Apple Metal through [candle](https://github.com/huggingface/candle), plus Apple **MLX** (Metal 4, M5 GPU Neural Accelerators) through [mlx-rs](https://github.com/oxiglade/mlx-rs).
- **Faithful:** validated against Hugging Face `transformers` — text, vision and audio outputs are identical in f32 on CPU and CUDA (cosine 1.000000).
- **Multimodal API:** one vector per item, or one vector for text + image together; Matryoshka sizes 768 / 512 / 256 / 128.
- **Drop-in contracts:** speaks [GAISe](https://github.com/ikcore/gaise)'s embedding contract (`/v1/embeddings`, `/v1/embeddings/contents`) and the Gemini API's `embedContent` / `batchEmbedContents`.
- **Self-contained:** one binary, weights in a plain `./models` folder, CPU and CUDA Docker images with `ffmpeg` and `pdftoppm` for video and PDF.

Sister project: [ai-harrier-embedding](https://github.com/ikcore/ai-harrier-embedding) (Microsoft Harrier text embeddings, same layout).

> **Which model is this?** Google ships two "Gemini Embedding 2"-era models. `gemini-embedding-2` is **API-only** (hosted, up to 3 072-d). **EmbeddingGemma 2** (`google/embeddinggemma-2`, Apache-2.0, released 2026-10-06) is the **open-weights** one, built on Gemma 4 — that is what this project runs. Nothing published says the two share an embedding space, so do not mix their vectors in one index.

## Contents

- [The model](#the-model) · [Quick start](#quick-start) · [Dimensions and precision](#dimensions-and-precision) · [Configuration](#configuration)
- [The contract](#the-contract) · [Library use](#library-use) · [How it works](#how-it-works) · [Backends](#backends) · [Docker](#docker) · [Limitations](#what-it-does-not-do-yet)

## The model

| | |
|---|---|
| Parameters | 740M in one checkpoint: 270M text encoder + 167M vision tower + 305M audio tower (load only what you need: `270m`, `440m`, `570m`, `740m`) |
| Output | 768-d, L2-normalized; Matryoshka 512 / 256 / 128 (truncate + re-normalize) |
| Context | 8 192 tokens per input, shared by all modalities |
| Image | 280 soft tokens by default (70 – 1 120), aspect-ratio preserving, 16-px patches pooled 3×3 |
| Video | 1 frame / s, ≤ 32 frames, 140 soft tokens per frame (audio track ignored) |
| Audio | 16 kHz mono, 25 soft tokens / s, ≤ 30 s per clip |
| PDF | pages rendered to images (≤ 6 pages by default) |
| Weights | 1.49 GB (bf16 safetensors) |

Two crates:

- **`crates/gemini_embed_core`** — the model (Gemma 4 text encoder, vision tower, audio Conformer) written once against a small `Backend` trait with candle and MLX implementations; image / audio / video / PDF preprocessing; tokenization, batching, mean pooling, Matryoshka; the task vocabulary and prompts; model loading (local directory first, Hub second).
- **`crates/gemini_embed_api`** — the HTTP server (`POST /v1/embeddings`, `POST /v1/embeddings/contents`, `POST /v1beta/models/{model}:embedContent` / `:batchEmbedContents`, `GET /v1/models`, `GET /health`) and the `download` command.

## Quick start

Rust 1.88+. For NVIDIA the CUDA toolkit (`nvcc`); for MLX macOS on Apple Silicon with Xcode and cmake.

```bash
git clone https://github.com/ikcore/ai-gemini-embeddings-2.git
cd ai-gemini-embeddings-2
```

### 1. Build

```bash
cargo build --release -p gemini_embed_api                       # CPU
cargo build --release --features cuda  -p gemini_embed_api      # NVIDIA                (alias: cargo cuda)
cargo build --release --features metal -p gemini_embed_api      # Apple, candle Metal   (alias: cargo metal)
cargo build --release --features mlx,metal -p gemini_embed_api  # Apple, MLX + fallback (alias: cargo mlx)
```

### 2. Get the weights (once)

```bash
target/release/gemini_embed_api download embeddinggemma-2      # → ./models/embeddinggemma-2/ (1.5 GB, plain files)
```

Skipping this also works: a model that is not in `./models` is fetched from the Hugging Face Hub on first start (`HF_HOME` / `HF_TOKEN` honoured).

**Where the weights come from.** They are not in this repository. They live on Hugging Face at **[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)** ([files](https://huggingface.co/google/embeddinggemma-2/tree/main), [model card](https://huggingface.co/google/embeddinggemma-2)), are licensed Apache-2.0, and are not gated, so no token is needed. To fetch them by hand instead (air-gapped installs, mirrors), put these files in `models/embeddinggemma-2/`:

| File | Needed for |
|---|---|
| [`model.safetensors`](https://huggingface.co/google/embeddinggemma-2/resolve/main/model.safetensors) (1.49 GB) | the weights (all three towers) |
| [`config.json`](https://huggingface.co/google/embeddinggemma-2/resolve/main/config.json) | architecture |
| [`tokenizer.json`](https://huggingface.co/google/embeddinggemma-2/resolve/main/tokenizer.json) | tokenizer |
| [`processor_config.json`](https://huggingface.co/google/embeddinggemma-2/resolve/main/processor_config.json) | image / video / audio preprocessing (defaults used if missing) |
| [`config_sentence_transformers.json`](https://huggingface.co/google/embeddinggemma-2/resolve/main/config_sentence_transformers.json) | task prompts (built-in copies used if missing) |

```bash
# any one of these
huggingface-cli download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
git clone https://huggingface.co/google/embeddinggemma-2 models/embeddinggemma-2      # needs git-lfs
mkdir -p models/embeddinggemma-2 && cd models/embeddinggemma-2 && for f in model.safetensors config.json tokenizer.json processor_config.json config_sentence_transformers.json; do curl -fLO "https://huggingface.co/google/embeddinggemma-2/resolve/main/$f"; done
```

Community GGUF conversions ([ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF), [unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF)) exist but are not read by this project yet; use the safetensors.

### 3. Serve

```bash
target/release/gemini_embed_api                                             # auto: CUDA → MLX → Metal → CPU
GEMINI_EMBED_DEVICE=mlx target/release/gemini_embed_api                     # force MLX
GEMINI_EMBED_MODELS=270m target/release/gemini_embed_api                    # text tower only (270M)
GEMINI_EMBED_MODALITIES=text,image target/release/gemini_embed_api          # text + vision (440M)
```

```text
gemini_embed_api 0.1.0 · backends: cpu, cuda · device: auto · models: embeddinggemma-2 · model dir: models
loaded gemini_embed::embeddinggemma-2 (740M params, inputs: text image video pdf audio, 768-d, max 8192 tokens, candle / cuda / bf16) in 1.3s
warmed up in 325 ms
listening on http://0.0.0.0:3000
```

### 4. Embed

```bash
# texts — the task becomes the model's prompt ("task: search result | query: …")
curl -s localhost:3000/v1/embeddings -H 'content-type: application/json' -d '{
  "task": "query", "input": ["a red bicycle", "how much protein should a female eat"]
}'

# anything — one vector per item; a `parts` item is ONE vector of all its parts
curl -s localhost:3000/v1/embeddings/contents -H 'content-type: application/json' -d '{
  "input": [
    { "type": "image", "data": "<base64 or data: URL>" },
    { "type": "audio", "data": "<base64 wav/mp3/flac/ogg/m4a>" },
    { "type": "file",  "name": "report.pdf", "data": "<base64>" },
    { "type": "parts", "parts": [ { "type": "text", "text": "a product photo of " },
                                  { "type": "image", "data": "<base64>" } ] }
  ]
}'
```

```json
{
  "output": [[0.0195, -0.0284, …], …],
  "usage": { "input": { "prompt_tokens": 629, "text_tokens": 34, "image_tokens": 532, "audio_tokens": 63 },
             "total": { "total_tokens": 629 } }
}
```

Ready-made requests for every endpoint are in [`examples/http/`](examples/http/) (VS Code REST Client / JetBrains HTTP Client). A CLI prints pairwise cosines across modalities:

```bash
cargo run --release -p gemini_embed_core --example embed -- --task query "a red circle" --image test.png --audio test.wav
```

```text
"a red circle on a colourful gradient"  1.000  0.652  0.565  0.777  0.614
"a recording of a rising tone"          0.652  1.000  0.549  0.639  0.653
"how much protein should a female eat"  0.565  0.549  1.000  0.516  0.518
test.png                                0.777  0.639  0.516  1.000  0.674
test.wav                                0.614  0.653  0.518  0.674  1.000
```

## Dimensions and precision

**Output size.** Every endpoint takes a requested size — `"dimensions": N` on `/v1/embeddings` and `/v1/embeddings/contents`, `outputDimensionality` on the Gemini-style routes. EmbeddingGemma 2 is trained for **768, 512, 256 and 128** (Matryoshka), so a request is snapped to the nearest of those (ties round up: 300 → 256, 384 → 512; anything above 768 returns 768). The full vector is truncated and then **re-normalized** to unit length; send `"normalize": false` for the raw truncated values. Without a size you get 768.

**Precision.** The checkpoint ships in bf16, and the model card warns that **fp16 overflows (NaNs)** — so fp16 is never the default:

| Backend | Default | |
|---|---|---|
| CUDA, Metal, MLX | **bf16** | cosine ≥ 0.9996 vs the f32 reference |
| CPU | **f32** | identical to the reference |

RMSNorms, attention softmax, the audio attention and the vision pooling always compute in f32 internally. Override with `GEMINI_EMBED_DTYPE=f32|bf16|f16` (f32 on CUDA is exact; f16 is accepted but not recommended).

## Where the models live

```
gemini-embeddings-2/
└── models/                          ← GEMINI_EMBED_MODEL_DIR (default ./models), git-ignored
    └── embeddinggemma-2/
        ├── config.json  tokenizer.json  processor_config.json  config_sentence_transformers.json …
        └── model.safetensors
```

`download` copies the files in as plain files (no symlinks, no cache hashes), so the folder can be mounted into a container, baked into an image, or copied to an air-gapped box. Resolution order for a model name: an explicit directory → `models/<id>` (for ids, aliases and size variants alike) → the Hub. With the folder populated, startup makes no network calls.

## Configuration

`gemini_embed_api --help` prints the same table.

| Variable | Meaning | Default |
|---|---|---|
| `GEMINI_EMBED_MODELS` | Comma-separated models; the first is the default. `embeddinggemma-2`, a size variant (`270m` text · `440m` + image · `570m` + audio · `740m` all), a Hub repo (`owner/name[@rev]`) or a directory | `embeddinggemma-2` |
| `GEMINI_EMBED_MODEL_DIR` | Directory of exported models, checked before the Hub | `./models` |
| `GEMINI_EMBED_DEVICE` | `auto` · `cpu` · `cuda[:n]` · `metal[:n]` · `mlx` · `gpu` (any accelerator, never CPU) | `auto` (CUDA → MLX → Metal → CPU) |
| `GEMINI_EMBED_DTYPE` | `f32` · `bf16` · `f16` (the model card warns f16 overflows) | `bf16` on GPU / MLX, `f32` on CPU |
| `GEMINI_EMBED_MODALITIES` | Towers to load: `all`, `text`, `text,image`, `text,audio` (video and PDF need `image`) | what the name implies, else `all` |
| `GEMINI_EMBED_MAX_INPUT_TOKENS` | Context per input (text is truncated; media that do not fit are rejected) | `8192` |
| `GEMINI_EMBED_MAX_BATCH_SIZE` / `_MAX_BATCH_TOKENS` | Rows / padded tokens per text forward pass | `32` / `16384` |
| `GEMINI_EMBED_IMAGE_TOKENS` | Soft tokens per image: `70` `140` `280` `560` `1120` | `280` |
| `GEMINI_EMBED_VIDEO_TOKENS` / `_VIDEO_FPS` / `_VIDEO_MAX_FRAMES` | Per-frame tokens, sampling rate, frame cap (evenly thinned beyond) | `140` / `1` / `32` |
| `GEMINI_EMBED_MAX_AUDIO_SECONDS` | Longest clip; longer audio is truncated | `30` |
| `GEMINI_EMBED_PDF_MAX_PAGES` / `_PDF_DPI` | Pages rendered per PDF, render resolution | `6` / `144` |
| `GEMINI_EMBED_FFMPEG` / `_PDFTOPPM` | External tools for video containers and PDFs | `ffmpeg` / `pdftoppm` on `PATH` |
| `GEMINI_EMBED_MAX_BODY_MB` | Largest request body (media arrive inline) | `64` |
| `GEMINI_EMBED_WARMUP` | One forward pass per loaded tower at startup | `true` |
| `GEMINI_EMBED_LOG_REQUESTS` | One stderr line per request | `true` |
| `GEMINI_EMBED_HOST`, `GEMINI_EMBED_PORT` / `PORT` | Bind address | `0.0.0.0`, `3000` |

## The contract

### `POST /v1/embeddings` — texts

GAISe's `GaiseEmbeddingsRequest` / `GaiseEmbeddingsResponse`:

| Field | Meaning |
|---|---|
| `model` | `gemini_embed::embeddinggemma-2`, the bare id, the Hub repo, an alias (`eg2`) or a size variant. Empty → the default. Any `provider::` prefix is stripped, so a GAISe router can forward requests unchanged. |
| `input` | one string or an array; vectors come back in input order |
| `task` | what the vectors are for (below). Default `document`. |
| `dimensions` | Matryoshka size, snapped to 128 / 256 / 512 / 768 (nearest, ties up — GAISe's rule) |
| `normalize` | unset / `true` → unit-length (re-normalized after truncation); `false` → the raw mean-pooled vector |
| `prompt` | *(extension)* a literal prefix instead of the task's; `""` embeds the text verbatim |
| `title` | *(extension)* document title: `title: {title} \| text: …` |

### `POST /v1/embeddings/contents` — any modality

GAISe's `GaiseContentEmbeddingsRequest`: same fields, `input` is a list of content items, one vector per item:

| `type` | Fields | |
|---|---|---|
| `text` | `text` | |
| `image` | `data`, `format?` | PNG, JPEG, WebP, GIF, BMP |
| `audio` | `data`, `format?` | WAV, MP3, FLAC, Ogg/Vorbis, M4A/AAC — mixed to mono, resampled to 16 kHz |
| `video` | `data` *or* `frames: [image…]`, `format?` | *(extension)* containers (MP4, MOV, WebM, …) need `ffmpeg`; `frames` needs nothing |
| `file` | `data`, `name?` | routed by name / MIME / magic bytes: image, audio, video, PDF (`pdftoppm`), or UTF-8 text |
| `parts` | `parts: [item…]` | everything in one sequence → **one** vector (text + image, a PDF's pages, …) |

`data` is accepted the way GAISe serialises it (a JSON array of bytes), as base64, or as a `data:` URL. Text-only items get the task prompt; media items get none (the model card's guidance) unless `prompt` is set. Inputs are laid out exactly as `EmbeddingGemma2Processor` does: `<bos>`, parts in order — text verbatim, each image as `<|image>` + soft tokens + `<image|>`, each video frame likewise with `<|video|>`, audio between `<|audio>` / `<audio|>` — then `<eos>`; media-only items separate parts with a space.

### Tasks → prompts

The prefixes come from the checkpoint's `config_sentence_transformers.json`. Every GAISe task (and its aliases, case-insensitive, `-`/space = `_`, plus the Gemini `taskType` enum names) maps to one:

| Task | Aliases | Prefix |
|---|---|---|
| `document` (default) | `retrieval_document`, `search_document`, `passage`, `index` | `title: none \| text: ` (`title: {title} \| text: ` with a title) |
| `query` | `retrieval_query`, `search_query`, `search` | `task: search result \| query: ` |
| `similarity` | `semantic_similarity`, `sts` | `task: sentence similarity \| query: ` |
| `classification` | `classify` | `task: classification \| query: ` |
| `clustering` | `cluster` | `task: clustering \| query: ` |
| `code_query` | `code_retrieval_query`, `code` | `task: code retrieval \| query: ` |
| `fact_verification` | `fact`, `fact_checking` | `task: fact checking \| query: ` |
| `question_answering` | `qa` | `task: question answering \| query: ` |
| a checkpoint prompt name | `BitextMining`, `Reranking`, `Summarization`, … | that prompt |
| anything else | — | `task: {value} \| query: ` |

### Gemini API style — `POST /v1beta/models/{model}:embedContent` and `:batchEmbedContents`

Shaped like the hosted `gemini-embedding-2`: `content.parts[]` with `text` / `inlineData {mimeType, data}`, `outputDimensionality` (top level or in `embedContentConfig`; camelCase or snake_case), responses `{"embedding": {"values": […]}}` / `{"embeddings": [{"values": […]}]}` plus `usageMetadata.promptTokenCount`. As with the hosted model, **text is embedded verbatim** — put the `task: … | query: ` prefix in the text yourself — unless a `taskType` (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, …) is given, which becomes the matching prefix. Several parts in one content give one vector. API keys are ignored; `fileData` (URIs) is rejected. Errors use Gemini's `{"error": {"code", "message", "status"}}` shape.

### `GET /v1/models`

GAISe-shaped records (`id`, `provider`, `capabilities.input` = the loaded modalities, `limits`) plus an `embedding` profile (`dimension_rule: {"set": [128, 256, 512, 768]}`, the resolved `task_prompts`) and `runtime` facts (`backend`, `device`, `dtype`, `parameters`, image / video token budgets, audio limit). `?operation=` and `?provider=` filter like GAISe.

Errors are plain-text bodies like gaise-api: `400` for bad input or undecodable media (including a missing `ffmpeg` / `pdftoppm`, named in the message), `404` for a model that is not loaded, `413` for a body over the limit, `422` for JSON that does not match the schema, `500` for inference failures.

### Using it from GAISe

- **As its own provider**: point a GAISe HTTP client at `/v1/embeddings` and `/v1/embeddings/contents`; requests and responses are GAISe's types, and `task`, `dimensions`, `normalize` behave as GAISe documents.
- **Through GAISe's Gemini adapter**: set the Gemini connection's base URL (`GaiseConnection` override) to `http://host:3000/v1beta` and use model `embeddinggemma-2`. GAISe's `gemini-embedding-2` profile sends the task as a text prefix, which this server receives verbatim — the same convention.

## Library use

```rust
use gemini_embed_core::{Content, ContentEmbeddingsRequest, Embedder, EmbedderOptions, EmbeddingTask, EmbeddingsRequest};

let embedder = Embedder::load("embeddinggemma-2", EmbedderOptions::default())?;
let query = embedder.embed_request(&EmbeddingsRequest {
    input: vec!["a red bicycle".to_string()].into(),
    task: Some(EmbeddingTask::Query),
    ..Default::default()
})?;
let photo = embedder.embed_contents(&ContentEmbeddingsRequest {
    input: vec![Content::Image { data: std::fs::read("bike.jpg")?.into(), format: None }].into(),
    ..Default::default()
})?;

// or decoded inputs, no prompts: Part::Text / Image(RgbImage) / Audio(16 kHz samples) / Video(frames)
use gemini_embed_core::Part;
let v = embedder.embed_inputs(&[vec![Part::Text("a photo of ".into()), Part::Image(image::open("bike.jpg")?.to_rgb8())]], Some(256), true)?;
```

`Embedder` is `Send + Sync`; forward passes are serialized through an internal lock so one `Arc<Embedder>` serves a whole server, while media decoding, resizing and tokenization run outside the lock, in parallel across the items of a request.

## How it works

- **One model, two tensor backends.** The three towers in `gemini_embed_core::model` are written against `backend::Backend` — about 35 operations: tensor algebra plus fused `rms_norm`, `rope`, `attention`, `conv2d`, `gather_last`. `CandleBackend` implements it for CPU / CUDA / Metal (exact query-chunked attention that keeps memory flat for long inputs, with a device-side bidirectional sliding-window mask); `MlxBackend` maps the same calls onto `mlx_rs::fast::{rms_norm, layer_norm, scaled_dot_product_attention}`, `ops::conv2d` (NHWC) and friends. Adding a backend means implementing the trait, not the model.
- **Text encoder** (Gemma 4, bidirectional): token embeddings × √512 rounded to the weight dtype; *projection-only per-layer embeddings* computed from the merged input embeddings (so media soft tokens feed them too); 24 layers — 5 sliding-window (head_dim 256, 2 KV heads, θ = 10⁴, `|q − k| ≤ 512`) to 1 full (head_dim 512, 1 KV head, θ = 10⁶) — each with q/k RMSNorms, a scale-free v RMSNorm, attention scale 1.0, sandwich norms around attention and the GELU-tanh MLP, a PLE gate and a learned `layer_scalar`; final norm and the 512 → 768 projection per token; mean pooling over every real token (prompt and specials included, like sentence-transformers' `Pooling(mean)`). Gemma 4's RMSNorm is `x · w` in f32, not Gemma 3's `x · (1 + w)`.
- **Vision tower**: aspect-preserving resize to multiples of 48 px within the soft-token budget (`get_aspect_ratio_preserving_size`), 16×16×3 patches mapped to `2(x − 0.5)`, learned x/y position tables, 16 layers with axial 2-D RoPE (θ = 100; first half of each head rotated by x, second by y), 3×3 average pooling over the patch grid × √768 in f32, then `embed_vision` (scale-free RMSNorm, 768 → 512). Images are encoded unpadded, one at a time — equivalent to the reference's padded batch, whose padding is masked out.
- **Audio tower**: the `Gemma4AudioFeatureExtractor` front end ported exactly (semicausal padding, periodic Hann, 512-point FFT magnitude, 128 HTK mel bins, `ln(mel + 0.001)`, frame-validity mask), two 3×3 stride-2 convs with LayerNorm + ReLU, 12 Conformer layers with clipped linears, chunked local attention (each token sees itself and the 11 before it; logits `q·k + q·r(t − s)` from a projected sinusoidal relative table, soft-capped at 50) computed in banded dense form, a causal depthwise conv, `output_proj` and `embed_audio`.
- **Prompts and layout** follow `EmbeddingGemma2Processor` and the checkpoint's `chat_template.jinja`, with placeholder runs built directly as token ids rather than by tokenizing thousands of placeholder strings (equivalent, since they are special tokens).

### Validation

Checked against `transformers` 5.19 (float32, eager attention) with [`scripts/reference.py`](scripts/reference.py): queries, a long document, an empty string, a padded mixed-length batch, a 640×480 image, a 768×768 image (no resize), text + image in one vector, and a 2.5 s audio clip. Token ids, placeholder layouts, mel frames and frame masks match exactly.

| Input | CPU f32 | CUDA f32 (RTX 4080) | CUDA bf16 |
|---|---|---|---|
| text (batched and alone) | cosine 1.000000, max \|Δ\| ≤ 1.2e-7 | 1.000000, ≤ 9.7e-8 | ≥ 0.99996 |
| image, no resize (vision tower) | 1.000000, 2.4e-7 | 1.000000, 2.3e-7 | 0.99993 |
| image, resized | 0.999987 | 0.999987 | 0.99996 |
| text + image | 0.999970 | 0.999970 | 0.99989 |
| audio (audio tower + mel front end) | 1.000000, 4.2e-7 | 1.000000, 1.6e-7 | 0.99967 |

The towers are exact; the only f32 difference is the image resampler (the `image` crate's Catmull-Rom vs torchvision's antialiased bicubic on uint8), visible only when an image is resized. Audio files go through symphonia and a windowed-sinc resampler, which are close to but not bit-identical with librosa / soxr when the source is not 16 kHz.

```bash
python -I scripts/reference.py models/embeddinggemma-2 target/reference          # torch, torchvision, transformers ≥ 5.19
GEMINI_EMBED_REFERENCE_DIR=target/reference cargo test --release -p gemini_embed_core --test reference -- --ignored --nocapture
GEMINI_EMBED_TEST_DEVICE=cuda GEMINI_EMBED_REFERENCE_DIR=target/reference cargo test --release --features cuda -p gemini_embed_core --test reference -- --ignored --nocapture
```

### Throughput

RTX 4080 (native Windows CUDA build, bf16) and i9-13900K (f32); one request each, server-side best of a few after warm-up, media sent base64 and decoded in the request.

| Request | CUDA bf16 | CPU f32 |
|---|---|---|
| 32 short texts (~19 tokens) | 27 ms | 1.26 s |
| 32 × ~260-token texts (8.3k tokens) | 116 ms (~72k tok/s) | 14.9 s |
| 1 image, 640×480 → 266 soft tokens | 122 ms | 9.0 s |
| 8 images | 725 ms | 70.7 s |
| 1 audio clip, 2.5 s | 85 ms | 0.63 s |
| 8 audio clips | 329 ms | 3.6 s |

Text and audio are cheap; an image is ~2 300 patches through the 16-layer vision tower plus its resize, and dominates on every backend (a 32-frame video is ~32 such passes at 140 tokens each). Use a GPU (or MLX) for image and video workloads, or lower `GEMINI_EMBED_IMAGE_TOKENS` / `_VIDEO_TOKENS` (70 tokens ≈ 4× fewer patches). Images within a request are encoded one after another; batching them through the vision tower is the next optimisation.

## Backends

| Backend | Build | Notes |
|---|---|---|
| CPU | `cargo build --release -p gemini_embed_api` | f32; `--features mkl` (Intel) / `accelerate` (macOS) speed up GEMMs. Fine for text and audio; images and especially video are slow on CPU (each image is ~2 400 patches through 16 layers). |
| CUDA | `--features cuda` (`cudnn` also available) | bf16 by default. Needs `nvcc` and a host compiler it accepts; on Windows see [the CUDA notes in ai-harrier-embedding](https://github.com/ikcore/ai-harrier-embedding#cuda-on-windows) (`NVCC_CCBIN`, `NVCC_APPEND_FLAGS`, `CUDA_COMPUTE_CAP` in `~/.cargo/config.toml`). |
| Metal (candle) | `--features metal` | Any Apple GPU, bf16. Portable fallback on macOS. |
| **MLX** | `--features mlx` (macOS only; add `metal` to keep the candle fallback in the same binary) | Apple's MLX via `mlx-rs` 0.32: Metal 4 TensorOps / Metal Performance Primitives, i.e. the GPU Neural Accelerators on M5-class chips (macOS 26.2+), plain Metal on earlier Apple Silicon. Arrays live in unified memory; graphs are lazy and evaluated when vectors are read. The dependency is target-gated, so `--features mlx` is a no-op on Linux / Windows. Building needs Xcode (with the Metal toolchain) and cmake. |

`GEMINI_EMBED_DEVICE=auto` prefers CUDA, then MLX, then candle Metal, then CPU. Feature flags are pass-throughs to candle (`cuda`, `cudnn`, `metal`, `accelerate`, `mkl`) and mlx-rs (`mlx`); `hf-hub` (default) enables Hub downloads.

**Status of MLX:** the MLX backend is written against `mlx-rs` 0.32.0's source (signatures checked) but has **not been compiled or run here** (no Mac was available while building this). Build it on a Mac and run the reference test with `GEMINI_EMBED_TEST_DEVICE=mlx` before relying on it; the candle Metal backend is the tested path on Apple hardware's API surface (it shares all model code with the validated CPU / CUDA runs).

## Docker

```bash
docker compose run --rm download                  # fill the weights volume once (1.5 GB)
docker compose up -d                              # CUDA, http://localhost:3000
docker compose --profile cpu up -d gemini-embed-cpu
```

| Image | Base | Contents |
|---|---|---|
| `docker/Dockerfile.cpu` | `debian:bookworm-slim` | the server + `ffmpeg` + `poppler-utils` (566 MB) |
| `docker/Dockerfile.cuda` | `ubuntu:22.04` | glibc binary + only the CUDA libraries candle links (`libcublas`, `libcublasLt`, `libcurand`, `libnvrtc`; `libcuda` comes from the host driver) + `ffmpeg` + `poppler-utils` (1.22 GB) |

Both read models from `/models` (`GEMINI_EMBED_MODEL_DIR`); `--target bake` copies `./models` into the image instead. The CUDA image needs the NVIDIA Container Toolkit and the GPU attached at start (`--gpus all` / `gpus: all`). On Docker Desktop for Windows prefer a named volume over a bind mount of `./models` — the 9p bridge makes the first load take ~30 s instead of ~1 s. MLX cannot run in a container (Docker on macOS has no GPU access): build natively on the Mac.

## What it does not do (yet)

- **MLX is unverified on hardware** (see above).
- **No quantisation / GGUF.** Weights run in bf16 / f32 from the safetensors (1.5 GB). `ggml-org/embeddinggemma-2-GGUF` exists; reading it would follow ai-harrier-embedding's GGUF path.
- **Video audio tracks are ignored** (as the hosted model does); embed the soundtrack separately as audio, or combine both in a `parts` item.
- **No cross-request batching** — each request is batched internally; concurrent requests queue on the model lock.
- **Attention is exact, not fused** — long text (8k tokens) and large images (1 120 soft tokens = 10 080 patches) are bound by `seq_len²` attention on candle backends.
- Video and PDF need external `ffmpeg` / `pdftoppm` (shipped in the Docker images); clients can always send frames or page images themselves.

## License

[MIT](LICENSE) © 2026 Ian Knowles. EmbeddingGemma 2's weights are released by Google under Apache-2.0 and are downloaded separately from the Hugging Face Hub; they are not part of this repository.
