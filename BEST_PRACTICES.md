# EmbeddingGemma 2 — best practices

How to get good results from EmbeddingGemma 2 through this server: how to phrase queries and documents, how to embed images, video, audio and PDFs, which vector size to pick, and how to store and search the vectors.

Most of this comes from Google's [model card](https://huggingface.co/google/embeddinggemma-2) and [developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/). Advice that is general embedding practice rather than Google's guidance is marked **(general practice)**, and notes about this server's own limits are marked **(this server)**.

All examples assume the server is running on `http://localhost:3000` (see the [README](README.md#quick-start)).

---

## 1. The one rule that matters most: prefix your text

EmbeddingGemma 2 was trained with short **task prefixes** on text. Using the right one measurably improves quality; leaving it out still works but is less precise. The server adds the prefix for you from the `task` field — you send plain text.

| What you are embedding | `task` | Prefix the server adds |
|---|---|---|
| Documents / passages to be searched (default) | `document` | `title: none \| text: ` |
| Search queries | `query` | `task: search result \| query: ` |
| Questions (answer retrieval) | `question_answering` | `task: question answering \| query: ` |
| Claims (evidence retrieval) | `fact_verification` | `task: fact checking \| query: ` |
| Natural-language code search queries | `code_query` | `task: code retrieval \| query: ` |
| Texts to classify | `classification` | `task: classification \| query: ` |
| Texts to cluster | `clustering` | `task: clustering \| query: ` |
| Texts to compare for similarity / duplicates | `similarity` | `task: sentence similarity \| query: ` |

### Asymmetric tasks (search, QA, fact checking, code search)

Queries and documents get **different** prefixes. Embed your corpus with `task: document` and your queries with the matching query task:

```bash
# index time — documents
curl -s localhost:3000/v1/embeddings -H 'content-type: application/json' -d '{
  "task": "document",
  "input": ["The northern lights are caused by charged particles from the sun hitting the atmosphere."]
}'

# query time — the matching query task
curl -s localhost:3000/v1/embeddings -H 'content-type: application/json' -d '{
  "task": "query",
  "input": "What causes the northern lights?"
}'
```

Pair them consistently: `query` ↔ `document` for search, `question_answering` ↔ `document` for QA, `fact_verification` ↔ `document` for evidence, `code_query` ↔ `document` (code) for code search.

### Symmetric tasks (classification, clustering, similarity)

Every input being compared gets the **same** prefix. Send them all with the same `task`:

```bash
curl -s localhost:3000/v1/embeddings -H 'content-type: application/json' -d '{
  "task": "similarity",
  "input": ["A man is playing a guitar.", "Someone plays an instrument.", "The stock market fell today."]
}'
```

For classification, embed both your labelled examples and the new items with `task: classification`; for clustering, embed every item with `task: clustering`.

### Titles

Google's format for documents is `title: {title} | text: {content}`, with `title: none` when there is no title. A real title helps — use the page title, section heading, or (for code) the **file name**.

- One title for the whole request: send `"title": "..."` with `task: document`.
- A different title per document: build the prefix yourself and turn the automatic one off with `"prompt": ""`:

```bash
curl -s localhost:3000/v1/embeddings -H 'content-type: application/json' -d '{
  "prompt": "",
  "input": [
    "title: Aurora borealis | text: The northern lights are caused by charged particles from the sun…",
    "title: src/parser.rs | text: pub fn parse_date(s: &str) -> Result<NaiveDate> { … }"
  ]
}'
```

### Custom tasks

Anything that is not one of the canonical tasks is looked up among the checkpoint's own prompt names (`Reranking`, `BitextMining`, `Summarization`, `InstructionRetrieval`, …) and otherwise becomes `task: {your text} | query: `. Stick to the documented tasks unless you have measured that a custom one helps — the model was trained on the documented ones.

### Through the Gemini-style endpoint

`/v1beta/models/embeddinggemma-2:embedContent` embeds text **verbatim**, like Google's hosted API. Either include the prefix in the text yourself, or send a `taskType` (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `CODE_RETRIEVAL_QUERY`, `QUESTION_ANSWERING`, `FACT_VERIFICATION`) and the server adds it.

---

## 2. Images, video, audio and documents

### No prefix on media

Google's guidance: **prefixes apply to text only; pass images, video and audio without any prefix.** The server follows this automatically — `task` is applied to text-only items, never to media.

A text query is compared directly with media embeddings; they share one vector space:

```bash
# index an image (no task needed)
curl -s localhost:3000/v1/embeddings/contents -H 'content-type: application/json' -d '{
  "input": [ { "type": "image", "data": "<base64 PNG/JPEG>" } ]
}'

# search it with text — use the query task
curl -s localhost:3000/v1/embeddings -H 'content-type: application/json' -d '{
  "task": "query", "input": "waterproof shoes for trail running"
}'
```

For image-to-image (or audio-to-audio) search, embed both sides as media with no prefix.

### One item, several parts (interleaving)

To embed a product listing, a slide with notes, or a captioned image as **one** vector, put the parts in a single `parts` item, in reading order. Put the spacing you want inside the text parts:

```bash
curl -s localhost:3000/v1/embeddings/contents -H 'content-type: application/json' -d '{
  "input": [ { "type": "parts", "parts": [
    { "type": "text",  "text": "Waterproof running shoes. " },
    { "type": "image", "data": "<shoe.jpg base64>" },
    { "type": "text",  "text": " Featuring a breathable mesh upper. " },
    { "type": "image", "data": "<mesh.jpg base64>" },
    { "type": "text",  "text": " Grip test on wet rock: " },
    { "type": "video", "data": "<demo.mp4 base64>", "format": "mp4" }
  ] } ]
}'
```

This is the server's equivalent of the model card's `{"text": "… <|image|> … <|video|>", "image": [...], "video": ...}`. Do not type `<|image|>`-style placeholders into text yourself — the server places them.

### The shared 8 192-token budget

Every input — text and media together — must fit in 8 192 tokens:

| Modality | Cost | Fits alone (defaults) |
|---|---|---|
| Text | 1 token per sub-word | 8 192 tokens |
| Image | 280 tokens (configurable 70 – 1 120) | ~29 images |
| Video | 140 tokens per frame, 1 frame / s | ~58 frames (server default cap: 32 frames) |
| Audio | 25 tokens per second | ~327 s by the model card (see below) |

`usage.input` in every response reports `text_tokens`, `image_tokens`, `video_tokens` and `audio_tokens`, so you can see what each item cost. An input that does not fit is rejected with a 400 that says how many tokens it needed.

### Image detail vs. speed

`EMBEDDINGGEMMA_IMAGE_TOKENS` (70, 140, **280**, 560, 1120) sets how many soft tokens each image becomes. More tokens capture finer detail (small text, dense charts) at the cost of latency; fewer tokens let more images fit per input and run several times faster. Raise it for document pages and charts; lower it for photo search at scale. Video frames have their own budget (`EMBEDDINGGEMMA_VIDEO_TOKENS`, default 140).

### Video

- Frames are sampled at **1 per second** (`EMBEDDINGGEMMA_VIDEO_FPS`), at most 32 per video (`EMBEDDINGGEMMA_VIDEO_MAX_FRAMES`), evenly thinned beyond that. The audio track is ignored.
- Long videos become a sparse summary. **(general practice)** For search inside long videos, split them into scenes or fixed windows (for example 30 – 60 s) and embed each window as its own item, storing its timestamps.
- Containers need `ffmpeg` on the server (the Docker images include it). Clients can also send pre-extracted frames: `{"type": "video", "frames": ["<jpg>", "<jpg>", …]}`.

### Audio

- Send any common format; the server mixes to mono and resamples to 16 kHz, the rate the model expects.
- **(this server)** Clips are capped at 30 s by default (`EMBEDDINGGEMMA_MAX_AUDIO_SECONDS`), the limit of the reference feature extractor. You can raise it, but this server's audio attention cost grows with the square of the clip length, so prefer **splitting long recordings into windows of up to ~1 – 2 minutes** (with a little overlap) over embedding one long clip.
- The audio tower is trained on speech in many languages, environmental sounds and acoustic events — spoken-query search over audio archives and sound-event classification are both intended uses.

### PDFs, slides and charts

Google's guidance is that visual documents go through the **vision encoder as page images**. Send a PDF as a file and the server renders its pages (`pdftoppm`, first 6 pages by default):

```bash
curl -s localhost:3000/v1/embeddings/contents -H 'content-type: application/json' -d '{
  "input": [ { "type": "file", "name": "report.pdf", "data": "<base64>" } ]
}'
```

A file item becomes **one** vector of all its pages. **(general practice)** For retrieval inside documents, embed **one page per item** (render pages yourself, or send single-page PDFs) so a query can land on the right page, and raise `EMBEDDINGGEMMA_IMAGE_TOKENS` to 560 or 1 120 for text-dense pages. If you also have the extracted text, a `parts` item with the page image and its text is a strong combination.

---

## 3. Choosing a vector size

EmbeddingGemma 2 is trained with Matryoshka Representation Learning: the 768-d vector can be cut to **512, 256 or 128** and re-normalized. Ask for a size with `"dimensions"` (or `outputDimensionality` on the Gemini endpoint); the server snaps to the nearest supported size and re-normalizes.

| Size | Storage vs 768 | Google's guidance |
|---|---|---|
| **768** (default) | 1× | Multimodal and visual-document retrieval; when recall matters more than storage |
| **512** | ⅔ | Same use cases; near-lossless (multilingual MTEB 61.2 vs 61.4) |
| **256** | ⅓ | Storage-constrained indexes; most quality kept on text and code, about 95 % on image, video and speech |
| **128** | ⅙ | Large text-only indexes and first-stage shortlisting; about 90 % on text and code but about 75 % on multimodal — validate on your own data first |

Rules from the model card:

- **Truncate only when storage or search speed require it.**
- **Queries and documents must use the same size.** A 768-d query cannot be scored against a 128-d index.
- **Re-normalize after truncating** — the server does this unless you send `"normalize": false`. Skipping it silently degrades ranking.

**(general practice)** A useful pattern is two-stage search: shortlist with 128-d or 256-d vectors, then re-score the shortlist with full 768-d vectors stored alongside (or with a cross-encoder reranker).

---

## 4. Storing and searching vectors (general practice)

- **Similarity:** vectors come back unit-length, so cosine similarity and dot product give identical rankings — use whichever your vector database calls "inner product" / "dot" for speed.
- **Record what produced each vector:** the model (`embeddinggemma-2`), the size (768/512/256/128), and the prefix scheme. Re-embed everything when any of them changes.
- **Never mix models in one index.** EmbeddingGemma 2 vectors are not comparable with Google's hosted `gemini-embedding-2`, with EmbeddingGemma 1, or with Harrier. Vectors from different tower subsets of the *same* checkpoint (`270m`, `440m`, `570m`, `740m`) **are** compatible — Google's guide: you can start with a text-only index and add modalities later without re-computing existing embeddings.
- **Store as float32 or bfloat16.** A million vectors take about 1.5 GB at 768-d bf16 and about 250 MB at 128-d. Scalar (int8) or binary quantization in the vector database saves more; measure recall before relying on it.
- **Hybrid search:** combining vector search with keyword search (BM25) and fusing the rankings usually helps for names, codes, SKUs and rare terms that embeddings blur.
- **Thresholds are corpus-specific.** Scores are relative; calibrate any "is this a match?" cut-off on your own labelled pairs rather than picking a fixed number.

---

## 5. Long text and chunking (general practice)

The model reads up to 8 192 tokens per input, but **one vector per 8 k tokens is too coarse for most retrieval** — a passage gets lost in an average over many topics.

- Chunk documents into roughly **200 – 500 tokens** (a few paragraphs), ideally on paragraph or section boundaries, with about **10 – 15 % overlap**.
- Put the document title (and section heading) in each chunk's `title: … | text: …` prefix so chunks keep their context.
- Inputs longer than the limit are **truncated** (the end is dropped) for text; media inputs that do not fit are rejected. Check `usage.input.prompt_tokens` if you are near the limit.
- Use whole-document embeddings for document-level tasks (classification, clustering, deduplication), where averaging is what you want.

---

## 6. Precision and hardware

- **Never run in float16.** The model's activations exceed float16's range: it returns NaNs or silently degraded vectors, with no error. Use **bfloat16** (the default on CUDA, Metal and MLX) or **float32** (the default on CPU). `EMBEDDINGGEMMA_DTYPE` overrides it.
- bf16 vectors match the f32 reference to cosine ≥ 0.9996 — fine for search. Use f32 if you need bit-for-bit reproducibility across machines.
- **Load only the towers you need:** `270m` (text and code), `440m` (+ images, video, PDF), `570m` (+ audio), `740m` (everything). Unused towers are never loaded. Set it with the model name (`EMBEDDINGGEMMA_MODELS=270m`) or `EMBEDDINGGEMMA_MODALITIES`.
- **(this server)** Text and audio are cheap on any hardware; images and video are expensive on CPU (seconds per image). Use a GPU (CUDA, Metal or MLX) for media workloads, or lower the image token budget.

---

## 7. Throughput tips (this server)

- **Batch.** Send many inputs per request (`input` is a list): texts are length-sorted and run together, and media decoding and resizing run in parallel across items. One request with 32 texts is far faster than 32 requests.
- **Keep similar lengths together.** Mixed-length batches are split internally so short texts are not padded to long ones, but sending a long document with tiny queries in one request still costs more than separate requests.
- **Pre-size images on the client** if you are uploading large photos: the server resizes them to about 0.6 megapixels at the default budget anyway, so sending 4K images only costs bandwidth and decode time. The request body limit is 64 MB (`EMBEDDINGGEMMA_MAX_BODY_MB`).
- Requests to one model run one at a time (forward passes share a lock); run more replicas behind a load balancer for more concurrency.

---

## 8. Checklist

- [ ] Documents embedded with `task: document` (with real titles where you have them).
- [ ] Queries embedded with the matching query task (`query`, `question_answering`, `fact_verification`, `code_query`).
- [ ] Symmetric tasks use the same task on every input.
- [ ] Media sent without text prefixes; text + media that belong together sent as one `parts` item.
- [ ] Same `dimensions` for the index and for queries; vectors normalized.
- [ ] Model, size and prefix scheme recorded with the index; no mixing with other models.
- [ ] Long text chunked; long video and audio split into windows; PDFs embedded per page for retrieval.
- [ ] bf16 or f32 — never f16.
- [ ] Results checked on a sample of your own data before relying on 128-d or custom tasks.

## Sources

- [google/embeddinggemma-2 model card](https://huggingface.co/google/embeddinggemma-2) — task prefixes, titles, selective loading, Matryoshka sizes, precision, multimodal interleaving, token budgets, sampling rates, limitations.
- [EmbeddingGemma 2: the developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/) — query vs. document prompts, media without prompts, choosing dimensions, storage figures, compatibility across tower subsets.
- [`config_sentence_transformers.json`](https://huggingface.co/google/embeddinggemma-2/blob/main/config_sentence_transformers.json) — the exact prefix strings the server uses.
