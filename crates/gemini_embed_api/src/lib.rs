//! Axum HTTP service for EmbeddingGemma 2, wire-compatible with GAISe's
//! embedding endpoints and with the Gemini API's `embedContent`.
//!
//! | Method | Path | Body | Returns |
//! |---|---|---|---|
//! | `POST` | `/v1/embeddings` | [`EmbeddingsRequest`] (texts) | [`EmbeddingsResponse`] |
//! | `POST` | `/v1/embeddings/contents` | [`ContentEmbeddingsRequest`] (text, image, audio, video, files, parts) | [`EmbeddingsResponse`] |
//! | `POST` | `/v1beta/models/{model}:embedContent` | Gemini `EmbedContentRequest` | `{"embedding": {"values": […]}}` |
//! | `POST` | `/v1beta/models/{model}:batchEmbedContents` | `{"requests": […]}` | `{"embeddings": [{"values": […]}]}` |
//! | `GET` | `/v1/models` | — | `{ "models": [ModelRecord] }` |
//! | `GET` | `/v1/models/{id}` | — | `ModelRecord` |
//! | `GET` | `/health` | — | `{ "status": "ok", … }` |

pub mod gemini;

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use gemini_embed_core::registry::{self, Modalities};
use gemini_embed_core::{
    ContentEmbeddingsRequest, DeviceRequest, EmbedError, Embedder, EmbedderOptions, EmbeddingTask,
    EmbeddingsRequest, EmbeddingsResponse, ModelInfo, OneOrMany, PROVIDER, Precision, ToolOptions,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Loaded models plus the default one.
pub struct AppState {
    /// Keyed by bare id (`embeddinggemma-2`).
    pub models: BTreeMap<String, Arc<Embedder>>,
    pub default_model: String,
    pub started: Instant,
    /// Log one line per request to stderr.
    pub log_requests: bool,
}

impl AppState {
    /// Build state from loaded embedders; the first one is the default.
    pub fn new(embedders: Vec<Arc<Embedder>>) -> Self {
        let default_model = embedders
            .first()
            .map(|e| e.info().id.clone())
            .unwrap_or_default();
        let models = embedders
            .into_iter()
            .map(|e| (e.info().id.clone(), e))
            .collect();
        Self {
            models,
            default_model,
            started: Instant::now(),
            log_requests: true,
        }
    }

    /// Find a loaded model by any spelling GAISe, a Gemini client or a user
    /// might send (`gemini_embed::embeddinggemma-2`, `models/eg2`, `270m`, …).
    pub fn model(&self, name: &str) -> Option<&Arc<Embedder>> {
        let bare = gemini::strip_models_prefix(registry::bare_id(name));
        if bare.is_empty() {
            return self.models.get(&self.default_model);
        }
        if let Some(m) = self.models.get(bare) {
            return Some(m);
        }
        if let Some((spec, _)) = registry::resolve(bare)
            && let Some(m) = self.models.get(spec.id)
        {
            return Some(m);
        }
        self.models.values().find(|m| {
            let info = m.info();
            info.id.eq_ignore_ascii_case(bare)
                || info.qualified_id.eq_ignore_ascii_case(bare)
                || info.source.eq_ignore_ascii_case(bare)
        })
    }

    pub fn loaded_ids(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    fn require(&self, name: &str) -> Result<Arc<Embedder>, ApiError> {
        self.model(name).cloned().ok_or_else(|| {
            ApiError(
                StatusCode::NOT_FOUND,
                format!(
                    "model `{name}` is not loaded (loaded: {})",
                    self.loaded_ids().join(", ")
                ),
            )
        })
    }
}

/// The router, with state attached. `max_body_bytes` bounds request bodies
/// (media arrive inline).
pub fn create_app(state: Arc<AppState>, max_body_bytes: usize) -> Router {
    Router::new()
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/v1/embeddings/contents", post(handle_content_embeddings))
        .route("/v1beta/models/{target}", post(handle_gemini))
        .route("/v1/models", get(handle_list_models))
        .route("/v1/models/{model}", get(handle_get_model))
        .route("/health", get(handle_health))
        .route("/", get(handle_health))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(middleware::from_fn_with_state(state.clone(), log_request))
        .with_state(state)
}

async fn log_request(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let started = Instant::now();
    let response = next.run(request).await;
    if state.log_requests {
        eprintln!(
            "{} {} {} {:.1}ms",
            method,
            path,
            response.status().as_u16(),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    response
}

/// Error response: status + plain-text message, like gaise-api.
#[derive(Debug)]
pub struct ApiError(pub StatusCode, pub String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

impl From<EmbedError> for ApiError {
    fn from(e: EmbedError) -> Self {
        let status = if e.is_client_error() {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        Self(status, e.to_string())
    }
}

impl From<JsonRejection> for ApiError {
    fn from(e: JsonRejection) -> Self {
        Self(e.status(), e.body_text())
    }
}

/// Run blocking model work off the async runtime.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, EmbedError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embedding task failed: {e}"),
            )
        })?
        .map_err(ApiError::from)
}

async fn handle_embeddings(
    State(state): State<Arc<AppState>>,
    body: Result<Json<EmbeddingsRequest>, JsonRejection>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    let Json(request) = body?;
    let embedder = state.require(&request.model)?;
    if request.input.is_empty() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "`input` must contain at least one text".into(),
        ));
    }
    Ok(Json(
        blocking(move || embedder.embed_request(&request)).await?,
    ))
}

async fn handle_content_embeddings(
    State(state): State<Arc<AppState>>,
    body: Result<Json<ContentEmbeddingsRequest>, JsonRejection>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    let Json(request) = body?;
    let embedder = state.require(&request.model)?;
    if request.input.is_empty() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "`input` must contain at least one item".into(),
        ));
    }
    Ok(Json(
        blocking(move || embedder.embed_contents(&request)).await?,
    ))
}

/// Gemini-shaped error bodies for the `/v1beta` routes.
fn gemini_error(e: ApiError) -> Response {
    let code = e.0.as_u16();
    (e.0, Json(gemini::GeminiError::new(code, e.1))).into_response()
}

async fn handle_gemini(
    State(state): State<Arc<AppState>>,
    Path(target): Path<String>,
    body: axum::body::Bytes,
) -> Response {
    match gemini_inner(state, target, body).await {
        Ok(r) => r,
        Err(e) => gemini_error(e),
    }
}

async fn gemini_inner(
    state: Arc<AppState>,
    target: String,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let bad = |m: String| ApiError(StatusCode::BAD_REQUEST, m);
    let (model, method) = gemini::split_target(&target).ok_or_else(|| {
        bad(format!(
            "expected `models/{{model}}:embedContent`, got `{target}`"
        ))
    })?;
    let embedder = state.require(model)?;
    let model = model.to_string();
    match method {
        "embedContent" => {
            let request: gemini::EmbedContentRequest =
                serde_json::from_slice(&body).map_err(|e| bad(e.to_string()))?;
            let converted = gemini::convert(&request).map_err(bad)?;
            let request = gemini::to_content_request(&model, converted);
            let mut response = blocking(move || embedder.embed_contents(&request)).await?;
            Ok(Json(gemini::EmbedContentResponse {
                embedding: gemini::Values {
                    values: response.output.pop().unwrap_or_default(),
                },
                usage_metadata: gemini::UsageMetadata {
                    prompt_token_count: prompt_tokens(&response),
                },
            })
            .into_response())
        }
        "batchEmbedContents" => {
            let request: gemini::BatchEmbedContentsRequest =
                serde_json::from_slice(&body).map_err(|e| bad(e.to_string()))?;
            if request.requests.is_empty() {
                return Err(bad("`requests` is empty".into()));
            }
            let requests = request
                .requests
                .iter()
                .map(|r| gemini::convert(r).map(|c| gemini::to_content_request(&model, c)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(bad)?;
            let responses = blocking(move || {
                // Requests with the same options share one call (and so
                // forward passes); vectors go back in request order.
                let mut groups: Vec<(ContentEmbeddingsRequest, Vec<usize>)> = Vec::new();
                for (i, r) in requests.into_iter().enumerate() {
                    let same = |g: &ContentEmbeddingsRequest| {
                        (&g.task, &g.prompt, &g.title, g.dimensions)
                            == (&r.task, &r.prompt, &r.title, r.dimensions)
                    };
                    match groups.iter_mut().find(|(g, _)| same(g)) {
                        Some((g, idx)) => {
                            let mut items = std::mem::take(&mut g.input).into_vec();
                            items.extend(r.input.into_vec());
                            g.input = OneOrMany::Many(items);
                            idx.push(i);
                        }
                        None => groups.push((r, vec![i])),
                    }
                }
                let mut out = vec![Vec::new(); groups.iter().map(|(_, i)| i.len()).sum()];
                let mut tokens = 0usize;
                for (g, idx) in groups {
                    let resp = embedder.embed_contents(&g)?;
                    tokens += prompt_tokens(&resp);
                    for (i, v) in idx.into_iter().zip(resp.output) {
                        out[i] = v;
                    }
                }
                Ok((out, tokens))
            })
            .await?;
            Ok(Json(gemini::BatchEmbedContentsResponse {
                embeddings: responses
                    .0
                    .into_iter()
                    .map(|values| gemini::Values { values })
                    .collect(),
                usage_metadata: gemini::UsageMetadata {
                    prompt_token_count: responses.1,
                },
            })
            .into_response())
        }
        other => Err(ApiError(
            StatusCode::NOT_FOUND,
            format!("unsupported method `{other}` (embedContent, batchEmbedContents)"),
        )),
    }
}

fn prompt_tokens(r: &EmbeddingsResponse) -> usize {
    r.usage
        .as_ref()
        .and_then(|u| u.total.as_ref())
        .and_then(|t| t.get("total_tokens").copied())
        .unwrap_or(0)
}

#[derive(Debug, Deserialize, Default)]
pub struct ListModelsQuery {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub operation: Option<String>,
}

async fn handle_list_models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListModelsQuery>,
) -> Result<Json<ListModelsResponse>, ApiError> {
    if let Some(op) = query.operation.as_deref() {
        let op = op.trim().to_ascii_lowercase();
        if !matches!(
            op.as_str(),
            "embeddings"
                | "embedding"
                | "embed"
                | "instruct"
                | "instruct_stream"
                | "speech"
                | "live"
                | "decision"
        ) {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                format!("unknown operation `{op}`"),
            ));
        }
        if !matches!(op.as_str(), "embeddings" | "embedding" | "embed") {
            return Ok(Json(ListModelsResponse::default()));
        }
    }
    if let Some(p) = query.provider.as_deref()
        && !p.trim().eq_ignore_ascii_case(PROVIDER)
    {
        return Ok(Json(ListModelsResponse::default()));
    }
    let models = state
        .models
        .values()
        .map(|m| ModelRecord::new(m, m.info().id == state.default_model))
        .collect();
    Ok(Json(ListModelsResponse {
        models,
        errors: Vec::new(),
    }))
}

async fn handle_get_model(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
) -> Result<Json<ModelRecord>, ApiError> {
    let embedder = state.require(&model)?;
    Ok(Json(ModelRecord::new(
        &embedder,
        embedder.info().id == state.default_model,
    )))
}

#[derive(Debug, Serialize)]
pub struct Health {
    pub status: &'static str,
    pub service: &'static str,
    pub version: &'static str,
    pub uptime_seconds: u64,
    pub default_model: String,
    pub models: Vec<String>,
    pub backends: Vec<&'static str>,
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Health> {
    Json(Health {
        status: "ok",
        service: "gemini_embed_api",
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: state.started.elapsed().as_secs(),
        default_model: state.default_model.clone(),
        models: state.loaded_ids(),
        backends: gemini_embed_core::compiled_backends(),
    })
}

/// `GET /v1/models` body — the shape of `GaiseListModelsResponse`.
#[derive(Debug, Serialize, Default)]
pub struct ListModelsResponse {
    pub models: Vec<ModelRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<serde_json::Value>,
}

/// One model, in the shape of `GaiseModel` plus an `embedding` profile
/// (GAISe's `EmbeddingProfile`) and the local `runtime` facts.
#[derive(Debug, Serialize)]
pub struct ModelRecord {
    pub id: String,
    pub provider: &'static str,
    pub display_name: String,
    pub status: &'static str,
    pub capabilities: Capabilities,
    pub limits: Limits,
    pub embedding: EmbeddingProfile,
    pub runtime: Runtime,
}

#[derive(Debug, Serialize)]
pub struct Capabilities {
    pub input: Modalities,
    pub output: &'static [&'static str],
    pub operations: &'static [&'static str],
    pub tools: &'static str,
    pub reasoning: &'static str,
    pub structured_output: &'static str,
    pub sources: &'static [&'static str],
}

#[derive(Debug, Serialize)]
pub struct Limits {
    pub max_input_tokens: u32,
    pub embedding_dimensions: u32,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingProfile {
    pub default_dimensions: u32,
    /// GAISe `DimensionRule::Set`.
    pub dimension_rule: DimensionRule,
    pub max_input_tokens: u32,
    pub normalized_output: bool,
    pub normalizes_truncation: bool,
    /// The service applies the task itself (`task` field); the prefixes it
    /// uses are listed in `task_prompts`.
    pub task_control: &'static str,
    pub task_prompts: BTreeMap<&'static str, String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionRule {
    Set(Vec<u32>),
}

#[derive(Debug, Serialize)]
pub struct Runtime {
    /// `candle` or `mlx`.
    pub backend: &'static str,
    pub device: String,
    pub dtype: Precision,
    pub parameters: String,
    pub source: String,
    pub image_soft_tokens: usize,
    pub video_soft_tokens: usize,
    pub max_audio_seconds: f64,
    pub load_seconds: f64,
    pub default: bool,
}

impl ModelRecord {
    pub fn new(embedder: &Embedder, default: bool) -> Self {
        let info: &ModelInfo = embedder.info();
        let mut sizes = info.matryoshka_dimensions.clone();
        sizes.sort_unstable();
        Self {
            id: info.qualified_id.clone(),
            provider: PROVIDER,
            display_name: info.display_name.clone(),
            status: "active",
            capabilities: Capabilities {
                input: info.modalities,
                output: &["embedding"],
                operations: &["embeddings"],
                tools: "unsupported",
                reasoning: "unsupported",
                structured_output: "unsupported",
                sources: &["registry", "provider"],
            },
            limits: Limits {
                max_input_tokens: info.max_input_tokens,
                embedding_dimensions: info.embedding_dimensions,
            },
            embedding: EmbeddingProfile {
                default_dimensions: info.embedding_dimensions,
                dimension_rule: DimensionRule::Set(sizes),
                max_input_tokens: info.max_input_tokens,
                normalized_output: true,
                normalizes_truncation: true,
                task_control: "task",
                task_prompts: EmbeddingTask::CANONICAL
                    .iter()
                    .map(|t| {
                        let name: &'static str = match t {
                            EmbeddingTask::Document => "document",
                            EmbeddingTask::Query => "query",
                            EmbeddingTask::Classification => "classification",
                            EmbeddingTask::Clustering => "clustering",
                            EmbeddingTask::Similarity => "similarity",
                            EmbeddingTask::CodeQuery => "code_query",
                            EmbeddingTask::FactVerification => "fact_verification",
                            EmbeddingTask::QuestionAnswering => "question_answering",
                            EmbeddingTask::Custom(_) => "custom",
                        };
                        (name, embedder.prompts().prefix(t, None))
                    })
                    .collect(),
            },
            runtime: Runtime {
                backend: info.backend,
                device: info.device.clone(),
                dtype: info.dtype,
                parameters: info.parameters.clone(),
                source: info.source.clone(),
                image_soft_tokens: info.image_soft_tokens,
                video_soft_tokens: info.video_soft_tokens,
                max_audio_seconds: info.max_audio_seconds,
                load_seconds: info.load_seconds,
                default,
            },
        }
    }
}

/// Operator configuration, read from the environment by the binary.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Models to load; the first is the default.
    pub models: Vec<String>,
    pub model_dir: std::path::PathBuf,
    pub options: EmbedderOptions,
    pub warmup: bool,
    pub log_requests: bool,
    pub max_body_bytes: usize,
}

impl ServerConfig {
    pub const ENV_HELP: &'static str = "\
GEMINI_EMBED_MODELS            comma-separated models to load; first is the default
                               (embeddinggemma-2, a size variant 270m / 440m / 570m / 740m,
                               a Hub repo[@rev] or a local directory)   default: embeddinggemma-2
GEMINI_EMBED_MODEL             single-model alias for GEMINI_EMBED_MODELS
GEMINI_EMBED_MODEL_DIR         directory of exported models, checked before the Hub (default ./models)
GEMINI_EMBED_DEVICE            auto | cpu | cuda[:n] | metal[:n] | mlx | gpu     (default auto:
                               CUDA, then MLX, then candle Metal, then CPU)
GEMINI_EMBED_DTYPE             f32 | bf16 | f16   (default bf16 on GPU / MLX, f32 on CPU; avoid f16)
GEMINI_EMBED_MODALITIES        towers to load: all | text,image,audio | text   (default: what the model
                               name implies, else all; video and PDF need image)
GEMINI_EMBED_MAX_INPUT_TOKENS  context per input, shared by all modalities (default 8192)
GEMINI_EMBED_MAX_BATCH_SIZE    rows per forward pass (default 32)
GEMINI_EMBED_MAX_BATCH_TOKENS  padded tokens per forward pass (default 16384)
GEMINI_EMBED_IMAGE_TOKENS      soft tokens per image: 70 | 140 | 280 | 560 | 1120 (default 280)
GEMINI_EMBED_VIDEO_TOKENS      soft tokens per video frame (default 140)
GEMINI_EMBED_VIDEO_FPS         video sampling rate (default 1)
GEMINI_EMBED_VIDEO_MAX_FRAMES  frames per video, evenly thinned beyond (default 32)
GEMINI_EMBED_MAX_AUDIO_SECONDS longest audio clip; longer audio is truncated (default 30)
GEMINI_EMBED_PDF_MAX_PAGES     pages rendered per PDF (default 6)
GEMINI_EMBED_PDF_DPI           PDF render resolution (default 144)
GEMINI_EMBED_FFMPEG            ffmpeg executable for video containers (default ffmpeg on PATH)
GEMINI_EMBED_PDFTOPPM          pdftoppm executable for PDFs (default pdftoppm on PATH)
GEMINI_EMBED_MAX_BODY_MB       largest request body (default 64)
GEMINI_EMBED_WARMUP            run one forward pass per tower at startup (default true)
GEMINI_EMBED_LOG_REQUESTS      log one line per request (default true)
GEMINI_EMBED_HOST              bind address (default 0.0.0.0)
GEMINI_EMBED_PORT / PORT       listen port (default 3000)
HF_HOME / HF_TOKEN             Hugging Face cache dir / token for downloads";

    pub fn from_env() -> Result<Self, String> {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// Parse from any key → value lookup (the environment, or a map in tests).
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let var = |k: &str| lookup(k).filter(|v| !v.trim().is_empty());
        let models: Vec<String> = var("GEMINI_EMBED_MODELS")
            .or_else(|| var("GEMINI_EMBED_MODEL"))
            .unwrap_or_else(|| gemini_embed_core::DEFAULT_MODEL_ID.to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if models.is_empty() {
            return Err("GEMINI_EMBED_MODELS is empty".into());
        }
        fn parse<T: std::str::FromStr>(
            k: &str,
            v: Option<String>,
            what: &str,
        ) -> Result<Option<T>, String> {
            v.map(|v| {
                v.trim()
                    .parse::<T>()
                    .map_err(|_| format!("{k} must be {what}, got `{v}`"))
            })
            .transpose()
        }
        let usize_var = |k: &str| parse::<usize>(k, var(k), "an integer");
        let f64_var = |k: &str| parse::<f64>(k, var(k), "a number");
        let bool_var = |k: &str, default: bool| -> Result<bool, String> {
            match var(k).map(|v| v.trim().to_ascii_lowercase()) {
                None => Ok(default),
                Some(v) if ["1", "true", "yes", "on"].contains(&v.as_str()) => Ok(true),
                Some(v) if ["0", "false", "no", "off"].contains(&v.as_str()) => Ok(false),
                Some(v) => Err(format!("{k} must be a boolean, got `{v}`")),
            }
        };
        let defaults = EmbedderOptions::default();
        let tool_defaults = ToolOptions::default();
        let options = EmbedderOptions {
            device: var("GEMINI_EMBED_DEVICE")
                .unwrap_or_default()
                .parse::<DeviceRequest>()
                .map_err(|e| e.to_string())?,
            dtype: var("GEMINI_EMBED_DTYPE")
                .map(|v| gemini_embed_core::parse_precision(&v))
                .transpose()
                .map_err(|e| e.to_string())?,
            modalities: var("GEMINI_EMBED_MODALITIES")
                .map(|v| Modalities::parse(&v))
                .transpose()?,
            max_input_tokens: usize_var("GEMINI_EMBED_MAX_INPUT_TOKENS")?,
            max_batch_size: usize_var("GEMINI_EMBED_MAX_BATCH_SIZE")?
                .unwrap_or(defaults.max_batch_size),
            max_batch_tokens: usize_var("GEMINI_EMBED_MAX_BATCH_TOKENS")?
                .unwrap_or(defaults.max_batch_tokens),
            image_soft_tokens: usize_var("GEMINI_EMBED_IMAGE_TOKENS")?,
            video_soft_tokens: usize_var("GEMINI_EMBED_VIDEO_TOKENS")?,
            max_audio_seconds: f64_var("GEMINI_EMBED_MAX_AUDIO_SECONDS")?,
            tools: ToolOptions {
                ffmpeg: var("GEMINI_EMBED_FFMPEG").unwrap_or(tool_defaults.ffmpeg),
                pdftoppm: var("GEMINI_EMBED_PDFTOPPM").unwrap_or(tool_defaults.pdftoppm),
                video_fps: f64_var("GEMINI_EMBED_VIDEO_FPS")?.unwrap_or(tool_defaults.video_fps),
                video_max_frames: usize_var("GEMINI_EMBED_VIDEO_MAX_FRAMES")?
                    .unwrap_or(tool_defaults.video_max_frames),
                pdf_max_pages: usize_var("GEMINI_EMBED_PDF_MAX_PAGES")?
                    .unwrap_or(tool_defaults.pdf_max_pages),
                pdf_dpi: parse::<u32>(
                    "GEMINI_EMBED_PDF_DPI",
                    var("GEMINI_EMBED_PDF_DPI"),
                    "an integer",
                )?
                .unwrap_or(tool_defaults.pdf_dpi),
            },
        };
        let port = parse::<u16>(
            "GEMINI_EMBED_PORT",
            var("GEMINI_EMBED_PORT").or_else(|| var("PORT")),
            "a port number",
        )?
        .unwrap_or(3000);
        let max_body_mb = usize_var("GEMINI_EMBED_MAX_BODY_MB")?.unwrap_or(64);
        Ok(Self {
            host: var("GEMINI_EMBED_HOST").unwrap_or_else(|| "0.0.0.0".into()),
            port,
            models,
            model_dir: var(gemini_embed_core::MODEL_DIR_ENV)
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from("models")),
            options,
            warmup: bool_var("GEMINI_EMBED_WARMUP", true)?,
            log_requests: bool_var("GEMINI_EMBED_LOG_REQUESTS", true)?,
            max_body_bytes: max_body_mb.max(1) * 1024 * 1024,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_resolves_nothing() {
        let state = AppState::new(Vec::new());
        assert!(state.model("").is_none());
        assert!(state.model("gemini_embed::embeddinggemma-2").is_none());
        assert!(state.loaded_ids().is_empty());
    }

    #[test]
    fn reads_the_environment() {
        let env: BTreeMap<&str, &str> = BTreeMap::from([
            ("GEMINI_EMBED_MODELS", "270m, embeddinggemma-2"),
            ("GEMINI_EMBED_DEVICE", "cuda:1"),
            ("GEMINI_EMBED_DTYPE", "bf16"),
            ("GEMINI_EMBED_MODALITIES", "text,audio"),
            ("GEMINI_EMBED_IMAGE_TOKENS", "560"),
            ("GEMINI_EMBED_VIDEO_MAX_FRAMES", "8"),
            ("PORT", "8080"),
            ("GEMINI_EMBED_WARMUP", "off"),
        ]);
        let c = ServerConfig::from_lookup(|k| env.get(k).map(|v| v.to_string())).unwrap();
        assert_eq!(c.models, vec!["270m", "embeddinggemma-2"]);
        assert_eq!(c.options.device, DeviceRequest::Cuda(1));
        assert_eq!(c.options.dtype, Some(Precision::BF16));
        assert_eq!(
            c.options.modalities,
            Some(Modalities {
                image: false,
                audio: true
            })
        );
        assert_eq!(c.options.image_soft_tokens, Some(560));
        assert_eq!(c.options.tools.video_max_frames, 8);
        assert_eq!(c.port, 8080);
        assert!(!c.warmup);
        assert_eq!(c.max_body_bytes, 64 << 20);
        let bad = BTreeMap::from([("GEMINI_EMBED_MAX_BATCH_SIZE", "lots")]);
        assert!(ServerConfig::from_lookup(|k| bad.get(k).map(|v| v.to_string())).is_err());
    }
}
