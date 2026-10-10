//! Gemini-API-style endpoints: `POST /v1beta/models/{model}:embedContent`
//! and `:batchEmbedContents`, shaped like the hosted `gemini-embedding-2`
//! (`content.parts[]` with `text` / `inlineData`, `outputDimensionality`,
//! `{"embedding": {"values": […]}}`). Point a Gemini client — or GAISe's
//! Gemini adapter with a base-URL override — at this server.
//!
//! As with the hosted model, text is embedded **verbatim** unless a
//! `taskType` is given: callers put the `task: … | query: ` prefix in the
//! text themselves. `taskType` (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, …)
//! and `title` are accepted as a convenience and become the matching prefix.
//! API keys (`?key=`, `x-goog-api-key`) are ignored.

use gemini_embed_core::content::decode_base64;
use gemini_embed_core::{Content, ContentEmbeddingsRequest, EmbeddingTask, OneOrMany};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentRequest {
    /// `models/embeddinggemma-2`; the path's model wins when both are set.
    #[serde(default)]
    pub model: Option<String>,
    pub content: GeminiContent,
    #[serde(default, alias = "task_type")]
    pub task_type: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, alias = "output_dimensionality")]
    pub output_dimensionality: Option<u32>,
    /// The newer nesting of the three fields above.
    #[serde(default, alias = "embed_content_config")]
    pub embed_content_config: Option<EmbedContentConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentConfig {
    #[serde(default, alias = "task_type")]
    pub task_type: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, alias = "output_dimensionality")]
    pub output_dimensionality: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct BatchEmbedContentsRequest {
    pub requests: Vec<EmbedContentRequest>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiContent {
    #[serde(default)]
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiPart {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default, alias = "inline_data")]
    pub inline_data: Option<InlineData>,
    #[serde(default, alias = "file_data")]
    pub file_data: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    #[serde(alias = "mime_type")]
    pub mime_type: String,
    /// base64.
    pub data: String,
}

#[derive(Debug, Serialize)]
pub struct Values {
    pub values: Vec<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentResponse {
    pub embedding: Values,
    pub usage_metadata: UsageMetadata,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchEmbedContentsResponse {
    pub embeddings: Vec<Values>,
    pub usage_metadata: UsageMetadata,
}

/// `{"error": {"code": 400, "message": …, "status": "INVALID_ARGUMENT"}}`.
#[derive(Debug, Serialize)]
pub struct GeminiError {
    pub error: GeminiErrorBody,
}

#[derive(Debug, Serialize)]
pub struct GeminiErrorBody {
    pub code: u16,
    pub message: String,
    pub status: &'static str,
}

impl GeminiError {
    pub fn new(code: u16, message: String) -> Self {
        let status = match code {
            400 => "INVALID_ARGUMENT",
            404 => "NOT_FOUND",
            413 => "INVALID_ARGUMENT",
            _ => "INTERNAL",
        };
        Self {
            error: GeminiErrorBody {
                code,
                message,
                status,
            },
        }
    }
}

/// `embeddinggemma-2:embedContent` → (`embeddinggemma-2`, `embedContent`).
/// Splits at the last `:` so model names may contain colons
/// (`embeddinggemma-2:570m:embedContent`).
pub fn split_target(target: &str) -> Option<(&str, &str)> {
    let (model, method) = target.rsplit_once(':')?;
    Some((model.trim_start_matches("models/"), method))
}

/// `models/embeddinggemma-2` → `embeddinggemma-2`.
pub fn strip_models_prefix(model: &str) -> &str {
    model.trim().trim_start_matches("models/")
}

fn part_content(part: &GeminiPart) -> Result<Content, String> {
    if part.file_data.is_some() {
        return Err("fileData (URIs) is not supported; send inlineData".into());
    }
    if let Some(text) = &part.text {
        return Ok(Content::text(text.clone()));
    }
    let inline = part
        .inline_data
        .as_ref()
        .ok_or_else(|| "a part needs `text` or `inlineData`".to_string())?;
    let mut bytes = decode_base64(&inline.data)?;
    let mime = inline.mime_type.trim().to_ascii_lowercase();
    bytes.mime = Some(mime.clone());
    Ok(match mime.split('/').next().unwrap_or_default() {
        "image" => Content::Image {
            data: bytes,
            format: Some(mime),
        },
        "audio" => Content::Audio {
            data: bytes,
            format: Some(mime),
        },
        "video" => Content::Video {
            data: Some(bytes),
            format: Some(mime),
            frames: Vec::new(),
        },
        _ => Content::File {
            data: bytes,
            name: None,
        },
    })
}

/// One Gemini request → one GAISe content item plus its options.
pub struct Converted {
    pub item: Content,
    pub task: Option<EmbeddingTask>,
    pub title: Option<String>,
    pub dimensions: Option<u32>,
}

pub fn convert(request: &EmbedContentRequest) -> Result<Converted, String> {
    let parts = request
        .content
        .parts
        .iter()
        .map(part_content)
        .collect::<Result<Vec<_>, _>>()?;
    let item = match parts.len() {
        0 => return Err("`content.parts` is empty".into()),
        1 => parts.into_iter().next().expect("one part"),
        _ => Content::Parts { parts },
    };
    let cfg = request.embed_content_config.as_ref();
    Ok(Converted {
        item,
        task: request
            .task_type
            .as_deref()
            .or(cfg.and_then(|c| c.task_type.as_deref()))
            .filter(|t| !t.is_empty() && !t.eq_ignore_ascii_case("TASK_TYPE_UNSPECIFIED"))
            .map(EmbeddingTask::parse),
        title: request.title.clone().or(cfg.and_then(|c| c.title.clone())),
        dimensions: request
            .output_dimensionality
            .or(cfg.and_then(|c| c.output_dimensionality)),
    })
}

/// A content request for one converted Gemini request.
pub fn to_content_request(model: &str, c: Converted) -> ContentEmbeddingsRequest {
    ContentEmbeddingsRequest {
        model: model.to_string(),
        input: OneOrMany::One(c.item),
        // No taskType → text verbatim, like the hosted model.
        prompt: c.task.is_none().then(String::new),
        task: c.task,
        title: c.title,
        dimensions: c.dimensions,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_camel_and_snake_case() {
        let camel = r#"{"model":"models/embeddinggemma-2","content":{"parts":[{"text":"a cat"},{"inlineData":{"mimeType":"image/png","data":"AQID"}}]},"outputDimensionality":256,"taskType":"RETRIEVAL_QUERY"}"#;
        let snake = r#"{"content":{"parts":[{"text":"a cat"},{"inline_data":{"mime_type":"image/png","data":"AQID"}}]},"output_dimensionality":256,"task_type":"RETRIEVAL_QUERY"}"#;
        for json in [camel, snake] {
            let r: EmbedContentRequest = serde_json::from_str(json).unwrap();
            let c = convert(&r).unwrap();
            assert_eq!(c.dimensions, Some(256));
            assert_eq!(c.task, Some(EmbeddingTask::Query));
            match c.item {
                Content::Parts { parts } => {
                    assert!(
                        matches!(&parts[1], Content::Image { data, .. } if data.data == vec![1, 2, 3])
                    );
                }
                other => panic!("{other:?}"),
            }
        }
        let nested = r#"{"content":{"parts":[{"text":"x"}]},"embedContentConfig":{"taskType":"RETRIEVAL_DOCUMENT","title":"T","outputDimensionality":128}}"#;
        let c = convert(&serde_json::from_str(nested).unwrap()).unwrap();
        assert_eq!(
            (c.task, c.title.as_deref(), c.dimensions),
            (Some(EmbeddingTask::Document), Some("T"), Some(128))
        );
    }

    #[test]
    fn plain_text_is_verbatim() {
        let r: EmbedContentRequest = serde_json::from_str(
            r#"{"content":{"parts":[{"text":"task: search result | query: x"}]}}"#,
        )
        .unwrap();
        let req = to_content_request("m", convert(&r).unwrap());
        assert_eq!(req.prompt.as_deref(), Some(""));
        assert!(req.task.is_none());
    }

    #[test]
    fn splits_targets() {
        assert_eq!(
            split_target("embeddinggemma-2:embedContent"),
            Some(("embeddinggemma-2", "embedContent"))
        );
        assert_eq!(
            split_target("embeddinggemma-2:570m:batchEmbedContents"),
            Some(("embeddinggemma-2:570m", "batchEmbedContents"))
        );
        assert_eq!(split_target("nocolon"), None);
        assert!(convert(&serde_json::from_str(r#"{"content":{"parts":[]}}"#).unwrap()).is_err());
        assert!(
            convert(
                &serde_json::from_str(
                    r#"{"content":{"parts":[{"fileData":{"fileUri":"gs://x"}}]}}"#
                )
                .unwrap()
            )
            .is_err()
        );
    }
}
