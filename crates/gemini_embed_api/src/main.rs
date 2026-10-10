use gemini_embed_api::{AppState, ServerConfig, create_app};
use gemini_embed_core::Embedder;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
usage:
  gemini_embed_api                    serve (configured by environment, see below)
  gemini_embed_api download MODEL...  fetch models into the model directory and exit
                                      (MODEL = embeddinggemma-2, a Hub repo[@rev] or a
                                      directory; comma-separated lists are accepted too)
  gemini_embed_api --help";

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "gemini_embed_api {}\n\n{USAGE}\n\nConfiguration (environment):\n{}",
            env!("CARGO_PKG_VERSION"),
            ServerConfig::ENV_HELP
        );
        return;
    }
    let result = match args.first().map(String::as_str) {
        Some("download") => download(&args[1..]),
        Some(other) => Err(format!("unknown argument `{other}`\n{USAGE}").into()),
        None => run().await,
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `gemini_embed_api download embeddinggemma-2` → `./models/embeddinggemma-2/…`
fn download(models: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let models: Vec<&str> = models
        .iter()
        .flat_map(|m| m.split(','))
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .collect();
    if models.is_empty() {
        return Err("download needs at least one model name".into());
    }
    let dir = gemini_embed_core::model_dir();
    for name in models {
        eprintln!("fetching {name} into {}…", dir.display());
        let target = gemini_embed_core::export_to_dir(name, &dir)?;
        let bytes: u64 = std::fs::read_dir(&target)?
            .filter_map(|e| e.ok()?.metadata().ok())
            .map(|m| m.len())
            .sum();
        eprintln!("  {} ({:.2} GB)", target.display(), bytes as f64 / 1e9);
    }
    Ok(())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig::from_env()?;
    eprintln!(
        "gemini_embed_api {} · backends: {} · device: {} · models: {} · model dir: {}",
        env!("CARGO_PKG_VERSION"),
        gemini_embed_core::compiled_backends().join(", "),
        config.options.device,
        config.models.join(", "),
        config.model_dir.display()
    );

    let mut embedders: Vec<Arc<Embedder>> = Vec::with_capacity(config.models.len());
    for name in &config.models {
        let started = Instant::now();
        eprintln!("loading {name}…");
        let (options, owned, dir) = (
            config.options.clone(),
            name.clone(),
            config.model_dir.clone(),
        );
        let embedder = Arc::new(
            tokio::task::spawn_blocking(move || Embedder::load_from(&owned, &dir, options))
                .await??,
        );
        let info = embedder.info();
        eprintln!(
            "loaded {} ({} params, inputs: {}, {}-d, max {} tokens, {} / {} / {}) in {:.1}s",
            info.qualified_id,
            info.parameters,
            info.modalities.inputs().join(" "),
            info.embedding_dimensions,
            info.max_input_tokens,
            info.backend,
            info.device,
            info.dtype,
            started.elapsed().as_secs_f64()
        );
        if embedders.iter().any(|e| e.info().id == info.id) {
            return Err(format!(
                "model `{}` is listed more than once in GEMINI_EMBED_MODELS (one variant per model)",
                info.id
            )
            .into());
        }
        if config.warmup {
            let started = Instant::now();
            let warm = embedder.clone();
            tokio::task::spawn_blocking(move || warm.warmup()).await??;
            eprintln!(
                "warmed up in {:.0} ms",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        embedders.push(embedder);
    }

    let mut state = AppState::new(embedders);
    state.log_requests = config.log_requests;
    let app = create_app(Arc::new(state), config.max_body_bytes);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    eprintln!("shut down");
    Ok(())
}
