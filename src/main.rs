use std::net::SocketAddr;

use anyhow::Context;
use rusty_inference::{api, config::AppConfig};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::from_env()?;
    let listen_addr: SocketAddr = config.listen_addr;
    let app = api::router(config)?;
    let listener = tokio::net::TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("could not bind to {listen_addr}"))?;

    info!(%listen_addr, "Rust inference gateway listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("inference gateway server failed")
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(%error, "failed to listen for shutdown signal");
    }
    info!("shutdown signal received");
}
