use std::{env, net::SocketAddr, time::Duration};

use anyhow::{Context, bail};
use url::Url;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub listen_addr: SocketAddr,
    pub backend_url: String,
    pub api_key: Option<String>,
    pub connect_timeout: Duration,
}

impl AppConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let port = env::var("PORT")
            .unwrap_or_else(|_| "8080".to_owned())
            .parse::<u16>()
            .context("PORT must be a valid TCP port")?;
        let backend_url =
            env::var("VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_owned());
        let parsed_backend_url =
            Url::parse(&backend_url).context("VLLM_BASE_URL must be an absolute URL")?;
        if !matches!(parsed_backend_url.scheme(), "http" | "https") {
            bail!("VLLM_BASE_URL must use http or https");
        }

        let connect_timeout_secs = env::var("CONNECT_TIMEOUT_SECS")
            .unwrap_or_else(|_| "5".to_owned())
            .parse::<u64>()
            .context("CONNECT_TIMEOUT_SECS must be a positive integer")?;
        if connect_timeout_secs == 0 {
            bail!("CONNECT_TIMEOUT_SECS must be greater than zero");
        }

        let api_key = env::var("API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty());

        Ok(Self {
            listen_addr: SocketAddr::from(([0, 0, 0, 0], port)),
            backend_url: backend_url.trim_end_matches('/').to_owned(),
            api_key,
            connect_timeout: Duration::from_secs(connect_timeout_secs),
        })
    }
}
