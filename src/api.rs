use std::{io, time::Duration};

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderName, Request, Response, StatusCode, header},
    response::IntoResponse,
    routing::{any, get},
};
use futures_util::TryStreamExt;
use serde::Serialize;
use tracing::{instrument, warn};

use crate::config::AppConfig;

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    config: AppConfig,
}

impl AppState {
    fn new(config: AppConfig) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .pool_max_idle_per_host(128)
            .build()?;
        Ok(Self { client, config })
    }
}

pub fn router(config: AppConfig) -> anyhow::Result<Router> {
    let state = AppState::new(config)?;
    Ok(Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", any(proxy))
        .route("/v1/{*path}", any(proxy))
        .with_state(state))
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(HealthResponse { status: "ok" }))
}

#[derive(Serialize)]
struct ReadyResponse {
    status: &'static str,
    backend: &'static str,
}

async fn ready(State(state): State<AppState>) -> impl IntoResponse {
    let health_url = format!("{}/health", state.config.backend_url);
    match state
        .client
        .get(health_url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => (
            StatusCode::OK,
            Json(ReadyResponse {
                status: "ready",
                backend: "ready",
            }),
        ),
        Ok(response) => {
            warn!(status = %response.status(), "vLLM health check failed");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ReadyResponse {
                    status: "not_ready",
                    backend: "unavailable",
                }),
            )
        }
        Err(error) => {
            warn!(%error, "vLLM health check could not reach the backend");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ReadyResponse {
                    status: "not_ready",
                    backend: "unavailable",
                }),
            )
        }
    }
}

#[derive(Serialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Serialize)]
struct ApiErrorDetail {
    message: &'static str,
    r#type: &'static str,
    code: &'static str,
}

fn error_response(status: StatusCode, message: &'static str, code: &'static str) -> Response<Body> {
    (
        status,
        Json(ApiError {
            error: ApiErrorDetail {
                message,
                r#type: "server_error",
                code,
            },
        }),
    )
        .into_response()
}

#[instrument(skip_all, fields(path = %request.uri().path()))]
async fn proxy(State(state): State<AppState>, request: Request<Body>) -> Response<Body> {
    if let Some(expected_key) = state.config.api_key.as_deref()
        && !has_bearer_key(request.headers(), expected_key)
    {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "A valid bearer token is required",
            "unauthorized",
        );
    }

    let target_url = format!(
        "{}{}",
        state.config.backend_url,
        request
            .uri()
            .path_and_query()
            .map(|path| path.as_str())
            .unwrap_or(request.uri().path())
    );
    let (parts, body) = request.into_parts();
    let mut headers = parts.headers;
    remove_hop_by_hop_headers(&mut headers);
    headers.remove(header::HOST);
    headers.remove(header::CONTENT_LENGTH);

    let upstream = match state
        .client
        .request(parts.method, target_url)
        .headers(headers)
        .body(reqwest::Body::wrap_stream(body.into_data_stream()))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            warn!(%error, "vLLM inference request failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                "The inference backend is unavailable",
                "backend_unavailable",
            );
        }
    };

    let status = upstream.status();
    let mut response_headers = upstream.headers().clone();
    remove_hop_by_hop_headers(&mut response_headers);
    let response_body = Body::from_stream(upstream.bytes_stream().map_err(io::Error::other));
    let mut response = Response::new(response_body);
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    response
}

fn has_bearer_key(headers: &HeaderMap, expected: &str) -> bool {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|provided| provided == expected)
}

fn remove_hop_by_hop_headers(headers: &mut HeaderMap) {
    for name in [
        header::CONNECTION,
        HeaderName::from_static("keep-alive"),
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
    ] {
        headers.remove(name);
    }
    headers.remove(HeaderName::from_static("proxy-connection"));
}
