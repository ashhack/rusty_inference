use std::{net::SocketAddr, time::Duration};

use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::Request,
    http::{StatusCode, header},
    response::IntoResponse,
    routing::any,
};
use rusty_inference::{api, config::AppConfig};
use serde_json::{Value, json};
use tokio::net::TcpListener;

async fn spawn(app: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    address
}

fn config(backend: SocketAddr, api_key: Option<&str>) -> AppConfig {
    AppConfig {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        backend_url: format!("http://{backend}"),
        api_key: api_key.map(str::to_owned),
        connect_timeout: Duration::from_secs(1),
    }
}

async fn echo_backend(request: Request<Body>) -> impl IntoResponse {
    let path = request.uri().path().to_owned();
    let query = request.uri().query().unwrap_or_default().to_owned();
    let method = request.method().to_string();
    let body = to_bytes(request.into_body(), 1024 * 1024).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    Json(json!({ "path": path, "query": query, "method": method, "body": body }))
}

#[tokio::test]
async fn proxies_openai_compatible_requests() {
    let backend = spawn(Router::new().fallback(any(echo_backend))).await;
    let gateway = spawn(api::router(config(backend, None)).unwrap()).await;
    let response = reqwest::Client::new()
        .post(format!("http://{gateway}/v1/chat/completions?stream=true"))
        .json(&json!({ "model": "assistant", "messages": [{ "role": "user", "content": "Hi" }] }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let payload: Value = response.json().await.unwrap();
    assert_eq!(payload["path"], "/v1/chat/completions");
    assert_eq!(payload["query"], "stream=true");
    assert_eq!(payload["method"], "POST");
    assert!(
        payload["body"]
            .as_str()
            .unwrap()
            .contains("\"model\":\"assistant\"")
    );
}

#[tokio::test]
async fn requires_bearer_key_when_configured() {
    let backend = spawn(Router::new().fallback(any(echo_backend))).await;
    let gateway = spawn(api::router(config(backend, Some("secret"))).unwrap()).await;

    let response = reqwest::get(format!("http://{gateway}/v1/models"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let payload: Value = response.json().await.unwrap();
    assert_eq!(payload["error"]["code"], "unauthorized");
}

#[tokio::test]
async fn accepts_configured_bearer_key() {
    let backend = spawn(Router::new().fallback(any(echo_backend))).await;
    let gateway = spawn(api::router(config(backend, Some("secret"))).unwrap()).await;

    let response = reqwest::Client::new()
        .get(format!("http://{gateway}/v1/models"))
        .bearer_auth("secret")
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn health_check_does_not_depend_on_backend() {
    let gateway = spawn(api::router(config("127.0.0.1:1".parse().unwrap(), None)).unwrap()).await;

    let response = reqwest::get(format!("http://{gateway}/health"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn readiness_checks_the_inference_backend() {
    let backend = spawn(Router::new().fallback(any(echo_backend))).await;
    let gateway = spawn(api::router(config(backend, None)).unwrap()).await;

    let response = reqwest::get(format!("http://{gateway}/ready"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let payload: Value = response.json().await.unwrap();
    assert_eq!(payload["status"], "ready");
}

#[tokio::test]
async fn returns_gateway_error_when_backend_is_down() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend = listener.local_addr().unwrap();
    drop(listener);
    let gateway = spawn(api::router(config(backend, None)).unwrap()).await;

    let response = reqwest::get(format!("http://{gateway}/v1/models"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let payload: Value = response.json().await.unwrap();
    assert_eq!(payload["error"]["code"], "backend_unavailable");
}

#[tokio::test]
async fn forwards_backend_health_status() {
    let backend = spawn(Router::new().route(
        "/health",
        any(|| async { (StatusCode::SERVICE_UNAVAILABLE, "loading") }),
    ))
    .await;
    let gateway = spawn(api::router(config(backend, None)).unwrap()).await;

    let response = reqwest::get(format!("http://{gateway}/ready"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn preserves_upstream_content_type() {
    let backend = spawn(Router::new().fallback(any(|| async {
        (
            [(header::CONTENT_TYPE, "text/event-stream")],
            "data: [DONE]\n\n",
        )
    })))
    .await;
    let gateway = spawn(api::router(config(backend, None)).unwrap()).await;

    let response = reqwest::get(format!("http://{gateway}/v1/chat/completions"))
        .await
        .unwrap();

    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
    assert_eq!(response.text().await.unwrap(), "data: [DONE]\n\n");
}
