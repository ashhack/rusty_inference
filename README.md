# rusty_inference

A small, asynchronous Rust API gateway in front of [vLLM](https://docs.vllm.ai/). vLLM runs the model on the GPU and provides its OpenAI-compatible inference API; Rust handles the client-facing HTTP API, health checks, optional bearer-key protection, and streaming proxy without buffering generated tokens.

> vLLM is the inference engine, not a Rust library. Its CUDA/Python runtime is packaged as the official vLLM container; the gateway and API layer are Rust.

## What is included

- OpenAI-compatible `/v1/*` passthrough, including chat completions, completions, embeddings, and other endpoints enabled by the vLLM version/model.
- Streaming responses (for example `stream: true`) are forwarded as they arrive.
- `/health` checks the Rust process; `/ready` checks the vLLM backend; `/metrics` proxies vLLM's Prometheus metrics.
- Optional `API_KEY` bearer-token protection on `/v1/*` and `/metrics`.
- Docker Compose deployment with persistent Hugging Face cache, local model mounts, GPU reservation, and vLLM health checks.
- Tests covering proxying, streaming headers, readiness, backend failures, and API-key enforcement.

## Requirements

- Docker Engine with Docker Compose v2.
- NVIDIA GPU, compatible host driver, and NVIDIA Container Toolkit for GPU inference.
- A model that fits the available GPU memory. CPU-only use is possible with a separately configured vLLM image/runtime, but is not the default deployment.

## Run with Docker Compose

```bash
cp .env.example .env
# Edit MODEL and other settings in .env if needed.
docker compose up --build
```

By default, the stack downloads `Qwen/Qwen3-0.6B` from Hugging Face into the persistent `hf-cache` volume and serves it as `assistant` on `http://localhost:8080`. First startup can take several minutes while model files download and load. Choose a model or quantized variant that fits your GPU.

### Use a Hugging Face model ID

Set `MODEL` in `.env` to a model repo ID, for example:

```dotenv
MODEL=Qwen/Qwen3-0.6B
SERVED_MODEL_NAME=assistant
HF_TOKEN=
```

For gated/private repositories, set `HF_TOKEN` to a Hugging Face access token with permission to download that model. Keep `.env` private; it is ignored by git.

### Use a model already downloaded locally

Set `MODEL_DIR` to the host directory containing the model folder and `MODEL` to its path inside the container:

```dotenv
MODEL_DIR=/absolute/path/to/models
MODEL=/models/my-local-model
SERVED_MODEL_NAME=assistant
```

The model directory is mounted read-only. The model folder should contain the standard Hugging Face model files (configuration, tokenizer, and weights). For example, `/absolute/path/to/models/my-local-model/config.json` must exist on the host.

Set `TENSOR_PARALLEL_SIZE` to the number of GPUs across which the model should be sharded. Ensure that the model and vLLM configuration support the selected number.

## API examples

Without a configured gateway key:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "assistant",
    "messages": [{"role": "user", "content": "Explain Rust ownership in one sentence."}],
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  }'
```

With `API_KEY` set in `.env`, add:

```bash
set -a
. ./.env
set +a
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"assistant","messages":[{"role":"user","content":"Hi"}]}'
```

The exact model name in requests is the `SERVED_MODEL_NAME` value. Other supported endpoints are forwarded unchanged, for example:

```bash
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"assistant","input":"A short example sentence."}'
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/metrics
```

Availability of embeddings, audio, pooling, reranking, and other vLLM endpoints depends on the selected vLLM release and model. This gateway forwards `/v1/*` requests without changing their JSON schema.

## Run the Rust gateway locally

If vLLM is already running on port `8000`, install Rust (edition 2024 support) and run:

```bash
VLLM_BASE_URL=http://127.0.0.1:8000 cargo run --release
```

Configuration:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | Rust gateway listen port |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | Base URL of the vLLM OpenAI-compatible server |
| `API_KEY` | unset | Optional bearer token required by the gateway |
| `CONNECT_TIMEOUT_SECS` | `5` | Timeout for establishing backend connections |

Inference response bodies do not have a total-duration timeout, so long generations and SSE streams can complete normally.

## Tests

```bash
cargo fmt --check
cargo test
```

Tests use a local mock HTTP backend and do not download a model or require a GPU.

## Deployment notes

- Only the Rust gateway port is published by Compose; vLLM is reachable only on the private Compose network.
- Set a strong `API_KEY` before exposing the gateway beyond a trusted local network. Terminate TLS at a reverse proxy/load balancer for remote clients.
- Pin `VLLM_IMAGE` to a tested version for reproducible production deployments instead of tracking `latest`.
- Model performance depends on GPU, vLLM version, model, quantization, context length, and concurrency; the gateway does not claim a fixed throughput.
