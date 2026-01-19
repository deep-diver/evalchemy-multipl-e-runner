# evalchemy-multipl-e-runner

A small runner repo to execute **Evalchemy + MultiPL-E** inside Docker with a reproducible setup across machines.

This runner supports:

* OpenAI chat-completions (with a hot-patch for GPT-5.* stop-parameter issues)
* Anthropic Claude via the `anthropic-chat-completions` backend (patched)
* Anthropic Claude via the `curator` backend
* Google Gemini via the `google-direct` provider (OpenAI-compatible endpoint)
* **Google Gemini via Vertex AI** (better performance and rate limits)
* OpenRouter (OpenAI-compatible; Gemini etc.) via `local-chat-completions`
* MultiPL-E code override via volume mount
* Java dependency (`javatuples-1.2.jar`) via volume mount
* Results persisted to `./logs`

## Repository layout

* `run.sh`: entrypoint script (one command to run everything)
* `build.sh`: script to build local Docker image with Google GenAI SDK
* `Dockerfile`: extends base image to install `google-genai` package
* `patches/openai_completions.py`: patched lm-eval OpenAI adapter
* `patches/anthropic_completions.py`: patched lm-eval Anthropic adapter
* `patches/google_completions.py`: patched lm-eval Google GenAI adapter
* `patches/vertex_completions.py`: patched lm-eval Vertex AI adapter
* `multipl/`: MultiPL-E benchmark override folder (mounted into the container)
* `deps/javatuples-1.2.jar`: Java dependency required by MultiPL-E Java runner
* `logs/`: output directory (generated; not committed)

## Prerequisites

* Docker (Docker Desktop on macOS is OK)
* An API key for the provider you want to use:

  * OpenAI: `OPENAI_API_KEY`
  * Anthropic: `ANTHROPIC_API_KEY`
  * Google: `GOOGLE_API_KEY` or `GEMINI_API_KEY` (get one at https://aistudio.google.com/apikey)
  * OpenRouter: `OPENROUTER_API_KEY`

If you are on Apple Silicon and targeting `linux/amd64`, Docker will emulate amd64. For final “fair” runs, it is recommended to run on a Linux x86_64 server.

## Quick start

1. Clone this repo and go into it.

2. Make sure the following files exist:

* `patches/openai_completions.py`
* `deps/javatuples-1.2.jar`
* `multipl/` (directory with MultiPL-E runner code)

3. Run.

### OpenAI (default)

```sh
export OPENAI_API_KEY="your_key_here"
chmod +x run.sh
./run.sh
```

Smoke test (debug mode):

```sh
export OPENAI_API_KEY="your_key_here"
DEBUG=1 ./run.sh
```

### Anthropic Claude 

This uses the `curator` backend inside evalchemy.

```sh
export ANTHROPIC_API_KEY="your_key_here"
PROVIDER=anthropic-direct MODEL="your_claude_model_id" ./run.sh
```

Smoke test:

```sh
export ANTHROPIC_API_KEY="your_key_here"
PROVIDER=anthropic-direct MODEL="your_claude_model_id" DEBUG=1 ./run.sh
```

### Google Gemini (Gemini 2.5 Pro, 3.0 Pro, etc.)

This uses the Google GenAI SDK with the patched adapter.

First, build the local Docker image with the Google GenAI SDK:

```sh
chmod +x build.sh
./build.sh
```

Then run with your Google API key:

```sh
export GOOGLE_API_KEY="your_key_here"  # or GEMINI_API_KEY
IMAGE=deepdiver/evalchemy-multipl-e:0.1-google PROVIDER=google-direct MODEL="gemini-2.5-flash" ./run.sh
```

Available Google models:
* `gemini-2.5-flash` - Best price-performance (1M context)
* `gemini-2.5-pro` - State-of-the-art thinking model (1M context)
* `gemini-3-pro-preview` - Most intelligent, multimodal (1M context)
* `gemini-2.5-flash-lite` - Fastest, cost-efficient

Smoke test:

```sh
export GOOGLE_API_KEY="your_key_here"
IMAGE=deepdiver/evalchemy-multipl-e:0.1-google PROVIDER=google-direct MODEL="gemini-2.5-flash" DEBUG=1 ./run.sh
```

### Google Gemini via Vertex AI (Recommended for better performance)

Vertex AI provides better performance and higher rate limits than the OpenAI-compatible endpoint.

**Setup:**

1. Build the Docker image with Google GenAI SDK:
```sh
chmod +x build.sh
./build.sh
```

2. Authenticate with Google Cloud:
```sh
gcloud auth application-default login
```

Or set service account credentials:
```sh
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

3. Run with Vertex AI:
```sh
export GOOGLE_CLOUD_PROJECT="your_project_id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # Optional, default is us-central1
IMAGE=deeepdiver/evalchemy-multipl-e:0.1-google PROVIDER=vertex-direct MODEL="gemini-2.5-pro" ./run.sh
```

**Available Vertex AI models:**
* `gemini-2.5-flash` - Fast, cost-efficient
* `gemini-2.5-pro` - Most capable reasoning model
* `gemini-2.5-flash-exp` - Experimental flash model
* `gemini-2.0-flash-exp` - Earlier generation flash model

**Benefits of Vertex AI:**
* ✅ Much better performance than OpenAI-compatible endpoint
* ✅ Higher rate limits (Tier 3 gets 1500+ QPM)
* ✅ Enterprise-grade reliability
* ✅ Supports streaming and advanced features

**Note:** Vertex AI uses Google Cloud credentials (not API keys). Make sure your account has Vertex AI API enabled.

## Configuration

You can override behavior using environment variables.

### Common

* `IMAGE`
  Docker image to run.
  Default: `deepdiver/evalchemy-multipl-e:0.1`
  Note: Use `IMAGE=deeepdiver/evalchemy-multipl-e:0.1-google` for `google-direct` or `vertex-direct` providers (requires building with `./build.sh`)

* `TASKS`
  Evalchemy tasks to run.
  Default: `MultiPLE`

* `NUM_CONCURRENT`
  Number of concurrent API requests.
  Default: `8`

* `PLATFORM`
  Docker platform (useful on Apple Silicon).
  Default: `linux/amd64`

* `MULTIPLE_LANGUAGES`
  List of languages to run test MultiPL-E benchmark.
  Default: `java`

### Provider selection

* `PROVIDER`
  `openai` (default), `anthropic-direct`, `anthropic-curator`, `google-direct`, `vertex-direct`, or `openrouter`

* `MODEL`
  Provider model id:

  * OpenAI example: `gpt-5.2`
  * Anthropic example: `claude-sonnet-4-20250514`
  * Google example: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-pro-preview`
  * OpenRouter example: any model ID supported by OpenRouter

### MultiPL-E override path

By default the runner mounts `./multipl` into the container’s MultiPL-E location. If you want to use an external MultiPL-E folder instead:

```sh
export HOST_MULTIPLE_DIR="/absolute/path/to/MultiPLE"
./run.sh
```

### OpenAI adapter patch location (inside container)

If the base image changes Python/lm-eval paths, you may need to override:

```sh
export DEST_OPENAI="/path/inside/container/to/lm_eval/models/openai_completions.py"
./run.sh
```

## Output

Results are written to `./logs` on the host and are mounted from `/app/logs` inside the container.

## Troubleshooting

### “docker run requires at least 1 argument”

Your script likely broke line continuation. Use the provided `run.sh` as-is (it uses arrays and is safe to edit).

### OpenAI GPT-5.* errors about unsupported `stop`

This runner hot-patches `lm_eval/models/openai_completions.py` from `patches/openai_completions.py` and forces Python bytecode cache under `/tmp` via `PYTHONPYCACHEPREFIX=/tmp/pycache`.

### macOS (Apple Silicon) and language toolchains

Some language toolchains (e.g., Swift, Mono/C#) can behave poorly in `linux/amd64` emulation. For reliable MultiPL-E runs across all languages, prefer a Linux x86_64 host (or run those language subsets on a Linux x86_64 server).

## License / notes

* `javatuples-1.2.jar` is included to satisfy a hardcoded MultiPL-E Java dependency path. If you prefer not to commit binaries, replace it with a download script and checksum.
* This repo is intended to be a reproducible runner; the actual benchmark engine is inside the Docker image and the `multipl/` override.

