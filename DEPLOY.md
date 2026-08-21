# Deploying the dashboard

The whole thing is one FastAPI service that serves both the API and the
dashboard from the same origin, so hosting it anywhere that can run the
Docker image gives you a public link with no extra configuration. The
trained model checkpoints are in the repo, so the container is ready to
serve as soon as it builds.

The one thing that matters for hosting is memory. PyTorch plus the models
needs more headroom than the smallest free tiers comfortably give, so the
notes below are honest about which option fits.

## Option 1: Hugging Face Spaces (no longer free — needs PRO)

Update, August 2026: Hugging Face now requires a PRO subscription to host
Docker (and Gradio) Spaces even on the basic CPU tier — repo creation fails
with "requires a PRO subscription" on a free account, which is how this note
got written. With PRO, the tier is 2 vCPUs and 16 GB of RAM, which is plenty
for PyTorch and the Monte Carlo endpoints. You get a public URL like
`https://huggingface.co/spaces/<your-username>/neural-options-lab`.

A Space is its own git repository. To deploy from the command line:

```bash
hf auth login          # paste a write token from huggingface.co/settings/tokens
hf repo create neural-options-lab --repo-type space --space-sdk docker
```

Creating the Space with the Docker SDK generates a `README.md` whose header
tells Spaces how to run the container. Push the project into that Space repo,
keeping the generated header. The header needs to contain at least:

```yaml
---
title: Neural Options Lab
sdk: docker
app_port: 8000
---
```

Once the code is pushed, the Space builds the Docker image and serves it. The
first build takes a few minutes.

## Option 2: Render (Docker) — the free path that works, and where the live demo runs

The repo includes `render.yaml`, so Render can deploy it directly:

1. Push the repo to GitHub (already done).
2. On render.com, choose New, then Blueprint, and point it at the repo.
3. Render reads `render.yaml`, builds the Docker image, and deploys.

Honest note on the free plan: it gives 512 MB of RAM and spins the service
down when idle, so the first request after a pause is slow while PyTorch
loads. The pricing and hedging tabs work, but the heavier Monte Carlo
endpoints (the convergence and benchmark charts) can be tight on memory.
A small paid instance removes both problems. Hugging Face Spaces is the
easier free choice for this app.

## Option 3: Any server you control

On a VM or VPS with Docker:

```bash
docker build -t neural-options-lab .
docker run -p 80:8000 neural-options-lab
```

Or without Docker, behind a reverse proxy such as Nginx or Caddy:

```bash
pip install -r requirements.txt
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

## Optional environment variables

- `GROQ_API_KEY`: enables the language-model risk report. Free key at
  console.groq.com. Without it, the app streams a deterministic offline
  summary instead.
- `PORT`: the port to listen on. Most hosts set this automatically; the
  container defaults to 8000.

## Sharing on your local network

If you only need it reachable from other machines on the same network, no
hosting is required. Run it bound to all interfaces and share your machine's
local IP address:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Then open `http://<your-local-ip>:8000` from another device.
