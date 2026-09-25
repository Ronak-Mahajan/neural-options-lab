# Deploying the dashboard

The whole thing is one FastAPI service that serves both the API and the
dashboard from the same origin, so hosting it anywhere that can run the
Docker image gives you a public link with no extra configuration. The
trained model checkpoints are in the repo, so the container is ready to
serve as soon as it builds.

The one thing that matters for hosting is memory. PyTorch plus the models
needs more headroom than the smallest free tiers comfortably give, so the
notes below say which option fits.

## Render (Docker): where the live demo runs

The repo includes `render.yaml`, so Render can deploy it directly:

1. Push the repo to GitHub.
2. On render.com, choose New, then Blueprint, and point it at the repo.
3. Render reads `render.yaml`, builds the Docker image, and deploys. Every
   push to `main` deploys again.

The live demo at neural-options-lab.onrender.com runs on Render's Starter
plan (`plan: starter` in `render.yaml`): always on, so there is no wake-up
delay on first load, with 512 MB of RAM and 0.5 CPU.

The service also fits the free plan if you change `plan` to `free`. That
plan gives the same 512 MB but 0.1 CPU, and spins the service down when
idle, so the first request after a pause is slow while PyTorch loads. The
app is sized to fit either way: PyTorch's import is a fixed ~300 MB floor,
so both Monte Carlo engines simulate in fixed-size path blocks (peak ~40 MB
per request regardless of the path count), the API admits one simulation
job at a time and queues the rest, and the dashboard loads its heavy panels
sequentially. What the free tier costs you is speed (at 0.1 CPU the
convergence chart takes several seconds to fill) and the idle spin-down;
the paid instance removes both.

### What the image contains

The Dockerfile installs PyTorch from the CPU-only index at the version
`requirements.txt` pins, then installs `requirements.txt` with that exact
build as a constraint, so the second step cannot swap in the CUDA wheel from
PyPI. The build then fails unless the installed torch is the `+cpu` build
with no CUDA, NVIDIA or Triton packages present, and a passing build prints
a line of the form `torch 2.11.0+cpu CPU build, no GPU packages`.
`.dockerignore` leaves out what the server never opens (`data/`, `docs/`,
`tests/`, CI files); the API reads only `backend/`, `frontend/`,
`artifacts/` and `scripts/_fit_jumps_last.json`.

CI builds the same Dockerfile on every push that CI runs on (`main` and
the `fix/**` and `rebuild/**` branches) and on pull requests, boots it with
the Starter plan's limits (`--memory=512m --cpus=0.5`), and checks
`/api/health`, the two pages and one `/api/price` request, so an image that
fails to build, boot or price fails CI before Render sees it.

## Operations

Render's health check calls `/api/health` (set in `render.yaml`). Per
Render's documentation, a check passes on a 2xx or 3xx answer within five
seconds; after 15 seconds of consecutive failures Render stops routing
traffic to the instance, and after 60 seconds it restarts it. A new deploy
that never passes within 15 minutes is cancelled and the previous instances
keep serving.

### Which commit is live

- Render dashboard, service, Events: each deploy lists the commit it built.
  Compare it with `git rev-parse origin/main`.
- Render sets `RENDER_GIT_COMMIT` to the deployed commit, so
  `echo $RENDER_GIT_COMMIT` in the service's Shell tab gives the same answer.
- `curl -s https://neural-options-lab.onrender.com/api/health` answers
  `"status": "ok"` with `model_loaded`, `iv_surface_loaded` and the loaded
  hedgers when the instance can price. `/api/model-info` returns the served
  checkpoint's metadata and evaluation report.

### Smoke test after a deploy

```bash
BASE=https://neural-options-lab.onrender.com
curl -sf "$BASE/api/health"
curl -sf -X POST "$BASE/api/price" -H 'Content-Type: application/json' \
  -d '{"spot":100,"strike":100,"maturity":1,"sigma":0.25,"rate":0.04,"option_type":"call","mc_paths":1000}'
```

Both must return 200. The second returns a JSON body with `nn.price` and
`mc.price`.

### Roll back

1. Render dashboard, service, Deploys: pick the last good deploy and click
   Rollback. Render reuses that deploy's build, so nothing is rebuilt.
2. A rollback from the dashboard turns off automatic deploys. Turn them back
   on in the service's Settings once the fix is on `main`, or the next push
   will not deploy.
3. Confirm the commit (see above) and run the smoke test.

Without the dashboard, `git revert` the bad commit and push to `main`; that
deploys through a normal build.

### Re-pin a model artifact

`artifacts/eval.json` records the SHA-256 of the `artifacts/model.pt` it was
measured against, and the regression suite fails when the two disagree.

1. Replace the checkpoint (`backend/quant/train.py` documents how the served
   model is promoted).
2. `python -m backend.quant.evaluate` rewrites `artifacts/eval.json`,
   including `checkpoint.sha256`.
3. `sha256sum artifacts/model.pt` must match `checkpoint.sha256` in
   `artifacts/eval.json`, and
   `python -m pytest tests/test_regression.py -k eval_report_matches_the_served_checkpoint`
   must pass.
4. Commit the checkpoint and the report together, push, and check
   `/api/model-info` on the live service once the deploy is live.

### Out-of-memory restart loop

Symptoms: Events show the instance restarting repeatedly, or the deploy
never passes the health check.

1. Roll back first; the site keeps serving while you look.
2. Open the failing deploy's build log and confirm the torch line reads
   `+cpu ... no GPU packages`. The build fails when it does not, so a
   finished build always carries the CPU wheel.
3. Reproduce locally at the plan's limits:
   ```bash
   docker build -t neural-options-lab .
   docker run --memory=512m --memory-swap=512m --cpus=0.5 -p 8000:8000 neural-options-lab
   ```
   then replay the request that preceded the restart and watch
   `docker stats`. A container killed for memory shows `OOMKilled=true`
   in `docker inspect`.

### Upstream data outage (Yahoo Finance or Deribit)

- Pricing, surfaces, hedging and the stream use no live feed. Only the
  ticker loader (`/api/market/{ticker}`, Yahoo Finance through yfinance)
  does, and it answers 400 with the feed's error while Yahoo is down; the
  rest of the dashboard is unaffected, so the service needs no action.
- The surface recorder (`.github/workflows/record_surfaces.yml`) runs in
  GitHub Actions, not on Render. A failed SPY capture is logged as
  `failed: ...` in `equity/_spy_status.log` on the `surfaces` branch and
  does not fail the run. A Deribit failure fails that run, and the next
  scheduled run two hours later tries again. Check Deribit's API with
  `curl -s https://www.deribit.com/api/v2/public/test`.

## Any server you control

On a VM or VPS with Docker:

```bash
docker build -t neural-options-lab .
docker run -p 80:8000 neural-options-lab
```

Or without Docker, behind a reverse proxy such as Nginx or Caddy. On Linux,
install the CPU build of PyTorch first; the default Linux wheel on PyPI
pulls the CUDA runtime, which this service never uses:

```bash
pip install "torch~=2.11.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

## Alternative: Hugging Face Spaces (requires PRO)

As of August 2026, Hugging Face requires a PRO subscription to host Docker
(and Gradio) Spaces, even on the basic CPU tier; repo creation fails with
"requires a PRO subscription" on a free account. With PRO, the tier is 2
vCPUs and 16 GB of RAM, which is plenty for PyTorch and the Monte Carlo
endpoints. You get a public URL like
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
