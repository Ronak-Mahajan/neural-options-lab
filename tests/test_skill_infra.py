"""The build, CI and deploy files against the code they ship.

Render builds the Dockerfile, so these tests pin the properties that decide
what the live image contains: the torch build it installs, the files
.dockerignore leaves out (checked against every file path the serving code
can open), the token each workflow job holds, and the CI job that boots the
image. They read files only; nothing here needs Docker or the network.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
REQUIREMENTS = ROOT / "requirements.txt"
CI = ROOT / ".github" / "workflows" / "ci.yml"
RECORDER = ROOT / ".github" / "workflows" / "record_surfaces.yml"
DEPLOY = ROOT / "DEPLOY.md"

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_LINE = re.compile(r"^torch[~=<>!]")

# Every file the serving process opens lives under one of these.
RUNTIME_PATHS = [
    "backend/api/main.py",
    "backend/quant/engine.py",
    "frontend/index.html",
    "frontend/methodology.html",
    "artifacts/model.pt",
    "artifacts/model_0dte.pt",
    "artifacts/iv_surface_0dte.pt",
    "artifacts/generator.pt",
    "artifacts/hedger_gbm.pt",
    "artifacts/hedger_rbergomi_jumps.pt",
    "artifacts/eval.json",
    "artifacts/rough_calibration.json",
    "scripts/_fit_jumps_last.json",
    "requirements.txt",
]

# Modules main.py imports inside route handlers.
LAZY_MODULES = ["backend.quant.llm", "backend.quant.solve_vol",
                "backend.quant.market_data", "backend.quant.explain"]


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _torch_requirement() -> str:
    lines = [ln.strip() for ln in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
             if TORCH_LINE.match(ln.strip())]
    assert len(lines) == 1, f"expected one torch line in requirements.txt, got {lines}"
    return lines[0]


def _dockerignore_patterns() -> list[str]:
    out = []
    for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def _ignored(path: str, patterns: list[str]) -> bool:
    """A path is left out when it, or any directory above it, matches a
    pattern. No pattern here uses `!` re-inclusion, which the test asserts."""
    parts = path.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    for pat in patterns:
        p = pat.rstrip("/")
        if any(fnmatch.fnmatchcase(pre, p) for pre in prefixes):
            return True
    return False


# ---- torch: one pin, the CPU build, everywhere it is installed ----

def test_requirements_pin_torch_to_one_bounded_minor():
    spec = SpecifierSet(_torch_requirement()[len("torch"):])
    assert Version("2.11.0+cpu") in spec
    assert Version("2.11.9+cpu") in spec
    assert Version("2.12.0+cpu") not in spec
    assert Version("2.14.0+cpu") not in spec


@pytest.mark.parametrize("path", [DOCKERFILE, CI, RECORDER], ids=lambda p: p.name)
def test_every_torch_install_reads_the_pin_and_uses_the_cpu_index(path):
    text = path.read_text(encoding="utf-8")
    assert "grep -E '^torch[~=<>!]' requirements.txt" in text
    assert f"--index-url {CPU_INDEX}" in text
    # The requirements install carries the installed torch as a constraint,
    # so it can keep the CPU build but never replace it.
    assert re.search(r"pip install -r requirements\.txt -c \S*torch-pin\.txt", text)
    # No unversioned torch install anywhere.
    assert not re.search(r"pip install\s+torch(\s|$)", text)


def test_pin_extractor_yields_the_requirement_spec():
    """The shell `grep -E '^torch[~=<>!]'` the build runs, applied in Python."""
    assert _torch_requirement() == "torch~=2.11.0"


@pytest.mark.parametrize("path", [DOCKERFILE, CI], ids=lambda p: p.name)
def test_build_fails_unless_torch_is_the_cpu_build(path):
    text = path.read_text(encoding="utf-8")
    assert "v.endswith('+cpu')" in text
    for prefix in ("'nvidia-'", "'cuda-'", "'triton'"):
        assert prefix in text


def test_dockerfile_orders_cpu_torch_before_requirements():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert text.index(CPU_INDEX) < text.index("pip install -r requirements.txt")
    assert text.index("pip install -r requirements.txt") < text.index("v.endswith('+cpu')")


# ---- .dockerignore: excluded paths are ones the server never opens ----

def test_dockerignore_keeps_every_runtime_path():
    patterns = _dockerignore_patterns()
    assert not any(p.startswith("!") for p in patterns)
    kept = [p for p in RUNTIME_PATHS if not _ignored(p, patterns)]
    assert kept == RUNTIME_PATHS, (
        f"left out of the image: {sorted(set(RUNTIME_PATHS) - set(kept))}")
    for p in RUNTIME_PATHS:
        assert (ROOT / p).exists(), p


def test_dockerignore_leaves_out_data_docs_and_tests():
    patterns = _dockerignore_patterns()
    for p in ("data/surfaces/deribit/x.json.gz", "docs/heston_reference.md",
              "tests/test_api.py", ".github/workflows/ci.yml"):
        assert _ignored(p, patterns), p


def _code_strings(path: Path) -> list[str]:
    """String constants in a module's code, docstrings excluded."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant):
                docs.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def _serving_module_names() -> list[str]:
    """The backend modules the server imports, from a fresh interpreter: the
    pytest process has already imported whatever other test files need, so
    its own sys.modules says nothing about the serving closure."""
    code = ("import importlib, json, sys\nimport backend.api.main\n"
            f"for n in {LAZY_MODULES!r}: importlib.import_module(n)\n"
            "print(json.dumps(sorted(n for n in sys.modules if n.startswith('backend'))))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_serving_code_never_names_an_excluded_directory():
    names = _serving_module_names()
    assert "backend.api.main" in names
    assert "backend.trading.quoter" not in names
    excluded = {p.rstrip("/") for p in _dockerignore_patterns() if p.endswith("/")}
    hits = []
    for name in names:
        origin = importlib.util.find_spec(name).origin
        if not origin or not origin.endswith(".py"):
            continue
        for s in _code_strings(Path(origin)):
            if s.rstrip("/") in excluded:
                hits.append((name, s))
    assert not hits, f"serving code names a directory the image leaves out: {hits}"


# ---- workflow tokens: read-only unless the job pushes ----

@pytest.mark.parametrize("path,writers", [
    (CI, {"failure-log"}),
    (RECORDER, {"publish", "failure-log"}),
], ids=["ci", "record_surfaces"])
def test_write_token_only_on_jobs_that_push(path, writers):
    wf = _workflow(path)
    assert wf["permissions"] == {"contents": "read"}
    for name, job in wf["jobs"].items():
        perms = job.get("permissions", {})
        run = "\n".join(s.get("run", "") for s in job["steps"])
        if name in writers:
            assert perms == {"contents": "write"}, name
            assert "git push" in run, name
            assert "pip install" not in run, name
        else:
            assert perms.get("contents", "read") == "read", name
            assert "git push" not in run, name
            for step in job["steps"]:
                if str(step.get("uses", "")).startswith("actions/checkout"):
                    assert step.get("with", {}).get("persist-credentials") is False, name


def test_failure_log_never_runs_on_main_or_pull_requests():
    job = _workflow(CI)["jobs"]["failure-log"]
    assert "github.ref != 'refs/heads/main'" in job["if"]
    assert "github.event_name == 'push'" in job["if"]


# ---- CI builds and boots the image Render deploys ----

def test_ci_boots_the_image_at_starter_limits():
    job = _workflow(CI)["jobs"]["docker"]
    run = "\n".join(s.get("run", "") for s in job["steps"])
    assert "docker build" in run
    assert "--memory=512m" in run and "--cpus=0.5" in run
    assert "/api/health" in run and "/api/price" in run
    assert job.get("timeout-minutes", 360) <= 30


def test_render_builds_the_dockerfile_with_the_health_check():
    render = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    svc = render["services"][0]
    assert svc["runtime"] == "docker"
    assert svc["healthCheckPath"] == "/api/health"


# ---- DEPLOY.md leads with the path that works ----

def test_deploy_leads_with_render_and_has_operations():
    text = DEPLOY.read_text(encoding="utf-8")
    assert text.index("## Render") < text.index("Hugging Face")
    for heading in ("## Operations", "### Which commit is live", "### Roll back",
                    "### Re-pin a model artifact", "### Out-of-memory restart loop",
                    "### Upstream data outage"):
        assert heading in text, heading
    assert "0.5 CPU" in text
    assert "full CPU share" not in text


def test_deploy_reads_in_product_voice():
    text = DEPLOY.read_text(encoding="utf-8")
    assert "—" not in text
    for phrase in ("already done", "how this note got written", "no longer",
                   "honest"):
        assert phrase not in text.lower(), phrase


def test_deploy_repin_command_names_a_real_test():
    text = DEPLOY.read_text(encoding="utf-8")
    m = re.search(r"-k (\w+)", text)
    assert m, "DEPLOY.md re-pin step names no test"
    reg = (ROOT / "tests" / "test_regression.py").read_text(encoding="utf-8")
    assert f"def test_{m.group(1)}(" in reg
