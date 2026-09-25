"""Checks that pin frontend/methodology.html to the artifacts it quotes.

  * Figure captions use a documented ink token from styles.css, and that
    token clears 4.5:1 on the page background.
  * Every "N-path references" count on the page equals eval.json's
    ref_paths, and the training sentence matches model.pt's meta.
  * The deep-hedging captions and error-bar prose say what the bars are over:
    seeded path blocks and one trained policy, whose training seed matches the
    meta of every simulated-measure checkpoint.
  * The control-variate factor is measured here on the contract the page
    names, and the contract dependence the page states holds.
  * The page carries no repository-history vocabulary.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.quant.monte_carlo import price_asian_mc

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "frontend" / "methodology.html"
CSS = ROOT / "frontend" / "styles.css"
ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="module")
def raw() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def text(raw: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", raw, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", html.unescape(body))


def _rgba(value: str) -> tuple[float, float, float, float]:
    value = value.strip()
    if value.startswith("#"):
        h = value[1:]
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0)
    nums = [float(x) for x in re.findall(r"[\d.]+", value)]
    return (nums[0], nums[1], nums[2], nums[3] if len(nums) > 3 else 1.0)


def _luminance(rgb: tuple[float, float, float]) -> float:
    def lin(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _root_tokens() -> dict[str, str]:
    css = CSS.read_text(encoding="utf-8")
    root = re.search(r":root\s*\{(.*?)\n\}", css, flags=re.S).group(1)
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", root))


def test_figcaption_uses_ink_token_with_aa_contrast(raw: str):
    rule = re.search(r"figure\.fig figcaption\s*\{([^}]*)\}", raw).group(1)
    colour = re.search(r"(?<![-\w])color:\s*([^;]+);", rule).group(1).strip()
    m = re.fullmatch(r"var\((--ink(?:-hi|-mid|-low)?)\)", colour)
    assert m, f"figcaption colour is not a documented ink token: {colour}"
    tokens = _root_tokens()
    fg = _rgba(tokens[m.group(1)])
    bg = _rgba(tokens["--bg-0"])
    a = fg[3]
    blended = tuple(a * f + (1 - a) * b for f, b in zip(fg[:3], bg[:3]))
    l1, l2 = sorted((_luminance(blended), _luminance(bg[:3])), reverse=True)
    assert (l1 + 0.05) / (l2 + 0.05) >= 4.5
    img = re.search(r"figure\.fig img\s*\{([^}]*)\}", raw).group(1)
    assert "var(--line)" in img and "var(--bg-0)" in img


def test_reference_path_counts_match_artifacts(text: str):
    ref_paths = json.loads((ARTIFACTS / "eval.json").read_text())["ref_paths"]
    bench = json.loads((ROOT / "docs" / "approximation_benchmark.json")
                       .read_text(encoding="utf-8"))
    # The held-out evaluation (eval.json): headline sentence and table caption.
    assert f"1.3 basis points of strike against {ref_paths:,}-path references" in text
    assert f"600 test points against {ref_paths:,}-path references" in text
    # The benchmark tables quote their own artifact's reference counts.
    lhs = bench["box_lhs"]["protocol"]["ref_paths"]
    grid = bench["grid"]["protocol"]["ref_paths"]
    assert f"300 Latin-hypercube points (seed 1992), versus {lhs:,}-path" in text
    assert f"r = 0.04, versus {grid:,}-path control-variate references" in text
    assert f"the {grid:,}-path reference behind the benchmark" in text


def test_training_sentence_matches_model_meta(text: str):
    meta = torch.load(ARTIFACTS / "model.pt", map_location="cpu",
                      weights_only=True)["meta"]
    expected = (f"trained on {meta['n_samples']:,} labels of "
                f"{meta['mc_paths_per_label']:,} paths each for "
                f"{meta['epochs']} epochs")
    assert expected in text


def test_hedging_error_bars_are_scoped(raw: str, text: str):
    seeds = {
        name: torch.load(ARTIFACTS / name, map_location="cpu",
                         weights_only=True)["meta"]["seed"]
        for name in ("hedger_gbm.pt", "hedger_rbergomi.pt",
                     "hedger_rbergomi_jumps.pt")
    }
    assert len(set(seeds.values())) == 1, seeds
    seed = next(iter(seeds.values()))
    assert f"all use training seed {seed}" in text
    assert f"one trained policy (training seed {seed})" in text
    assert f"one training run (seed {seed})" in text
    # "5 seeds x 3,000 paths" reads as five training seeds; the path blocks
    # are named as such instead.
    assert not re.search(r"\b5 seeds\b", text)
    assert text.count("5 seeded path blocks") >= 2
    assert "They do not include the spread from retraining the network" in text
    # The retraining spread the page quotes is the committed seed study's.
    rows = json.loads((ROOT / "docs" / "deep_hedging_seed_spread.json")
                      .read_text(encoding="utf-8"))["rows"]
    gaps = [-r["deep_minus_delta_bp"] for r in rows if r["cost"] == 0.005]
    assert len({r["train_seed"] for r in rows}) == 4 and len(gaps) == 8
    assert f"between {min(gaps):.0f} and {max(gaps):.0f} bp of strike" in text


def test_real_history_checkpoint_named_by_what_it_is(text: str):
    doc = (ROOT / "docs" / "hedging_real_paths.md").read_text(encoding="utf-8")
    assert "`hedger.pt` is the GAN-measure policy" in doc
    assert "hedger.pt checkpoint (a policy trained on the GAN market simulator" in text
    for word in ("retired", "legacy", "byte-identical"):
        assert word not in text.lower(), word


def _cv_ratio(S, K, T, sigma, r, n_paths=5_000, reps=300):
    kw = dict(n_paths=n_paths, n_steps=50)
    plain = [price_asian_mc(S, K, T, sigma, r, seed=i, control_variate=False,
                            **kw).price for i in range(reps)]
    cv = [price_asian_mc(S, K, T, sigma, r, seed=i, control_variate=True,
                         **kw).price for i in range(reps)]
    return float(np.std(plain, ddof=1) / np.std(cv, ddof=1))


def test_control_variate_factor_on_the_named_contract(text: str):
    assert ("On an at-the-money one-year contract at 20% volatility and a 5% rate"
            in text)
    assert "300 seeded replications with antithetic sampling in both arms" in text
    # The page quotes about 24x on this contract. 300 replications leave a few
    # percent of noise on each standard deviation.
    atm = _cv_ratio(100.0, 100.0, 1.0, 0.20, 0.05)
    assert 20.0 < atm < 30.0, atm


def _vr_range(rows, **contract) -> tuple[float, float]:
    vals = [r["sd_ratio"] for r in rows
            if all(r[k] == v for k, v in contract.items())]
    assert vals, contract
    return min(vals), max(vals)


def test_control_variate_factor_depends_on_contract(text: str):
    """Every factor the page quotes is the range over the committed
    replication sets in docs/variance_reduction.json."""
    rows = json.loads((ROOT / "docs" / "variance_reduction.json")
                      .read_text(encoding="utf-8"))["rows"]
    lo, hi = _vr_range(rows, S=100, K=100, T=1.0, sigma=0.2, n_paths=5000)
    assert f"{lo:.0f}x to {hi:.0f}x at 5,000 paths" in text
    lo, hi = _vr_range(rows, S=100, K=100, T=1.0, sigma=0.2, n_paths=20000)
    assert f"{lo:.0f}x to {hi:.0f}x at 20,000" in text
    lo, hi = _vr_range(rows, S=100, K=100, T=0.25, sigma=0.2)
    assert f"{lo:.0f}x to {hi:.0f}x on a three-month at-the-money contract at 20% volatility" in text
    others = [r["sd_ratio"] for r in rows
              if (r["S"], r["T"], r["sigma"]) in ((100, 1.0, 0.6), (80, 0.25, 0.3), (130, 2.0, 0.7))]
    assert len(others) == 6
    assert f"{min(others):.0f}x to {max(others):.0f}x at three other points" in text
    assert "away from the money" not in text
