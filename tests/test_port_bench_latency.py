"""Checks on docs/approximation_benchmark.json and its rendered page.

The benchmark's seeded prices reproduce digit for digit across runs; timings
do not. These tests therefore recompute every summary statistic from the
stored per-point prices, tie the artifact to the served checkpoint, and check
that docs/approximation_benchmark.md is the render of the artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.quant.evaluate import SERVED_CHECKPOINT, checkpoint_fingerprint
from scripts import benchmark_approximations as bench

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "docs" / "approximation_benchmark.json"
MD_PATH = ROOT / "docs" / "approximation_benchmark.md"


@pytest.fixture(scope="module")
def payload():
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def test_artifact_names_served_checkpoint(payload):
    """Both sections were measured on the checkpoint being served."""
    fp = checkpoint_fingerprint(SERVED_CHECKPOINT)
    for key in ("box_lhs", "grid"):
        assert payload[key]["checkpoint"]["sha256"] == fp["sha256"]


def test_box_summary_recomputes(payload):
    """Stored box statistics equal what the stored prices imply."""
    box = payload["box_lhs"]
    stats = bench.box_statistics(box)
    for name, stored in box["summary"].items():
        for k, v in stored.items():
            assert stats["summary"][name][k] == pytest.approx(v, rel=1e-12), \
                (name, k)
    assert stats["reference_se_bps"]["rms"] == pytest.approx(
        box["reference_se_bps"]["rms"], rel=1e-12)
    assert stats["by_sigma"] == box["by_sigma"]
    assert stats["by_reference_price"] == box["by_reference_price"]


def test_box_ratio_interval_recomputes(payload):
    """The headline RMSE ratio and its bootstrap interval come from the
    stored prices and the seed recorded in the protocol."""
    box = payload["box_lhs"]
    names = list(box["methods"])
    ref = np.array(box["reference"]["price"])
    err = {k: np.array(box["methods"][k]["price"]) - ref for k in names[:2]}
    rng = np.random.default_rng(
        bench._point_seed(box["protocol"]["seed"], 2, 0))
    got = bench.rmse_ratio_interval(err[names[1]], err[names[0]],
                                    box["protocol"]["bootstrap_resamples"],
                                    rng)
    stored = box["rmse_ratios"]["levy_over_surrogate"]
    assert got["ratio"] == pytest.approx(stored["ratio"], rel=1e-12)
    assert got["ci95"] == pytest.approx(stored["ci95"], rel=1e-12)


def test_grid_summary_recomputes(payload):
    """Stored grid statistics equal what the stored prices imply.

    The grid summary is computed before prices are rounded to 1e-10 for
    storage; 1e-10 of a strike-100 price is 1e-8 bps, the comparison scale.
    """
    grid = payload["grid"]
    ref = np.array(grid["reference"]["price"])
    for name, stored in grid["summary"].items():
        err = ((np.array(grid["methods"][name]["price"]) - ref)
               / bench.STRIKE * 1e4)
        assert float(np.mean(np.abs(err))) == pytest.approx(
            stored["mean_abs_bps"], abs=1e-6)
        assert float(np.max(np.abs(err))) == pytest.approx(
            stored["max_abs_bps"], abs=1e-6)
        assert float(np.mean(err)) == pytest.approx(stored["bias_bps"],
                                                    abs=1e-6)


def test_page_is_render_of_artifact(payload):
    """docs/approximation_benchmark.md holds the same digits as the JSON."""
    assert MD_PATH.read_text(encoding="utf-8") == bench.build_report(payload)
