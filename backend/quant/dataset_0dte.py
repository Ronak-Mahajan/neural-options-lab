"""0DTE Rough Volatility Dataset Generation.

Generates a dataset of extremely short-dated options (0 to 12 days)
priced under the Rough Bergomi model. This data trains the specialized
`model_0dte.pt` surrogate.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from backend.quant.rough_vol import rough_bergomi_mc

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


# Bump when the simulated dynamics change in a way that invalidates an existing
# calibration. A fit is only meaningful for the kernel it was fitted under.
KERNEL_ID = "riemann_liouville_volterra_joint_v1"


def load_calibrated_dynamics() -> dict:
    """Calibrated rough-vol dynamics from calibrate.py, if they are USABLE.

    Three ways a calibration is rejected, all of which the previous version
    accepted:

    1. Wrong kernel. Parameters are only meaningful for the driver they were
       fitted under. Everything calibrated before the Riemann-Liouville fix
       was fitted against the Type-I fBm covariance, i.e. against a process
       that is not rough Bergomi, so those (eta, rho, H) carry no calibration
       information now. Such files have no `kernel` field.
    2. Market-closed quotes. The committed rough_calibration.json is stamped
       03:43 New York with quote_source "last_trade_market_closed" and
       accepted true, because the gate looked only at RMSE and bound-pinning.
       Those exact parameters were trained into the served model.
    3. Not accepted by calibrate.py's own gate.
    """
    import json
    cal_file = ARTIFACTS / "rough_calibration.json"
    defaults = {"eta": 1.5, "rho": -0.7, "H": 0.1}
    if not cal_file.exists():
        return defaults

    cal = json.loads(cal_file.read_text())
    kernel = cal.get("kernel")
    source = str(cal.get("quote_source", ""))
    if kernel != KERNEL_ID:
        print(f"REFUSING calibration: fitted under kernel {kernel!r}, this "
              f"build simulates {KERNEL_ID!r}. Re-run "
              f"'python -m backend.quant.calibrate' during market hours. "
              f"Using uncalibrated defaults.")
        return defaults
    if "closed" in source or "last_trade" in source:
        print(f"REFUSING calibration: quote_source {source!r} means the fit "
              f"used stale/market-closed prices. Using defaults.")
        return defaults
    if not cal.get("accepted"):
        print("calibration present but not accepted (quality gate); "
              "using defaults")
        return defaults
    return {k: float(cal.get(k, defaults[k])) for k in defaults}


def generate_0dte_dataset(n_samples: int = 25000, batch_size: int = 5000, seed: int = 42,
                          eta: float = 1.5, rho: float = -0.7, H: float = 0.1):
    print(f"dynamics: eta={eta:.3f} rho={rho:.3f} H={H:.3f}")
    torch.manual_seed(seed)
    
    # Latin Hypercube Sampling for parameter space
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    lhs_samples = sampler.random(n=n_samples)
    
    bounds = np.array([
        [85.0, 115.0],        # spot
        [1/252.0, 12/252.0],  # maturity
        [0.05, 0.80],         # sigma
        [0.0, 0.10],          # rate
    ])
    
    scaled = qmc.scale(lhs_samples, bounds[:, 0], bounds[:, 1])
    
    spot = torch.tensor(scaled[:, 0], dtype=torch.float32)
    maturity = torch.tensor(scaled[:, 1], dtype=torch.float32)
    sigma = torch.tensor(scaled[:, 2], dtype=torch.float32)
    rate = torch.tensor(scaled[:, 3], dtype=torch.float32)
    
    strike = torch.full((n_samples,), 100.0, dtype=torch.float32)
    
    xi = sigma ** 2
    eta_t = torch.full((n_samples,), eta, dtype=torch.float32)
    rho_t = torch.full((n_samples,), rho, dtype=torch.float32)
    
    prices = torch.zeros(n_samples, dtype=torch.float32)
    
    print(f"Generating {n_samples} 0DTE Rough Vol prices...")
    t0 = time.perf_counter()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        p = rough_bergomi_mc(
            spot=spot[i:end].to(device),
            strike=strike[i:end].to(device),
            maturity=maturity[i:end].to(device),
            xi=xi[i:end].to(device),
            eta=eta_t[i:end].to(device),
            rho=rho_t[i:end].to(device),
            rate=rate[i:end].to(device),
            n_paths=20000, 
            n_steps=50,
            H=H
        )
        prices[i:end] = p.cpu()
        print(f"  [{end}/{n_samples}] computed in {time.perf_counter() - t0:.1f}s")

    target_price = prices / strike
    features = torch.stack([spot / strike, maturity, sigma, rate], dim=1)
    
    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "dataset_0dte.pt"
    torch.save({"X": features, "y": target_price,
                "params": {"eta": eta, "rho": rho, "H": H,
                           "kernel": KERNEL_ID,
                           "n_paths": 20000, "n_steps": 50}}, out_path)
    print(f"Dataset saved to {out_path} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    generate_0dte_dataset(**load_calibrated_dynamics())
