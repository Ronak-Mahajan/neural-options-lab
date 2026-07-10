"""Rough Volatility (Rough Bergomi) Monte Carlo Engine for 0DTE Options.

Implements a fractional Brownian motion (fBm) generator and a fast 
PyTorch-based Monte Carlo pricer for the Rough Bergomi model.
This is used to generate the ground-truth dataset for training the 
0DTE Neural Surrogate model.
"""

import math
import torch

def generate_fbm_covariance(n_steps: int, dt: float, H: float, device: torch.device) -> torch.Tensor:
    """Exact covariance matrix for fractional Brownian motion."""
    t = torch.arange(1, n_steps + 1, dtype=torch.float32, device=device) * dt
    ti_2H = t ** (2 * H)
    tj_2H = t ** (2 * H)
    ti_tj_2H = torch.abs(t.unsqueeze(1) - t.unsqueeze(0)) ** (2 * H)
    
    C = 0.5 * (ti_2H.unsqueeze(1) + tj_2H.unsqueeze(0) - ti_tj_2H)
    return C

def rough_bergomi_mc(spot: torch.Tensor, strike: torch.Tensor, maturity: torch.Tensor,
                     xi: torch.Tensor, eta: torch.Tensor, rho: torch.Tensor, rate: torch.Tensor,
                     n_paths: int = 50000, n_steps: int = 50, H: float = 0.1,
                     seed: int | None = None,
                     return_std_error: bool = False):
    """Prices European Call options using the Rough Bergomi model.
    
    Args:
        spot, strike, maturity: (B,) tensors for standard option parameters
        xi: (B,) initial forward variance (similar to sigma^2)
        eta: (B,) volatility of volatility
        rho: (B,) correlation between spot and variance
        rate: (B,) risk-free rate
        n_paths: number of MC paths per contract
        n_steps: number of time steps (high resolution for intraday)
        H: Hurst parameter (H < 0.5 is rough, typically ~0.1 in markets)
        
    Returns:
        prices: (B,) tensor of European call prices
    """
    B = spot.shape[0]
    device = spot.device
    prices = torch.zeros(B, device=device)
    std_errors = torch.zeros(B, device=device)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    for b in range(B):
        T = maturity[b].item()
        dt = T / n_steps
        
        C = generate_fbm_covariance(n_steps, dt, H, device)
        C = C + torch.eye(n_steps, device=device) * 1e-6  # numerical stability
        L = torch.linalg.cholesky(C)
        
        # Standard driving normals for the volatility process
        Z_vol = torch.randn(n_paths, n_steps, device=device, generator=gen)
        
        # Fractional Brownian motion paths
        W_tilde = torch.matmul(Z_vol, L.T)
        
        # Volatility process (Rough Bergomi)
        t = torch.arange(1, n_steps + 1, dtype=torch.float32, device=device) * dt
        t_2H = t ** (2 * H)
        eta_b = eta[b]
        
        V = xi[b] * torch.exp(eta_b * W_tilde - 0.5 * (eta_b ** 2) * t_2H)

        # Left-point (predictable) variance: the variance applied over step i
        # must not contain step i's own innovation. Using the right-endpoint
        # V_i against a spot shock built from the same normal breaks the
        # martingale property whenever rho != 0 (the -V/2 dt correction no
        # longer offsets E[exp(sqrt(V) dW)]), which collapses prices at
        # negative correlation. This is the discrete analogue of the Ito
        # integral being left-point by construction.
        V = torch.cat([xi[b].expand(n_paths, 1), V[:, :-1]], dim=1)

        # Standard driving normals for the spot process (correlated with Z_vol)
        Z_indep = torch.randn(n_paths, n_steps, device=device, generator=gen)
        Z_spot = rho[b] * Z_vol + math.sqrt(1 - rho[b].item()**2) * Z_indep
        dW_spot = Z_spot * math.sqrt(dt)
        
        # Euler scheme for the log spot process
        integral_drift = torch.sum((rate[b] - 0.5 * V) * dt, dim=1)
        integral_vol = torch.sum(torch.sqrt(V) * dW_spot, dim=1)
        
        S_T = spot[b] * torch.exp(integral_drift + integral_vol)
        
        # European Call Payoff
        payoff = torch.clamp(S_T - strike[b], min=0.0)
        disc = math.exp(-rate[b].item() * T)
        prices[b] = payoff.mean() * disc
        std_errors[b] = payoff.std() * disc / math.sqrt(n_paths)

    if return_std_error:
        return prices, std_errors
    return prices

if __name__ == "__main__":
    # Quick sanity check
    spot = torch.tensor([100.0])
    strike = torch.tensor([100.0])
    maturity = torch.tensor([5.0 / 252.0]) # 5 Days to expiry
    xi = torch.tensor([0.25 ** 2])         # 25% initial vol
    eta = torch.tensor([1.5])              # high vol of vol
    rho = torch.tensor([-0.7])             # strong negative correlation (leverage effect)
    rate = torch.tensor([0.05])
    
    import time
    t0 = time.perf_counter()
    price = rough_bergomi_mc(spot, strike, maturity, xi, eta, rho, rate, n_paths=100000, n_steps=50)
    print(f"0DTE Rough Vol Price: ${price[0].item():.4f} (computed in {time.perf_counter() - t0:.2f}s)")
