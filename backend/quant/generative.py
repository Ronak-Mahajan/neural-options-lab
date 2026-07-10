"""Generative Market Simulator using WGAN-GP.

Trains a Wasserstein GAN with Gradient Penalty on historical log-returns 
(SPY) to generate realistic 30-day market paths with fat tails, 
replacing standard Geometric Brownian Motion.
"""

import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
N_STEPS = 30
DT = 1.0 / 252.0


class PathGenerator(nn.Module):
    """Generates 30-day log-return paths conditioned on volatility and rate."""
    def __init__(self, noise_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, N_STEPS)
        )

    def forward(self, z: torch.Tensor, sigma: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        # z: (B, noise_dim), sigma: (B, 1), rate: (B, 1)
        x = torch.cat([z, sigma, rate], dim=-1)
        # Network outputs raw structure; we scale it by sigma * sqrt(dt)
        base_scale = sigma * math.sqrt(DT)
        return self.net(x) * base_scale


def risk_neutralize(log_returns: torch.Tensor, sigma: torch.Tensor,
                    rate: torch.Tensor) -> torch.Tensor:
    """Map generated real-world paths onto the pricing (risk-neutral) measure.

    The raw WGAN-GP output carries whatever drift and vol level it learned
    from history (plus any training bias) — hedging P&L against a
    Black-Scholes premium computed at (sigma, r) would then conflate
    directional alpha with hedging skill. Cross-sectional standardization
    per time step preserves the generator's dependence structure, skew and
    fat tails, while enforcing the first two moments of the pricing
    measure:  mean (r - sigma^2/2) dt,  std sigma sqrt(dt).
    """
    mu = log_returns.mean(dim=0, keepdim=True)
    sd = log_returns.std(dim=0, keepdim=True).clamp_min(1e-8)
    eps = (log_returns - mu) / sd
    return (rate - 0.5 * sigma ** 2) * DT + sigma * math.sqrt(DT) * eps


class PathDiscriminator(nn.Module):
    """Discriminates real vs fake 30-day log-return paths."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STEPS + 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, returns: torch.Tensor, sigma: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        x = torch.cat([returns, sigma, rate], dim=-1)
        return self.net(x)


def compute_gradient_penalty(D: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor, 
                             sigma: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """Calculates the gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, sigma, rate)
    fake = torch.ones(real_samples.size(0), 1, device=real_samples.device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def fetch_historical_paths() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downloads SPY history and creates 30-day rolling log-return windows."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required. Run: pip install yfinance")
    
    print("Downloading SPY historical data...")
    df = yf.download("SPY", start="2000-01-01", progress=False)
    if df.empty:
        raise ValueError("Failed to download SPY data.")
    
    prices = df['Close'].to_numpy().squeeze()
    log_returns = np.diff(np.log(prices))
    
    # Create rolling windows of size N_STEPS
    windows = []
    sigmas = []
    rates = []
    
    for i in range(len(log_returns) - N_STEPS):
        window = log_returns[i:i + N_STEPS]
        # Annualized realized volatility for this window
        realized_vol = np.std(window) * np.sqrt(252)
        # We clamp realized vol to [0.08, 0.65] to match Hedger training box
        realized_vol = np.clip(realized_vol, 0.08, 0.65)
        
        # We don't have historical risk-free rate easily here, so we simulate 
        # random rates from [0.0, 0.09] to train the generator conditionally.
        fake_rate = np.random.uniform(0.0, 0.09)
        
        windows.append(window)
        sigmas.append([realized_vol])
        rates.append([fake_rate])
        
    windows_t = torch.tensor(np.array(windows), dtype=torch.float32)
    sigmas_t = torch.tensor(np.array(sigmas), dtype=torch.float32)
    rates_t = torch.tensor(np.array(rates), dtype=torch.float32)
    print(f"Extracted {len(windows_t)} rolling 30-day paths.")
    return windows_t, sigmas_t, rates_t


def train_generative_model(iters: int = 3000, batch_size: int = 512, lr: float = 1e-4):
    """Trains the WGAN-GP model."""
    real_returns, sigmas, rates = fetch_historical_paths()
    dataset = torch.utils.data.TensorDataset(real_returns, sigmas, rates)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    G = PathGenerator()
    D = PathDiscriminator()
    
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))
    
    lambda_gp = 10.0
    n_critic = 5
    
    print("Training WGAN-GP Generative Market Simulator...")
    t0 = time.perf_counter()
    
    # Infinite dataloader iterator
    def get_infinite_batches(dl):
        while True:
            for b in dl:
                yield b
    
    batch_iter = get_infinite_batches(loader)
    
    for it in range(1, iters + 1):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(n_critic):
            real_batch, sig_batch, r_batch = next(batch_iter)
            opt_D.zero_grad()
            
            z = torch.randn(batch_size, G.noise_dim)
            fake_batch = G(z, sig_batch, r_batch).detach()
            
            real_validity = D(real_batch, sig_batch, r_batch)
            fake_validity = D(fake_batch, sig_batch, r_batch)
            
            gp = compute_gradient_penalty(D, real_batch, fake_batch, sig_batch, r_batch)
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
            d_loss.backward()
            opt_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        opt_G.zero_grad()
        real_batch, sig_batch, r_batch = next(batch_iter)
        z = torch.randn(batch_size, G.noise_dim)
        fake_batch = G(z, sig_batch, r_batch)
        
        fake_validity = D(fake_batch, sig_batch, r_batch)
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        opt_G.step()
        
        if it % 500 == 0 or it == 1:
            print(f"iter {it:>5}/{iters}  D loss: {d_loss.item():.4f}  G loss: {g_loss.item():.4f}  ({time.perf_counter() - t0:.1f}s)")

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({
        "generator": G.state_dict(),
        "noise_dim": G.noise_dim,
        "n_steps": N_STEPS,
    }, ARTIFACTS / "generator.pt")
    print(f"Saved {ARTIFACTS / 'generator.pt'}")


if __name__ == "__main__":
    train_generative_model()
