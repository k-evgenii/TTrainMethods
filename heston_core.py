import numpy as np, torch, time
from scipy.integrate import quad

# ---------------- Semi-analytic Heston (referee) ----------------
def heston_cf(u, T, kappa, theta, xi, rho, v0, F0):
    """Characteristic function of log F_T under the forward measure ('little Heston trap' form)."""
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi ** 2 * (iu + u ** 2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    e = np.exp(-d * T)
    C = (kappa * theta / xi ** 2) * ((kappa - rho * xi * iu - d) * T
                                     - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = ((kappa - rho * xi * iu - d) / xi ** 2) * ((1.0 - e) / (1.0 - g * e))
    return np.exp(C + D * v0 + iu * np.log(F0))

def heston_call_analytic(K, T, kappa, theta, xi, rho, v0, F0):
    """Undiscounted call on the forward. Gil-Pelaez / Heston P1-P2."""
    k = np.log(K)
    def integrand(u, j):
        if j == 1:
            num = heston_cf(u - 1j, T, kappa, theta, xi, rho, v0, F0)
            return np.real(np.exp(-1j * u * k) * num / (1j * u * F0))
        return np.real(np.exp(-1j * u * k) * heston_cf(u, T, kappa, theta, xi, rho, v0, F0) / (1j * u))
    P1 = 0.5 + quad(integrand, 1e-8, 200.0, args=(1,), limit=400)[0] / np.pi
    P2 = 0.5 + quad(integrand, 1e-8, 200.0, args=(2,), limit=400)[0] / np.pi
    return F0 * P1 - K * P2

# ---------------- Black-76 + IV inversion ----------------
def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

def black76(F, K, T, sigma):
    """Undiscounted Black-76 call. Torch, broadcasting."""
    sT = sigma * torch.sqrt(T)
    d1 = (torch.log(F / K) + 0.5 * sT ** 2) / sT
    d2 = d1 - sT
    return F * _norm_cdf(d1) - K * _norm_cdf(d2)

def black76_put(F, K, T, sigma):
    """Undiscounted Black-76 put (put-call parity on the forward)."""
    return black76(F, K, T, sigma) - (F - K)

def implied_vol(price, F, K, T, lo=1e-3, hi=3.0, iters=60, otm_price=False):
    """
    Vectorised bisection on Black-76. Monotone in sigma, so bisection is unconditionally safe.

    otm_price=True means `price` is ALREADY the OTM option price. Inversion always uses the OUT-OF-THE-MONEY wing: puts for K<F, calls for K>=F.
    ITM options are nearly all intrinsic value -> vega ~ 0 -> implied vol is ill-conditioned
    and the solver returns the bracket floor. This is the same OTM-wing convention used to
    build the market surface in Phase 3.  `price` is always the CALL price; the put is
    obtained by parity, so the caller does not change.
    """
    is_put = K < F
    if not otm_price:
        price = torch.where(is_put, price - (F - K), price)   # call price -> put by parity
    lo = torch.full_like(price, lo); hi = torch.full_like(price, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        model = torch.where(is_put, black76_put(F, K, T, mid), black76(F, K, T, mid))
        too_high = model > price
        hi = torch.where(too_high, mid, hi)
        lo = torch.where(too_high, lo, mid)
    return 0.5 * (lo + hi)

# ---------------- FTE Euler Monte-Carlo simulator ----------------
def heston_mc_forwards(V, T_grid, F0, n_paths, n_steps, device, generator=None, antithetic=True):
    """
    V: (n_V, 5) tensor of (kappa, theta, xi, rho, v0).
    Returns F at each maturity: (n_V, n_paths_eff, n_T).
    ONE path set -> ALL maturities (record F as we pass each T_j).
    Full-truncation Euler (Lord et al. 2010): v^+ = max(v, 0) everywhere it is used.
    """
    n_V = V.shape[0]
    kappa, theta, xi, rho, v0 = [V[:, i].view(n_V, 1) for i in range(5)]
    T_max = float(T_grid[-1])
    dt = T_max / n_steps
    sdt = np.sqrt(dt)
    n_half = n_paths // 2 if antithetic else n_paths
    n_eff = 2 * n_half if antithetic else n_paths

    # record indices: first step index at/after each maturity
    rec = [min(int(np.ceil(float(t) / dt)), n_steps) for t in T_grid]

    logF = torch.full((n_V, n_eff), float(np.log(F0)), device=device)
    v = v0.expand(n_V, n_eff).clone()
    out = torch.empty(n_V, n_eff, len(T_grid), device=device)

    ri = 0
    for step in range(1, n_steps + 1):
        z1 = torch.randn(n_V, n_half, device=device, generator=generator)
        z2 = torch.randn(n_V, n_half, device=device, generator=generator)
        if antithetic:
            z1 = torch.cat([z1, -z1], dim=1); z2 = torch.cat([z2, -z2], dim=1)
        zv = z1
        zs = rho * z1 + torch.sqrt(1.0 - rho ** 2) * z2

        vp = torch.clamp(v, min=0.0)                       # full truncation
        sq = torch.sqrt(vp)
        logF = logF - 0.5 * vp * dt + sq * sdt * zs        # forward: zero drift
        v = v + kappa * (theta - vp) * dt + xi * sq * sdt * zv

        while ri < len(rec) and step == rec[ri]:
            out[:, :, ri] = torch.exp(logF)
            ri += 1
    return out

def surface_from_forwards(FT, T_grid, K_grid, F0, device):
    """
    FT: (n_V, n_paths, n_T) terminal forwards.  K_grid: (n_K,) strikes.
    Returns implied-vol surface (n_V, n_T, n_K).
    """
    n_V, _, n_T = FT.shape; n_K = len(K_grid)
    K = torch.tensor(K_grid, dtype=torch.float32, device=device).view(1, n_K, 1)
    Tt = torch.tensor(T_grid, dtype=torch.float32, device=device).view(1, 1, n_T)
    # MC price of the call: mean over paths
    payoff = torch.clamp(FT.unsqueeze(1) - K.unsqueeze(-1), min=0.0)   # (n_V, n_K, n_paths, n_T)
    price = payoff.mean(dim=2)                                          # (n_V, n_K, n_T)
    Fb = torch.full_like(price, F0)
    iv = implied_vol(price, Fb, K.expand_as(price), Tt.expand_as(price))
    return iv.permute(0, 2, 1)                                          # (n_V, n_T, n_K)

# ---------------- Surface grid + Step-1-compatible simulate() ----------------
# Strikes are defined in ATM STANDARD DEVIATIONS, not fixed moneyness:
#     K(T, z) = F0 * exp(z * SIG_REF * sqrt(T))
# Fixed moneyness makes short maturities many sd OTM -> MC price ~ 0 -> IV inversion
# fails. sqrt-T scaling keeps every cell ~z sd out at every maturity. Verified: with
# this grid sigma_MC scales as 1/sqrt(n_paths) (1.8x for 4x paths) and is uniform
# across maturities; with fixed moneyness it did neither.

def make_strikes(T_grid, Z, F0, sig_ref=0.20):
    """-> list of per-maturity strike arrays, each (len(Z),)."""
    return [F0 * np.exp(np.asarray(Z) * sig_ref * np.sqrt(T)) for T in T_grid]


def heston_surface(V, T_grid, Z, F0, n_paths, n_steps, device, sig_ref=0.20, generator=None):
    """
    V: (n_V, 5) tensor (kappa, theta, xi, rho, v0)  ->  IV surface (n_V, n_T, n_Z).
    One path set -> all maturities. Strikes sqrt-T scaled per maturity.
    """
    FT = heston_mc_forwards(V, T_grid, F0, n_paths, n_steps, device, generator=generator)
    Ks = make_strikes(T_grid, Z, F0, sig_ref)
    n_V, n_T, n_Z = V.shape[0], len(T_grid), len(Z)
    out = torch.empty(n_V, n_T, n_Z, device=device)
    for i, T in enumerate(T_grid):
        K = torch.tensor(Ks[i], dtype=torch.float32, device=device).view(1, n_Z, 1)
        FTi = FT[:, :, i].unsqueeze(1)                       # (n_V, 1, n_paths)
        is_put = (K < F0)                                    # (1, n_Z, 1)
        # Price the OTM option DIRECTLY in MC. Do NOT price the call and use parity:
        # for K << F the call is ~all intrinsic, its time value is far smaller than the
        # MC standard error on E[F_T], so the price lands below intrinsic and no implied
        # vol exists. The OTM payoff is small, so its MC error scales with its own size.
        payoff = torch.where(is_put, torch.clamp(K - FTi, min=0.0),
                                     torch.clamp(FTi - K, min=0.0))
        price = payoff.mean(dim=2)                           # (n_V, n_Z) OTM price
        Kx = K.view(1, n_Z).expand_as(price)
        out[:, i, :] = implied_vol(price, torch.full_like(price, F0), Kx,
                                   torch.full_like(price, float(T)), otm_price=True)
    return out


def simulate_heston(V_np, T_grid, Z, F0, n_paths, n_steps, sigma_obs, device, rng,
                    sig_ref=0.20, chunk=256):
    """
    Drop-in replacement for Step 1's simulate().
    V_np: (n, 5) numpy -> X: (n, n_T * n_Z) numpy, flattened IV surface + observation noise E.

    NOTE: n_paths is part of the model. It MUST be identical at train and deploy --
    finite MC makes F itself stochastic (that stochasticity IS the nuisance W).
    """
    outs = []
    for s in range(0, V_np.shape[0], chunk):
        V = torch.tensor(V_np[s:s+chunk], dtype=torch.float32, device=device)
        iv = heston_surface(V, T_grid, Z, F0, n_paths, n_steps, device, sig_ref)
        outs.append(iv.reshape(V.shape[0], -1).cpu().numpy())
    X = np.concatenate(outs, axis=0)
    return X + rng.normal(0.0, sigma_obs, size=X.shape)   # + E
