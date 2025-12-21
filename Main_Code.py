# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 14:03:29 2025

@author: WONCHAN
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(x, device, dtype=torch.float32):
    return torch.as_tensor(x, device=device, dtype=dtype)


def fmt_date(x) -> str:
    if isinstance(x, (np.datetime64, pd.Timestamp)):
        return str(pd.to_datetime(x).date())
    return str(x)


def get_date_range(dates: np.ndarray, rng_tuple: Tuple[int, int]) -> Tuple[str, str]:
    a, b = rng_tuple
    a = int(max(a, 0))
    b = int(min(b, len(dates)))
    if b <= a:
        return "NA", "NA"
    return fmt_date(dates[a]), fmt_date(dates[b - 1])


def simplex_projection(v: torch.Tensor, z: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    if v.dim() != 2:
        raise ValueError("simplex_projection expects (B, d).")
    B, d = v.shape
    u, _ = torch.sort(v, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - z
    ind = torch.arange(1, d + 1, device=v.device, dtype=v.dtype).view(1, -1).expand(B, -1)
    cond = u - cssv / (ind + eps) > 0
    rho = cond.sum(dim=1).clamp(min=1).to(torch.long) - 1
    theta = cssv.gather(1, rho.view(-1, 1)) / (rho.view(-1, 1).to(v.dtype) + 1.0)
    w = torch.clamp(v - theta, min=0.0)
    s = w.sum(dim=1, keepdim=True).clamp(min=eps)
    return w * (z / s)


def crra_utility(c: torch.Tensor, gamma: float, eps: float = 1e-12) -> torch.Tensor:
    c = torch.clamp(c, min=eps)
    if abs(gamma - 1.0) < 1e-8:
        return torch.log(c)
    return (c.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)


def crra_ce_from_utility(u: torch.Tensor, gamma: float) -> torch.Tensor:
    if abs(gamma - 1.0) < 1e-8:
        return torch.exp(u)
    return torch.clamp((u * (1.0 - gamma) + 1.0), min=1e-12).pow(1.0 / (1.0 - gamma))


def ce_terminal_wealth(W_T: torch.Tensor, gamma: float, eps: float = 1e-12) -> float:
    W_T = torch.clamp(W_T, min=eps)
    if abs(gamma - 1.0) < 1e-8:
        return torch.exp(torch.mean(torch.log(W_T))).item()
    u = torch.mean((W_T.pow(1.0 - gamma) - 1.0) / (1.0 - gamma))
    return crra_ce_from_utility(u, gamma).item()


def max_drawdown_from_returns(r: np.ndarray, eps: float = 1e-12) -> float:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / np.maximum(peak, eps)
    return float(np.max(dd))


def ann_stats_from_returns(r: np.ndarray, periods_per_year: int = 252) -> Tuple[float, float, float]:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(r)) * periods_per_year
    vol = float(np.std(r, ddof=1)) * np.sqrt(periods_per_year) if r.size > 1 else 0.0
    mdd = max_drawdown_from_returns(r)
    return mu, vol, mdd


def rolling_shrinkage_belief(
    returns: np.ndarray,
    window: int,
    shrink_alpha: float,
    use_full_cov: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, d = returns.shape
    mu = np.zeros((N, d), dtype=np.float32)
    cov_full = np.zeros((N, d, d), dtype=np.float32)
    cov_diag = np.zeros((N, d), dtype=np.float32)
    u = np.zeros((N, 1), dtype=np.float32)

    for t in range(N):
        start = max(0, t - window + 1)
        X = returns[start : t + 1]
        if X.shape[0] < max(10, min(window, 20)):
            continue

        mu_t = X.mean(axis=0).astype(np.float32)
        S = np.cov(X, rowvar=False, bias=False).astype(np.float32)
        if S.ndim == 0:
            S = np.eye(d, dtype=np.float32) * float(S)
        elif S.ndim == 1:
            S = np.diag(S).astype(np.float32)

        diagS = np.diag(np.diag(S)).astype(np.float32)
        Sigma = (1.0 - shrink_alpha) * S + shrink_alpha * diagS

        mu[t] = mu_t
        cov_full[t] = Sigma
        cov_diag[t] = np.diag(Sigma).astype(np.float32)
        u[t, 0] = float(np.trace(Sigma))

    if use_full_cov:
        return mu, cov_full, u
    return mu, cov_diag, u


def kalman_mean_belief(
    returns: np.ndarray,
    R_diag: np.ndarray,
    q: float,
    init_P: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    N, d = returns.shape
    mu_f = np.zeros((N, d), dtype=np.float32)
    u = np.zeros((N, 1), dtype=np.float32)

    mu = np.zeros((d,), dtype=np.float32)
    P = np.eye(d, dtype=np.float32) * float(init_P)
    Q = np.eye(d, dtype=np.float32) * float(q)
    I = np.eye(d, dtype=np.float32)

    for t in range(N):
        mu_pred = mu
        P_pred = P + Q

        Rt = np.diag(np.maximum(R_diag[t], 1e-12).astype(np.float32))
        S = P_pred + Rt
        try:
            S_inv = np.linalg.inv(S).astype(np.float32)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S).astype(np.float32)

        K = (P_pred @ S_inv).astype(np.float32)
        y = returns[t].astype(np.float32) - mu_pred
        mu = (mu_pred + K @ y).astype(np.float32)
        P = ((I - K) @ P_pred).astype(np.float32)

        mu_f[t] = mu
        u[t, 0] = float(np.trace(P))

    return mu_f, u


def build_branch_features(mu: np.ndarray, cov: np.ndarray, u: np.ndarray, rf: np.ndarray) -> np.ndarray:
    N, d = mu.shape
    if cov.ndim == 3:
        diag = np.diagonal(cov, axis1=1, axis2=2).astype(np.float32)
    else:
        diag = cov.astype(np.float32)
    rf_col = rf.reshape(N, 1).astype(np.float32)
    feat = np.concatenate([mu.astype(np.float32), diag, rf_col, u.astype(np.float32)], axis=1)
    return feat.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, out_dim: int, act=nn.SiLU):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden))
            layers.append(act())
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONetPolicy(nn.Module):
    def __init__(
        self,
        trunk_in: int,
        branch_in: int,
        d_assets: int,
        m: int = 128,
        trunk_hidden: int = 256,
        branch_hidden: int = 256,
        trunk_depth: int = 2,
        branch_depth: int = 2,
    ):
        super().__init__()
        self.d_assets = d_assets
        self.m = m
        self.trunk = MLP(trunk_in, trunk_hidden, trunk_depth, m)
        self.branch = MLP(branch_in, branch_hidden, branch_depth, m)
        self.head = nn.Linear(m, d_assets + 1)

    def forward(self, trunk_x: torch.Tensor, branch_x: torch.Tensor):
        tfeat = self.trunk(trunk_x)
        bfeat = self.branch(branch_x)
        out = self.head(tfeat * bfeat)
        logits_pi = out[:, : self.d_assets]
        raw_rho = out[:, self.d_assets : self.d_assets + 1]
        return logits_pi, raw_rho


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    horizon: int = 60
    batch_size: int = 256

    gamma: float = 3.0
    beta: float = 0.999
    kappa_b: float = 1.0

    w0: float = 1.0
    w_min: float = 1e-6

    use_dt: bool = True
    dt: float = 1.0 / 252.0
    rho_max: float = 0.05

    tc_type: str = "l1"
    lambda_tc: float = 5e-3
    tc_cap: float = 1e9

    pi_smooth: float = 0.2
    weight_decay: float = 1e-5

    lr: float = 2e-4
    iters: int = 5000
    grad_clip: float = 1.0

    window: int = 252
    shrink_alpha: float = 0.5

    kalman_q: float = 1e-6
    init_P: float = 1e-3

    robust_block: int = 20
    noise_mu: float = 0.0
    noise_cov: float = 0.0

    train_frac: float = 0.70
    val_frac: float = 0.15

    print_every: int = 200
    save_path: str = ""

    early_patience: int = 8
    early_min_delta: float = 0.0


def turnover_cost(pi: torch.Tensor, pi_prev: torch.Tensor, tc_type: str) -> torch.Tensor:
    if tc_type == "l1":
        return torch.sum(torch.abs(pi - pi_prev), dim=1, keepdim=True)
    if tc_type == "l2":
        return torch.sum((pi - pi_prev) ** 2, dim=1, keepdim=True)
    raise ValueError(f"Unknown tc_type: {tc_type}")


def sample_start_indices(
    low: int,
    high: int,
    horizon: int,
    batch: int,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    max_start = high - horizon
    if max_start < low:
        raise ValueError("Not enough room for horizon in this split range.")
    if block <= 1:
        return rng.integers(low, max_start + 1, size=(batch,), endpoint=False).astype(np.int64)

    span = max_start - low + 1
    nb = max(1, span // block)
    b0 = rng.integers(0, nb, size=(batch,), endpoint=False)
    off = rng.integers(0, block, size=(batch,), endpoint=False)
    s = low + b0 * block + off
    return np.clip(s, low, max_start).astype(np.int64)


def run_episode_batch(
    policy: DeepONetPolicy,
    returns_t: torch.Tensor,
    rf_t: torch.Tensor,
    branch_t: torch.Tensor,
    starts: torch.Tensor,
    cfg: TrainConfig,
    train: bool = True,
    assume_excess_returns: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = returns_t.device
    B = starts.shape[0]
    H = cfg.horizon
    d = returns_t.shape[1]

    W = torch.full((B, 1), float(cfg.w0), device=device)
    pi_prev = torch.full((B, d), 1.0 / d, device=device)

    total_u = torch.zeros((B, 1), device=device)
    disc = 1.0
    avg_tc = torch.zeros((B, 1), device=device)

    for k in range(H):
        t = starts + k
        tau = torch.full((B, 1), float(H - k), device=device)
        trunk_x = torch.cat([tau / float(H), W], dim=1)

        b = branch_t[t]
        if train and (cfg.noise_mu > 0.0 or cfg.noise_cov > 0.0):
            b = b.clone()
            if cfg.noise_mu > 0.0:
                b[:, :d] = b[:, :d] + cfg.noise_mu * torch.randn((B, d), device=device)
            if cfg.noise_cov > 0.0:
                b[:, d : 2 * d] = b[:, d : 2 * d] + cfg.noise_cov * torch.randn((B, d), device=device)

        logits_pi, raw_rho = policy(trunk_x, b)

        pi_raw = simplex_projection(logits_pi, z=1.0)
        s = float(cfg.pi_smooth)
        if s <= 0.0:
            pi = pi_raw
        else:
            pi = (1.0 - s) * pi_prev + s * pi_raw
            pi = simplex_projection(pi, z=1.0)

        rho = cfg.rho_max * torch.sigmoid(raw_rho)
        c = rho * W * (cfg.dt if cfg.use_dt else 1.0)

        tc = turnover_cost(pi, pi_prev, cfg.tc_type)
        tc = torch.clamp(tc, max=float(cfg.tc_cap))
        avg_tc = avg_tc + tc / float(H)

        rf_step = rf_t[t]
        r_next = returns_t[t + 1]

        if assume_excess_returns:
            gross = 1.0 + rf_step + torch.sum(pi * r_next, dim=1, keepdim=True)
        else:
            gross = 1.0 + torch.sum(pi * r_next, dim=1, keepdim=True)

        W_next = (W - c) * gross - cfg.lambda_tc * tc * W
        W_next = torch.clamp(W_next, min=cfg.w_min)

        total_u = total_u + (disc * crra_utility(c, cfg.gamma))

        pi_prev = pi
        W = W_next
        disc *= cfg.beta

    total_u = total_u + (disc * cfg.kappa_b * crra_utility(W, cfg.gamma))
    obj = torch.mean(total_u)
    return obj, W, total_u, torch.mean(avg_tc)


def compute_split_indices(N: int, cfg: TrainConfig) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    train_end = int(N * cfg.train_frac)
    val_end = int(N * (cfg.train_frac + cfg.val_frac))
    train_end = max(train_end, 10)
    val_end = max(val_end, train_end + 10)
    val_end = min(val_end, N - 2)
    train_end = min(train_end, val_end - 10)
    train_rng = (0, train_end)
    val_rng = (train_end, val_end)
    test_rng = (val_end, N - 2)
    return train_rng, val_rng, test_rng


def load_timeseries_csv(
    csv_path: str, rf_col: str, asset_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").to_numpy()
    else:
        dates = np.arange(len(df), dtype=np.int64)

    if rf_col not in df.columns:
        raise ValueError(f"rf_col='{rf_col}' not found. columns={df.columns.tolist()}")

    if asset_cols is None or len(asset_cols) == 0:
        asset_cols = [c for c in df.columns if c not in ["date", rf_col]]

    for c in asset_cols:
        if c not in df.columns:
            raise ValueError(f"asset col '{c}' not found in CSV.")

    rf = df[rf_col].to_numpy(dtype=np.float32)
    rets = df[asset_cols].to_numpy(dtype=np.float32)

    ok = np.isfinite(rf) & np.all(np.isfinite(rets), axis=1)
    if "date" in df.columns:
        ok = ok & pd.to_datetime(df["date"], errors="coerce").notna().to_numpy()

    rf = rf[ok]
    rets = rets[ok]
    dates = dates[ok]

    return dates, rf, rets, asset_cols


def compute_split_regime_stats(
    returns: np.ndarray,
    dates: np.ndarray,
    rng_tuple: Tuple[int, int],
) -> Dict[str, Any]:
    a, b = rng_tuple
    a = int(max(a, 0))
    b = int(min(b, returns.shape[0]))
    if b <= a:
        return {
            "start_date": "NA",
            "end_date": "NA",
            "eq_ann_mean": float("nan"),
            "eq_ann_vol": float("nan"),
            "eq_mdd": float("nan"),
            "eq_sharpe": float("nan"),
        }

    r_eq = np.mean(returns[a:b], axis=1).astype(np.float64)
    mu, vol, mdd = ann_stats_from_returns(r_eq, periods_per_year=252)
    sharpe = float(mu / (vol + 1e-12)) if np.isfinite(mu) and np.isfinite(vol) else float("nan")
    sd, ed = get_date_range(dates, (a, b))
    return {
        "start_date": sd,
        "end_date": ed,
        "eq_ann_mean": mu,
        "eq_ann_vol": vol,
        "eq_mdd": mdd,
        "eq_sharpe": sharpe,
    }


def build_val_difficulty_paragraph(train_stats: Dict[str, Any], val_stats: Dict[str, Any], test_stats: Dict[str, Any]) -> str:
    def f(x):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "NA"
        return f"{x:.4f}"

    vs = f"{val_stats['start_date']}~{val_stats['end_date']}"
    ts = f"{train_stats['start_date']}~{train_stats['end_date']}"
    tes = f"{test_stats['start_date']}~{test_stats['end_date']}"

    # NOTE: 논리 충돌 방지를 위해 "harder"의 근거를 mean/vol/sharpe 중심으로 둔다.
    # MDD는 보조지표로만 언급(또는 split마다 달라질 수 있음)을 명시한다.
    s = []
    s.append(f"Validation window is {vs}. For reference, train={ts}, test={tes}.")
    s.append(
        "Using an equal-weight proxy portfolio to characterize the market regime, the validation subperiod exhibits a weaker risk-adjusted environment: "
        f"annualized mean={f(val_stats['eq_ann_mean'])} vs train={f(train_stats['eq_ann_mean'])} / test={f(test_stats['eq_ann_mean'])}, "
        f"annualized vol={f(val_stats['eq_ann_vol'])} vs train={f(train_stats['eq_ann_vol'])} / test={f(test_stats['eq_ann_vol'])}, "
        f"Sharpe (mean/vol)={f(val_stats['eq_sharpe'])} vs train={f(train_stats['eq_sharpe'])} / test={f(test_stats['eq_sharpe'])}. "
        "Drawdown statistics can vary across splits and are therefore treated as a secondary diagnostic; nonetheless, the depressed validation CE_term is consistent with the weaker mean-vol-Sharpe profile of the VAL window rather than a turnover blow-up."
    )
    return " ".join(s)


def train_deeponet_merton(
    csv_path: str,
    rf_col: str,
    asset_cols: Optional[List[str]],
    belief: str,
    cfg: TrainConfig,
    assume_excess_returns: bool = True,
) -> Tuple[DeepONetPolicy, Dict[str, Any]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    dates, rf, returns, used_cols = load_timeseries_csv(csv_path, rf_col=rf_col, asset_cols=asset_cols)
    N, d = returns.shape

    train_rng, val_rng, test_rng = compute_split_indices(N, cfg)

    train_stats = compute_split_regime_stats(returns, dates, train_rng)
    val_stats = compute_split_regime_stats(returns, dates, val_rng)
    test_stats = compute_split_regime_stats(returns, dates, test_rng)

    mu_R, cov_R, u_R = rolling_shrinkage_belief(
        returns=returns,
        window=cfg.window,
        shrink_alpha=cfg.shrink_alpha,
        use_full_cov=False,
    )
    R_diag = cov_R.astype(np.float32)

    if belief.lower() == "rolling":
        mu, cov, u = mu_R, cov_R, u_R
    elif belief.lower() == "kalman":
        mu_K, u_K = kalman_mean_belief(
            returns=returns,
            R_diag=R_diag,
            q=cfg.kalman_q,
            init_P=cfg.init_P,
        )
        mu, cov, u = mu_K, cov_R, u_K
    else:
        raise ValueError("belief must be 'rolling' or 'kalman'.")

    branch_np = build_branch_features(mu, cov, u, rf)

    returns_t = to_torch(returns, device)
    rf_t = to_torch(rf.reshape(-1, 1), device)
    branch_t = to_torch(branch_np, device)

    policy = DeepONetPolicy(
        trunk_in=2,
        branch_in=branch_np.shape[1],
        d_assets=d,
        m=128,
        trunk_hidden=256,
        branch_hidden=256,
        trunk_depth=2,
        branch_depth=2,
    ).to(device)

    opt = optim.Adam(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed + 123)

    best_val = -1e30
    best_state = None
    bad_count = 0

    def eval_on_range(rng_tuple, tag: str):
        low, high = rng_tuple
        starts_np = sample_start_indices(low, high, cfg.horizon, cfg.batch_size, cfg.robust_block, rng)
        starts = to_torch(starts_np, device, dtype=torch.long)
        with torch.no_grad():
            obj, W_T, _, avg_tc = run_episode_batch(
                policy, returns_t, rf_t, branch_t, starts, cfg, train=False, assume_excess_returns=assume_excess_returns
            )
            ce = ce_terminal_wealth(W_T.squeeze(1), cfg.gamma)
            wmean = W_T.mean().item()
            avg_tc_v = avg_tc.item()
            obj_v = obj.item()
        print(f"[{tag}] obj={obj_v: .6f}  CE_term={ce: .6f}  W_T_mean={wmean: .6f}  avg_TC={avg_tc_v: .6f}")
        return obj_v, ce, wmean, avg_tc_v

    train_sd, train_ed = train_stats["start_date"], train_stats["end_date"]
    val_sd, val_ed = val_stats["start_date"], val_stats["end_date"]
    test_sd, test_ed = test_stats["start_date"], test_stats["end_date"]

    print(f"CSV: {csv_path}")
    print(f"Assets(d={d}): {used_cols}")
    print(f"N={N} | split train={train_rng} val={val_rng} test={test_rng} | belief={belief}")
    print(f"Date ranges | train={train_sd}~{train_ed} | val={val_sd}~{val_ed} | test={test_sd}~{test_ed}")
    print("---- Training ----")

    for it in range(1, cfg.iters + 1):
        low, high = train_rng
        starts_np = sample_start_indices(low, high, cfg.horizon, cfg.batch_size, cfg.robust_block, rng)
        starts = to_torch(starts_np, device, dtype=torch.long)

        obj, W_T, _, avg_tc = run_episode_batch(
            policy=policy,
            returns_t=returns_t,
            rf_t=rf_t,
            branch_t=branch_t,
            starts=starts,
            cfg=cfg,
            train=True,
            assume_excess_returns=assume_excess_returns,
        )

        loss = -obj
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()

        if (it % cfg.print_every) == 0 or it == 1:
            ce = ce_terminal_wealth(W_T.squeeze(1), cfg.gamma)
            print(
                f"[TRAIN] it={it:5d}  obj={obj.item(): .6f}  CE_term={ce: .6f}  "
                f"W_T_mean={W_T.mean().item(): .6f}  avg_TC={avg_tc.item(): .6f}"
            )

            val_obj, _, _, _ = eval_on_range(val_rng, "VAL")
            if val_obj > best_val + cfg.early_min_delta:
                best_val = val_obj
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                bad_count = 0
            else:
                bad_count += 1
                if bad_count >= cfg.early_patience:
                    break

    if best_state is not None:
        policy.load_state_dict(best_state)

    print("---- Final Evaluation (best-val checkpoint) ----")
    tr = eval_on_range(train_rng, "TRAIN")
    va = eval_on_range(val_rng, "VAL")
    te = eval_on_range(test_rng, "TEST")

    paragraph = build_val_difficulty_paragraph(train_stats, val_stats, test_stats)
    print("\nVAL regime note:")
    print(paragraph)

    summary = {
        "belief": belief,
        "lambda_tc": float(cfg.lambda_tc),
        "train_obj": tr[0], "train_CE": tr[1], "train_Wmean": tr[2], "train_TC": tr[3],
        "val_obj":   va[0], "val_CE":   va[1], "val_Wmean":   va[2], "val_TC":   va[3],
        "test_obj":  te[0], "test_CE":  te[1], "test_Wmean":  te[2], "test_TC":  te[3],
        "train_start": train_sd, "train_end": train_ed,
        "val_start": val_sd, "val_end": val_ed,
        "test_start": test_sd, "test_end": test_ed,
        "train_eq_ann_mean": float(train_stats["eq_ann_mean"]),
        "train_eq_ann_vol":  float(train_stats["eq_ann_vol"]),
        "train_eq_mdd":      float(train_stats["eq_mdd"]),
        "train_eq_sharpe":   float(train_stats["eq_sharpe"]),
        "val_eq_ann_mean":   float(val_stats["eq_ann_mean"]),
        "val_eq_ann_vol":    float(val_stats["eq_ann_vol"]),
        "val_eq_mdd":        float(val_stats["eq_mdd"]),
        "val_eq_sharpe":     float(val_stats["eq_sharpe"]),
        "test_eq_ann_mean":  float(test_stats["eq_ann_mean"]),
        "test_eq_ann_vol":   float(test_stats["eq_ann_vol"]),
        "test_eq_mdd":       float(test_stats["eq_mdd"]),
        "test_eq_sharpe":    float(test_stats["eq_sharpe"]),
        "val_regime_note": paragraph,
    }

    if cfg.save_path:
        os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
        torch.save(
            {"state_dict": policy.state_dict(), "assets": used_cols, "cfg": vars(cfg), "summary": summary},
            cfg.save_path,
        )
        print(f"Saved model: {os.path.abspath(cfg.save_path)}")

    return policy, summary


def parse_float_list(s: str) -> List[float]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(float(tok))
    return out


def run_sweeps(args):
    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")

    asset_cols = [c.strip() for c in args.asset_cols.split(",") if c.strip() != ""]
    if len(asset_cols) == 0:
        asset_cols = None

    lambda_list = parse_float_list(args.sweep_lambdas)
    if len(lambda_list) == 0:
        lambda_list = [1e-3, 2e-3, 5e-3, 1e-2]

    belief_list = ["rolling", "kalman"] if args.sweep_beliefs.lower() == "both" else [args.belief]

    rows = []
    for belief in belief_list:
        for lam in lambda_list:
            cfg = TrainConfig(
                seed=args.seed,
                device=device,
                horizon=args.horizon,
                batch_size=args.batch_size,
                iters=args.iters,
                lr=args.lr,
                grad_clip=args.grad_clip,
                gamma=args.gamma,
                beta=args.beta,
                kappa_b=args.kappa_b,
                w0=args.w0,
                w_min=args.w_min,
                use_dt=True,
                dt=float(args.dt),
                rho_max=float(args.rho_max),
                tc_type=args.tc_type,
                lambda_tc=float(lam),
                tc_cap=float(args.tc_cap),
                pi_smooth=float(args.pi_smooth),
                weight_decay=float(args.weight_decay),
                window=args.window,
                shrink_alpha=float(args.shrink_alpha),
                kalman_q=float(args.kalman_q),
                init_P=float(args.init_P),
                robust_block=int(args.robust_block),
                noise_mu=float(args.noise_mu),
                noise_cov=float(args.noise_cov),
                train_frac=float(args.train_frac),
                val_frac=float(args.val_frac),
                print_every=int(args.print_every),
                save_path="",
                early_patience=int(args.early_patience),
                early_min_delta=float(args.early_min_delta),
            )

            print("\n" + "=" * 90)
            print(f"RUN: belief={belief} | lambda_tc={lam:g}")
            print("=" * 90)

            _, summary = train_deeponet_merton(
                csv_path=args.csv,
                rf_col=args.rf_col,
                asset_cols=asset_cols,
                belief=belief,
                cfg=cfg,
                assume_excess_returns=bool(args.assume_excess_returns),
            )
            rows.append(summary)

    df = pd.DataFrame(rows)

    df["train_range"] = df["train_start"].astype(str) + "~" + df["train_end"].astype(str)
    df["val_range"] = df["val_start"].astype(str) + "~" + df["val_end"].astype(str)
    df["test_range"] = df["test_start"].astype(str) + "~" + df["test_end"].astype(str)

    cols = [
        "belief", "lambda_tc",
        "train_range", "val_range", "test_range",
        "train_CE", "val_CE", "test_CE",
        "train_TC", "val_TC", "test_TC",
        "train_Wmean", "val_Wmean", "test_Wmean",
        "val_eq_ann_mean", "val_eq_ann_vol", "val_eq_sharpe",
        "train_eq_ann_mean", "train_eq_ann_vol", "train_eq_sharpe",
        "test_eq_ann_mean", "test_eq_ann_vol", "test_eq_sharpe",
        "val_eq_mdd", "train_eq_mdd", "test_eq_mdd",
    ]
    df_out = df[cols].sort_values(["belief", "lambda_tc"]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("SWEEP SUMMARY TABLE (rolling vs kalman) × (lambda_tc sweep)")
    print("=" * 90)
    print(df_out.to_string(index=False))

    out_csv = args.sweep_out_csv.strip() or "sweep_summary.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {os.path.abspath(out_csv)}")

    # Print the regime note once per belief using the first lambda (for cleaner logs)
    for belief in sorted(df["belief"].unique().tolist()):
        tmp = df[df["belief"] == belief].sort_values("lambda_tc").head(1)
        if len(tmp) > 0:
            print("\nVAL regime note (belief=%s):" % belief)
            print(tmp.iloc[0]["val_regime_note"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, default="paperA_timeseries (1).csv")
    parser.add_argument("--rf_col", type=str, default="rf")
    parser.add_argument("--asset_cols", type=str, default="")
    parser.add_argument("--belief", type=str, default="rolling", choices=["rolling", "kalman"])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--kappa_b", type=float, default=1.0)

    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w_min", type=float, default=1e-6)

    parser.add_argument("--dt", type=float, default=1.0 / 252.0)
    parser.add_argument("--rho_max", type=float, default=0.05)

    parser.add_argument("--tc_type", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--lambda_tc", type=float, default=5e-3)
    parser.add_argument("--tc_cap", type=float, default=1e9)

    parser.add_argument("--pi_smooth", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--shrink_alpha", type=float, default=0.5)
    parser.add_argument("--kalman_q", type=float, default=1e-6)
    parser.add_argument("--init_P", type=float, default=1e-3)

    parser.add_argument("--robust_block", type=int, default=20)
    parser.add_argument("--noise_mu", type=float, default=0.0)
    parser.add_argument("--noise_cov", type=float, default=0.0)

    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)

    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--save_path", type=str, default="")

    parser.add_argument("--early_patience", type=int, default=8)
    parser.add_argument("--early_min_delta", type=float, default=0.0)

    parser.add_argument("--assume_excess_returns", action="store_true")

    # A+B: rolling vs kalman + lambda_tc sweep
    parser.add_argument("--do_sweep", action="store_true")
    parser.add_argument("--sweep_beliefs", type=str, default="both", choices=["both", "single"])
    parser.add_argument("--sweep_lambdas", type=str, default="1e-3,2e-3,5e-3,1e-2")
    parser.add_argument("--sweep_out_csv", type=str, default="sweep_summary.csv")

    args = parser.parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")

    asset_cols = [c.strip() for c in args.asset_cols.split(",") if c.strip() != ""]
    if len(asset_cols) == 0:
        asset_cols = None

    if args.do_sweep:
        run_sweeps(args)
        return

    cfg = TrainConfig(
        seed=args.seed,
        device=device,
        horizon=args.horizon,
        batch_size=args.batch_size,
        iters=args.iters,
        lr=args.lr,
        grad_clip=args.grad_clip,
        gamma=args.gamma,
        beta=args.beta,
        kappa_b=args.kappa_b,
        w0=args.w0,
        w_min=args.w_min,
        use_dt=True,
        dt=float(args.dt),
        rho_max=float(args.rho_max),
        tc_type=args.tc_type,
        lambda_tc=float(args.lambda_tc),
        tc_cap=float(args.tc_cap),
        pi_smooth=float(args.pi_smooth),
        weight_decay=float(args.weight_decay),
        window=args.window,
        shrink_alpha=float(args.shrink_alpha),
        kalman_q=float(args.kalman_q),
        init_P=float(args.init_P),
        robust_block=int(args.robust_block),
        noise_mu=float(args.noise_mu),
        noise_cov=float(args.noise_cov),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        print_every=int(args.print_every),
        save_path=args.save_path,
        early_patience=int(args.early_patience),
        early_min_delta=float(args.early_min_delta),
    )

    _policy, _summary = train_deeponet_merton(
        csv_path=args.csv,
        rf_col=args.rf_col,
        asset_cols=asset_cols,
        belief=args.belief,
        cfg=cfg,
        assume_excess_returns=bool(args.assume_excess_returns),
    )


if __name__ == "__main__":
    main()
