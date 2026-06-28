"""Monte Carlo robustness analysis via trade-order bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_trades(
    trade_pnls: pd.Series, initial_equity: float, n_sims: int = 2000, seed: int = 7,
) -> pd.DataFrame:
    """Resample trade P&L sequences; return distribution of outcomes."""
    rng = np.random.default_rng(seed)
    pnls = trade_pnls.to_numpy()
    if len(pnls) == 0:
        return pd.DataFrame()
    finals, max_dds = np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        curve = initial_equity + np.cumsum(sample)
        peak = np.maximum.accumulate(np.concatenate([[initial_equity], curve]))
        max_dds[i] = ((np.concatenate([[initial_equity], curve]) - peak) / peak).min()
        finals[i] = curve[-1]
    q = lambda a, p: float(np.percentile(a, p))
    return pd.DataFrame({
        "metric": ["final_equity", "max_drawdown"],
        "p5":  [q(finals, 5),  q(max_dds, 5)],
        "p50": [q(finals, 50), q(max_dds, 50)],
        "p95": [q(finals, 95), q(max_dds, 95)],
        "prob_loss": [float((finals < initial_equity).mean()), np.nan],
    })
