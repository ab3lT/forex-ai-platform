"""Performance metrics: Sharpe, Sortino, drawdown, win rate, profit factor."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualisation_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    bar_sec = np.median(np.diff(index.view("int64"))) / 1e9
    return (365.25 * 24 * 3600) / max(bar_sec, 1.0) * (5 / 7)  # FX trades ~5/7 days


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    af = annualisation_factor(returns.index)
    return float((returns.mean() - rf / af) / returns.std() * np.sqrt(af))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    af = annualisation_factor(returns.index)
    return float((returns.mean() - rf / af) / downside.std() * np.sqrt(af))


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1
    return float(dd.min())


def performance_summary(equity: pd.Series, trades: pd.DataFrame) -> dict:
    returns = equity.pct_change().dropna()
    closed = trades[trades["pnl"].notna()] if len(trades) else trades
    wins = closed[closed["pnl"] > 0] if len(closed) else closed
    losses = closed[closed["pnl"] <= 0] if len(closed) else closed
    gross_win = wins["pnl"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0.0
    return {
        "final_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0,
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "n_trades": int(len(closed)),
        "win_rate": float(len(wins) / len(closed)) if len(closed) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
    }
