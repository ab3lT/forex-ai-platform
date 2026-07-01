"""Walk-forward analysis: rolling train/test splits with no look-ahead.

`run_walk_forward` takes a model factory so each fold retrains from scratch —
the only honest way to estimate out-of-sample behaviour for a learned strategy.
"""
from __future__ import annotations

from typing import Callable, Iterator

import pandas as pd

from .engine import Backtester, BacktestResult
from .metrics import performance_summary


def walk_forward_splits(
    df: pd.DataFrame, train_bars: int, test_bars: int, step: int | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    step = step or test_bars
    start = 0
    while start + train_bars + test_bars <= len(df):
        yield (df.iloc[start : start + train_bars],
               df.iloc[start + train_bars : start + train_bars + test_bars])
        start += step


def run_walk_forward(
    df: pd.DataFrame,
    make_signal_fn: Callable[[pd.DataFrame], "Callable"],
    train_bars: int = 5000,
    test_bars: int = 1000,
    **bt_kwargs,
) -> pd.DataFrame:
    rows = []
    for k, (train, test) in enumerate(walk_forward_splits(df, train_bars, test_bars)):
        signal_fn = make_signal_fn(train)          # fit only on the past
        bt = Backtester(signal_fn=signal_fn, **bt_kwargs)
        res: BacktestResult = bt.run(pd.concat([train.iloc[-bt.warmup:], test]))
        summary = performance_summary(res.equity_curve, res.trades_df)
        summary["fold"] = k
        rows.append(summary)
    return pd.DataFrame(rows).set_index("fold")
