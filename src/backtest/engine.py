"""Event-driven bar backtester.

Replays history bar-by-bar through the *same* signal + risk code path used
live. Models costs as spread (in pips) + slippage. Intrabar SL/TP fills use
the conservative assumption: if both SL and TP are touched in one bar, SL
fills first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from forex_ai.risk import RiskEngine, RiskConfig


class SignalFn(Protocol):
    def __call__(self, window: pd.DataFrame) -> tuple[int, float]:
        """Return (direction, confidence): direction ∈ {-1, 0, +1}."""


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    direction: int
    entry: float
    exit: float | None
    units: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    risk_amount: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]
    returns: pd.Series

    @property
    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


@dataclass
class Backtester:
    signal_fn: SignalFn
    initial_equity: float = 100_000.0
    spread_pips: float = 1.0
    pip_size: float = 0.0001
    slippage_pips: float = 0.2
    warmup: int = 250
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        risk = RiskEngine(self.risk_config)
        equity = self.initial_equity
        cost = (self.spread_pips + self.slippage_pips) * self.pip_size
        open_trade: Trade | None = None
        trades: list[Trade] = []
        eq = np.full(len(df), np.nan)

        for i in range(self.warmup, len(df)):
            ts = df.index[i]
            bar = df.iloc[i]
            risk.on_new_bar(ts.date(), equity)

            # --- manage open position against this bar's range ---
            if open_trade is not None:
                t = open_trade
                hit_sl = bar["low"] <= t.stop_loss if t.direction == 1 else bar["high"] >= t.stop_loss
                hit_tp = bar["high"] >= t.take_profit if t.direction == 1 else bar["low"] <= t.take_profit
                exit_px, why = None, ""
                if hit_sl:                       # conservative: SL first
                    exit_px, why = t.stop_loss, "stop_loss"
                elif hit_tp:
                    exit_px, why = t.take_profit, "take_profit"
                if exit_px is not None:
                    pnl = t.direction * (exit_px - t.entry) * t.units
                    equity += pnl
                    t.exit, t.exit_time, t.pnl, t.reason = exit_px, ts, pnl, why
                    risk.record_trade_result(pnl, t.risk_amount)
                    trades.append(t)
                    open_trade = None

            # --- new signal on bar close ---
            if open_trade is None and not risk.daily_drawdown_breached(equity):
                direction, conf = self.signal_fn(df.iloc[: i + 1])
                if direction != 0:
                    plan = risk.plan_position(
                        equity=equity, price=float(bar["close"]),
                        atr=float(bar["atr_14"]), direction=direction,
                        confidence=conf,
                    )
                    if not plan.rejected:
                        entry = float(bar["close"]) + direction * cost  # pay the spread
                        open_trade = Trade(
                            entry_time=ts, exit_time=None, direction=direction,
                            entry=entry, exit=None, units=plan.units,
                            stop_loss=plan.stop_loss, take_profit=plan.take_profit,
                            risk_amount=plan.risk_amount,
                        )

            mtm = 0.0
            if open_trade is not None:
                mtm = open_trade.direction * (bar["close"] - open_trade.entry) * open_trade.units
            eq[i] = equity + mtm

        curve = pd.Series(eq, index=df.index).dropna()
        returns = curve.pct_change().dropna()
        return BacktestResult(equity_curve=curve, trades=trades, returns=returns)
