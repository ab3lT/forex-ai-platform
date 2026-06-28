"""Train + walk-forward-validate the GBM direction model.

python scripts/train_ml.py --csv data/EURUSD_H1.csv
(or omit --csv to use synthetic data for a pipeline check)
"""
import argparse

import pandas as pd

from forex_ai.backtest.walk_forward import run_walk_forward
from forex_ai.data.feeds import synthetic_ohlcv
from forex_ai.features import add_indicators
from forex_ai.models.gbm import GBMDirectionModel


def make_signal_fn(train: pd.DataFrame):
    model = GBMDirectionModel().fit(train)
    return lambda window: model.signal_for_last_bar(window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    args = ap.parse_args()
    df = (pd.read_csv(args.csv, parse_dates=[0], index_col=0)
          if args.csv else synthetic_ohlcv(12000))
    df = add_indicators(df)
    report = run_walk_forward(df, make_signal_fn, train_bars=4000, test_bars=1000)
    print(report.to_string())
    print("\nmean OOS sharpe:", report["sharpe"].mean().round(3))


if __name__ == "__main__":
    main()
