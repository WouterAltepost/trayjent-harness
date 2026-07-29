"""Calm — the shortvol sleeve's strategy + backtest package.

`backtest.py` is the whole v1 engine: signal, state machine, sizing, cost
model, metrics, grid, and robust-neighborhood selection. Pure and offline —
inputs come from the stored data/shortvol Parquet via data_layer.shortvol.
"""
