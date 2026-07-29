"""Rotor — the crypto momentum-rotation sleeve's strategy + backtest package.

`backtest.py` is the whole v1 engine: point-in-time universe, weekly
rotation, BTC trend gate, costs, metrics, grid, and robust selection. Pure
and offline — inputs come from the stored data/rotor Parquet via
data_layer.rotor. (Distinct from `data_layer/rotor.py`, the data loader.)
"""
