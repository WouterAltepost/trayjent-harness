"""``python -m runner`` entry point -> the backtest CLI."""
import sys

from runner.cli import main

sys.exit(main())
