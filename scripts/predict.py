"""Command-line forecast from the saved model bundle.

    py scripts/predict.py                    # best model, trained horizon
    py scripts/predict.py --model random_forest
    py scripts/predict.py --json

Useful for cron jobs, and as a no-dependency sanity check that the artefacts
the web apps load actually work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import service  # noqa: E402
from src.service import ModelNotTrained  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict the next gold move")
    ap.add_argument("--model", default=None, help="trained model name")
    ap.add_argument("--refresh", action="store_true", help="re-download market data")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    try:
        f = service.forecast(model_name=args.model, refresh_data=args.refresh)
    except ModelNotTrained as exc:
        sys.exit(f"{exc}\nTrain first:  py -m src.pipeline")

    payload = {k: v for k, v in f.items() if k not in ("history",)}

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    ret = f["predicted_return"]
    arrow = "▲" if ret > 0 else ("▼" if ret < 0 else "→")
    print("=" * 60)
    print(f"Gold forecast  |  model={f['model']}  |  {f['trained_horizon_days']} business days")
    print("=" * 60)
    print(f"  as of          {f['as_of_date']}")
    print(f"  entry price    ${f['entry_price_usd']:,.2f} / oz")
    print(f"  predicted      ${f['predicted_price_usd']:,.2f} / oz   {arrow} {ret * 100:+.2f}%")
    print(f"  target date    {f['target_date']}")
    m = f.get("metrics") or {}
    if m:
        print("-" * 60)
        print(f"  held-out RMSE  {m.get('rmse_ret', float('nan')):.5f} (return)  "
              f"{m.get('rmse_usd', float('nan')):.2f} USD")
        print(f"  R2={m.get('r2_ret', float('nan')):+.4f}  "
              f"MAE={m.get('mae_ret', float('nan')):.5f}  "
              f"skill vs naive={m.get('skill_vs_naive', float('nan')):+.4f}")
    print("=" * 60)
    print("Not investment advice. A near-zero skill score means no edge.")


if __name__ == "__main__":
    main()
