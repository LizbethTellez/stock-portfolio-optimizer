"""Compare exact enumeration with simulated annealing on real stock-price data."""

from __future__ import annotations

import csv
import itertools
import random
import time
from pathlib import Path

import numpy as np
import yfinance as yf


TICKERS = ["PLTR", "SPCX", "AAPL", "TSLA", "NVDA", "KO"]
START_DATE = "2026-06-12"  # SpaceX began public trading on this date.
ASSETS_TO_HOLD = 3
RISK_AVERSION = 4.0
CARDINALITY_PENALTY = 10.0
SEED = 17
RESULT_FILE = Path(__file__).resolve().parent / "results.csv"


def download_prices() -> tuple[list[str], np.ndarray, list[str]]:
    """Download adjusted daily closes and keep dates shared by every stock."""
    frame = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame.empty or "Close" not in frame:
        raise RuntimeError("No prices returned. Check your internet connection and ticker symbols.")

    closes = frame["Close"]
    if getattr(closes, "ndim", 1) == 1:
        closes = closes.to_frame(name=TICKERS[0])
    missing = [ticker for ticker in TICKERS if ticker not in closes.columns]
    if missing:
        raise RuntimeError(f"No price data returned for: {', '.join(missing)}")

    # Keeping common trading days gives every ticker the same comparison window.
    closes = closes[TICKERS].dropna()
    if len(closes) < 3:
        raise RuntimeError("Too few shared trading days to estimate portfolio risk and return.")
    return TICKERS, closes.to_numpy(dtype=float), [str(day.date()) for day in closes.index]


def portfolio_metrics(selection: np.ndarray, mean_returns: np.ndarray,
                      covariance: np.ndarray) -> tuple[float, float]:
    daily_return = float(selection @ mean_returns)
    daily_variance = float(selection @ covariance @ selection)
    return daily_return * 252, np.sqrt(max(daily_variance, 0.0) * 252)


def energy(selection: np.ndarray, mean_returns: np.ndarray,
           covariance: np.ndarray) -> float:
    expected_return = float(selection @ mean_returns)
    risk = float(selection @ covariance @ selection)
    count_penalty = CARDINALITY_PENALTY * (int(selection.sum()) - ASSETS_TO_HOLD) ** 2
    return RISK_AVERSION * risk - expected_return + count_penalty


def exact_search(n_assets: int, mean_returns: np.ndarray,
                 covariance: np.ndarray) -> np.ndarray:
    best = None
    best_score = float("inf")
    for chosen in itertools.combinations(range(n_assets), ASSETS_TO_HOLD):
        candidate = np.zeros(n_assets, dtype=int)
        candidate[list(chosen)] = 1
        score = energy(candidate, mean_returns, covariance)
        if score < best_score:
            best, best_score = candidate, score
    assert best is not None
    return best


def simulated_annealing(n_assets: int, mean_returns: np.ndarray,
                        covariance: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    current = np.array([1] * ASSETS_TO_HOLD + [0] * (n_assets - ASSETS_TO_HOLD))
    rng.shuffle(current)
    best = current.copy()
    current_energy = energy(current, mean_returns, covariance)
    best_energy = current_energy

    for step in range(12000):
        temperature = 0.03 * (0.0001 / 0.03) ** (step / 11999)
        i, j = rng.sample(range(n_assets), 2)
        candidate = current.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        candidate_energy = energy(candidate, mean_returns, covariance)
        delta = candidate_energy - current_energy
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            current, current_energy = candidate, candidate_energy
        if current_energy < best_energy:
            best, best_energy = current.copy(), current_energy
    return best


def main() -> None:
    assets, prices, dates = download_prices()
    daily_returns = prices[1:] / prices[:-1] - 1
    mean_returns = daily_returns.mean(axis=0)
    covariance = np.cov(daily_returns, rowvar=False)
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError("Could not estimate risk from the downloaded data.")

    methods = []
    for name, solver in (
        ("Simulated annealing", lambda: simulated_annealing(len(assets), mean_returns, covariance, SEED)),
        ("Exhaustive search", lambda: exact_search(len(assets), mean_returns, covariance)),
    ):
        start = time.perf_counter()
        selection = solver()
        elapsed = time.perf_counter() - start
        annual_return, volatility = portfolio_metrics(selection, mean_returns, covariance)
        methods.append({
            "method": name,
            "assets": ", ".join(asset for asset, selected in zip(assets, selection) if selected),
            "selected_count": int(selection.sum()),
            "annualized_return_estimate": annual_return,
            "annualized_volatility_estimate": volatility,
            "qubo_energy": energy(selection, mean_returns, covariance),
            "runtime_seconds": elapsed,
        })

    print(f"Price source: Yahoo Finance via yfinance; adjusted closes")
    print(f"Shared data window: {dates[0]} through {dates[-1]} ({len(daily_returns)} daily returns)")
    print(f"Assets: {', '.join(assets)} | choose {ASSETS_TO_HOLD} | risk aversion: {RISK_AVERSION}\n")
    for row in methods:
        print(row["method"])
        print(f"  Selected: {row['assets']}")
        print(f"  Annualized return estimate: {row['annualized_return_estimate']:.2%}")
        print(f"  Annualized volatility estimate: {row['annualized_volatility_estimate']:.2%}")
        print(f"  QUBO energy: {row['qubo_energy']:.6f}")
        print(f"  Runtime: {row['runtime_seconds']:.4f} seconds")
    print("\nExhaustive search is the exact optimum for this small problem.")
    print("These estimates use a short shared history and do not predict future performance.")

    with RESULT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=methods[0].keys())
        writer.writeheader()
        writer.writerows(methods)
    print(f"Wrote {RESULT_FILE.name}")


if __name__ == "__main__":
    main()
