# Stock Portfolio Optimizer

This project compares two ways of choosing 3 stocks from a group of 6:

- Palantir (PLTR)
- SpaceX (SPCX)
- Apple (AAPL)
- Tesla (TSLA)
- Nvidia (NVDA)
- Coca-Cola (KO)

It compares **simulated annealing**, an optimization method, with **exhaustive search**, which checks every possible combination and finds the best one for this small example.

## How it works

The program downloads adjusted daily stock prices from Yahoo Finance, calculates estimated return and risk, and uses those numbers to compare possible portfolios. It prints the results and saves a summary to `results.csv`.

Because SpaceX started trading publicly on June 12, 2026, the program uses prices from that date onward for all six stocks. This gives each stock the same comparison period, but the history is short.

## Run the project

You need Python and an internet connection. In a terminal, run:

```bash
python -m pip install numpy yfinance
python portfolio.py
