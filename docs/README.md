# MCAB: Monte Carlo Simulations for A/B Testing

[![PyPI version](https://img.shields.io/pypi/v/mcab)](https://pypi.org/project/mcab/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

![MCAB Logo](https://raw.githubusercontent.com/sslastochkin/mcab/refs/heads/main/docs/img/logo/mcab_logo_adaptive.svg)

A Python toolkit to design and validate A/B experiments via simulation: estimate power, find MDE, and control type-I error. Supports iid &amp; ratio metrics, linearization, CUPED/CUPAC variance reduction, bootstrap tests, permutation tests, multiple-testing corrections, and AA/AB benchmarking.

## Quick Start

You can install the package directly from PyPI:

```bash
pip install mcab
```

## Main Features
- Fast and computatively efficient Monte Carlo, faster then `for i ...`
- Multiprocessing supported with `n_jobs=-1`
- Different effect strategies compatible with any variance reduction
- Flexible api to reduce variance with ML-models
- Iid and Ratio metrics simulations
- Multiple testing corrections simulations
- Ready emperical tests like bootstrap and permutations with optionally multiprocessing
- Plots and other diagnostics
- Fast and straight way to compare multiple designs on one data to challenge the best

## Why you need MCAB?
1. Get results of simulartions 2x faster then for loop approach
2. Don't waste your time to copy a lot of code between jupyter notebooks
3. Get additional diagnostical data about your tests and simulations validity (see tutorial)
4. Fast and easy way to challenge different effect strategies within any metrics
5. Challenge any variance reduction techniques and any CUPAC setups
6. Challenge usual experiments and multi tests comparisons via Monte Carlo with any p-value correction
7. Fast way to compare different designs
8. Works with both iid and ratio metrics out of the box — no manual delta method boilerplate
9. Built-in type I error validity checks with color-coded diagnostics (green / yellow / red)
   — know immediately if your simulation setup is statistically sound
10. Non-parametric tests included — bootstrap and permutation with optional multiprocessing
11. Reproducible experiments — seeded random data generation via RandomData class
12. One API for all metric types — proportion, continuous, ratio (CTR, avg order),
    CUPED/CUPAC — swap strategies without rewriting simulations
13. Scales linearly with n_sims; at 20 000 simulations n_jobs=-1 is 6× faster than for loop
    even starting from a cold process pool

### Computational benchmark
![MCAB Benchmark](https://raw.githubusercontent.com/sslastochkin/mcab/refs/heads/main/benchmark/mcab_benchmark.png)
