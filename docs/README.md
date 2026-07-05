# MCAB: Monte Carlo Simulations for A/B Testing

[![PyPI version](https://img.shields.io/pypi/v/mcab)](https://pypi.org/project/mcab/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

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
