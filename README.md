# Data-Driven Multi-Objective Optimisation

## Overview
This repository provides a framework for data-driven multi-objective optimisation, combining Bayesian Optimisation (BO) with Pareto-front search algorithms such as NSGA-II. It is designed to help explore trade-offs between competing objectives, e.g., performance vs cost, across multiple domains.

## Features
- **Bayesian Optimisation (BO):** Uses Gaussian Process surrogate models and acquisition functions to efficiently explore design spaces.
- **Pareto-Front Search:** Generates non-dominated solutions, mapping the trade-off curve between multiple objectives.
- **Custom Cost Functions:** Supports flexible objective definitions, e.g., `"Cost" = -Gain + α*Area`, or domain-specific metrics.
- **Cross-Industry Applications:** Beyond electronics, applicable to finance, manufacturing, logistics, and other industries requiring multi-objective trade-off analysis.

## Example Applications
- CMOS transistor design for optimal gain vs area.
- Material usage optimisation in manufacturing.
- Portfolio allocation in finance balancing risk vs return.
- Factory production planning optimising throughput vs cost.

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/YourRepoName.git
