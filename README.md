# spc-lib
Statistical Process Control (SPC) library for quality monitoring in manufacturing and research.

Overview
spc-lib is a Python library for building control charts and monitoring process stability. It implements the most commonly used SPC tools.

Key features:

- Variables control charts — X-bar & R, X-bar & s, I-MR
- Attributes control charts — p, c
- Time-weighted charts — CUSUM (standard and variance), EWMA
- Western Electric rules — 8 rules with violation detection
- Process capability — Cp, Cpk, Cpl, Cpu
- Plotly-based visualization — interactive charts with zone highlighting

Installation
```bash
pip install spc-lib
```

For library capabilities and detailed examples, see the demonstration notebook (RU).

## Quick Start

### I-MR Chart

```python
import pandas as pd
import numpy as np
import plotly.io as pio

from spc_lib import IMRChart


df = pd.read_csv(
    "../data/data_with_violations.csv",
    parse_dates=["Date"]
)

df.head()
```

| Date       | Point 1   | Point 2   | Point 3   | Point 4   | Point 5   |
|------------|-----------|-----------|-----------|-----------|-----------|
| 2024-01-01 | 56.636941 | 47.950078 | 59.845189 | 72.393290 | 46.806996 |
| 2024-01-02 | 45.657186 | 73.655152 | 60.906013 | 42.631966 | 58.162100 |
| 2024-01-03 | 42.618528 | 42.821776 | 54.132580 | 21.012350 | 24.544079 |
| 2024-01-04 | 41.000834 | 35.072435 | 55.434494 | 35.143817 | 28.416997 |
| 2024-01-05 | 72.273268 | 46.511833 | 51.198496 | 28.326785 | 41.877554 |

```python
plate_columns = [
    "Point 1",
    "Point 2",
    "Point 3",
    "Point 4",
    "Point 5"
]

data = df[plate_columns].to_numpy()

dates = df["Date"]

chart = IMRChart(
    data=data, # A two‑dimensional array is used as the data input (including for I‑Mr maps, where the array has the form [[0.93],[0.89],... ]). The data is aggregated by rows.
    datetimes=dates
)

chart.fit(method="classic", baseline_mask=None) # it's possible to use the fit method to calculate the control limits.
# The method parameter can be set to "classic", "made" or "percentile". The default is "classic".
# baseline_mask is a boolean array that indicates which data points should be used to calculate the baseline. If None (Default), all data points are used.

fig_i, fig_mr = chart.plot(last_n=50)

fig_i.show()
```
![i-chart](https://raw.githubusercontent.com/denccchick/spc_lib/master/examples/images/i_chart.gif)


```python
fig_rules = chart.plot_rules(last_n=100)
fig_rules.show()
```
![violation-rules](https://raw.githubusercontent.com/denccchick/spc_lib/master/examples/images/violations.gif)
