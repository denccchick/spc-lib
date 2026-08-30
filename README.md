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

For library capabilities and an example of its operation, see (RU) [https://github.com/denccchick/spc-lib/blob/master/examples/demonstration.html](examples/demonstration.html).
