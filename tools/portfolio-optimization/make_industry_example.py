"""Offline: build portfolio-optimization/examples/us_industries_12.json from the Kenneth R. French
Data Library. Kept outside the workflow: it is data preparation, not part of the workflow.

Source: "12 Industry Portfolios" (value-weighted monthly returns), Copyright Eugene F. Fama and
Kenneth R. French, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Only derived statistics are written (annualised mean returns and covariance), not the series.

    python tools/portfolio-optimization/make_industry_example.py 12_Industry_Portfolios.csv \
        [first=201501] [last=202412]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NAMES = {
    "NoDur": "Consumer non-durables", "Durbl": "Consumer durables", "Manuf": "Manufacturing",
    "Enrgy": "Energy", "Chems": "Chemicals", "BusEq": "Business equipment (tech)",
    "Telcm": "Telecom", "Utils": "Utilities", "Shops": "Retail and wholesale",
    "Hlth": "Healthcare", "Money": "Finance", "Other": "Other",
}  # fmt: skip


def monthly_value_weighted(path: Path) -> tuple[list[str], dict[int, list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, ln in enumerate(lines) if "Average Value Weighted Returns -- Monthly" in ln)
    header = [h.strip() for h in lines[start + 1].split(",")[1:]]
    rows: dict[int, list[float]] = {}
    for ln in lines[start + 2 :]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header) + 1 or not parts[0].isdigit() or len(parts[0]) != 6:
            break  # the monthly table ends at the first non-monthly row
        values = [float(p) for p in parts[1:]]
        if any(v <= -99.99 for v in values):
            raise ValueError(f"missing data in {parts[0]}")
        rows[int(parts[0])] = values
    return header, rows


def main(csv: str, first: int = 201501, last: int = 202412) -> None:
    header, rows = monthly_value_weighted(Path(csv))
    months = [m for m in sorted(rows) if first <= m <= last]
    if len(months) != (last // 100 - first // 100) * 12 + (last % 100 - first % 100) + 1:
        raise ValueError("months missing from the requested window")
    r = np.array([rows[m] for m in months]) / 100.0
    out = {
        "assets": [NAMES[h] for h in header],
        "expected_returns": [round(float(x), 5) for x in r.mean(axis=0) * 12],
        "covariance": [[round(float(x), 6) for x in row] for row in np.cov(r, rowvar=False) * 12],
        "num_to_select": 4,
        "risk_aversion": 1.0,
    }
    dest = Path(__file__).resolve().parents[2] / "portfolio-optimization" / "examples" / "us_industries_12.json"
    dest.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{len(months)} months {months[0]}-{months[-1]} -> {dest.name}")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
