"""Parse NOAA PSL monthly climate-index files (Nino 3.4, DMI) into a tidy table.

Source files (downloaded 2026-11-12, see docs/data_sources.md):
- data/raw/climate_indices/nina34.anom.data  (ERSST v6 Nino 3.4 anomaly, 1981-2010 base)
- data/raw/climate_indices/dmi.had.long.data  (HadISST1.1 Dipole Mode Index)

PSL format: first line "start_year end_year", then one row per year with 12 monthly
values, then a line holding the missing-value code, then free-text notes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/climate_indices")
INDEX_FILES = {
    "nino34": RAW_DIR / "nina34.anom.data",
    "dmi": RAW_DIR / "dmi.had.long.data",
}


def parse_psl_file(path: Path, name: str) -> pd.DataFrame:
    """Return a long DataFrame (year, month, <name>) with missing values as NaN."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start, end = (int(x) for x in lines[0].split()[:2])
    n_years = end - start + 1
    rows = []
    for line in lines[1 : 1 + n_years]:
        parts = line.split()
        year = int(parts[0])
        values = [float(v) for v in parts[1:13]]
        if len(values) != 12:
            raise ValueError(f"{path}: year {year} has {len(values)} values, expected 12")
        rows.extend((year, m, v) for m, v in enumerate(values, start=1))
    missing_code = float(lines[1 + n_years].split()[0])

    df = pd.DataFrame(rows, columns=["year", "month", name])
    df.loc[np.isclose(df[name], missing_code), name] = np.nan
    if df["year"].min() != start or df["year"].max() != end:
        raise ValueError(f"{path}: header says {start}-{end}, rows say {df['year'].min()}-{df['year'].max()}")
    return df


def load_climate_indices(files: dict[str, Path] | None = None) -> pd.DataFrame:
    """Merge all index files into one monthly table keyed by (year, month)."""
    files = files or INDEX_FILES
    out = None
    for name, path in files.items():
        df = parse_psl_file(path, name)
        out = df if out is None else out.merge(df, on=["year", "month"], how="outer")
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def main() -> None:
    df = load_climate_indices()
    out = Path("data/processed/climate_indices_monthly.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} rows; non-missing counts: {df.drop(columns=['year', 'month']).notna().sum().to_dict()}")


if __name__ == "__main__":
    main()
