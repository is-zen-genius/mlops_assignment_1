"""Download the UCI Cleveland Heart Disease dataset to data/raw/.

Usage:
    python src/ingest.py                  # fetch from UCI
    python src/ingest.py --source <path>  # copy/convert a local .data file instead
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

from config import RAW_COLUMNS, RAW_DATA_FILE

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def _parse(raw_bytes: bytes) -> pd.DataFrame:
    from io import StringIO

    text = raw_bytes.decode("utf-8")
    return pd.read_csv(StringIO(text), header=None, names=RAW_COLUMNS, na_values="?")


def fetch_from_uci(url: str = UCI_URL, timeout: int = 30) -> pd.DataFrame:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return _parse(response.content)


def fetch_from_local_file(path: Path) -> pd.DataFrame:
    return _parse(path.read_bytes())


def save(df: pd.DataFrame, out_file: Path = RAW_DATA_FILE) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help="local .data file instead of UCI")
    parser.add_argument("--url", default=UCI_URL, help="override the UCI download URL")
    args = parser.parse_args()

    try:
        if args.source:
            df = fetch_from_local_file(args.source)
        else:
            df = fetch_from_uci(args.url)
    except (requests.RequestException, OSError) as exc:
        print(f"Download failed ({exc}).", file=sys.stderr)
        if RAW_DATA_FILE.exists():
            print(f"Keeping existing {RAW_DATA_FILE} as-is.", file=sys.stderr)
            return
        print(
            "No cached copy found. Download processed.cleveland.data manually from "
            "https://archive.ics.uci.edu/dataset/45/heart+disease and re-run with --source.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_file = save(df)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} cols -> {out_file}")


if __name__ == "__main__":
    main()
