"""
ingest.py
Downloads international football data from the martj42 GitHub repository.
No API keys required — all data is publicly available via raw GitHub URLs.
"""

import requests
import pandas as pd
from pathlib import Path
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master"
)

SOURCES = {
    "results.csv": f"{BASE_URL}/results.csv",
    "goalscorers.csv": f"{BASE_URL}/goalscorers.csv",
    "shootouts.csv": f"{BASE_URL}/shootouts.csv",
}

# Minimum expected rows — pipeline fails if data looks truncated
MIN_ROWS = {
    "results.csv": 45_000,
    "goalscorers.csv": 40_000,
    "shootouts.csv": 500,
}


# ── Functions ─────────────────────────────────────────────────────────────────

def download_file(url: str, destination: Path, timeout: int = 30) -> pd.DataFrame:
    """
    Download a CSV from a URL and save it locally.
    Retries up to 3 times on failure.
    """
    for attempt in range(1, 4):
        try:
            logger.info(f"Downloading {destination.name} (attempt {attempt}/3)...")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.content)

            df = pd.read_csv(destination)
            logger.info(f"  ✓ {destination.name}: {len(df):,} rows downloaded")
            return df

        except requests.RequestException as e:
            logger.warning(f"  Attempt {attempt} failed: {e}")
            if attempt == 3:
                raise RuntimeError(
                    f"Failed to download {destination.name} after 3 attempts"
                ) from e


def validate_download(df: pd.DataFrame, filename: str) -> None:
    """
    Basic sanity checks on downloaded data.
    Fails loudly rather than silently passing bad data downstream.
    """
    min_rows = MIN_ROWS.get(filename, 0)
    if len(df) < min_rows:
        raise ValueError(
            f"{filename} has only {len(df):,} rows — "
            f"expected at least {min_rows:,}. Download may be corrupt."
        )
    logger.info(f"  ✓ {filename} validation passed ({len(df):,} rows)")


def ingest_all(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Download all source files and return as a dict of DataFrames.
    Skips download if file already exists and is recent (< 12 hours old).
    """
    import time

    dataframes = {}

    for filename, url in SOURCES.items():
        destination = raw_dir / filename
        twelve_hours = 12 * 60 * 60

        # Incremental loading: skip if recently downloaded
        if destination.exists():
            age = time.time() - destination.stat().st_mtime
            if age < twelve_hours:
                logger.info(
                    f"Skipping {filename} — downloaded "
                    f"{age/3600:.1f}h ago (< 12h threshold)"
                )
                dataframes[filename] = pd.read_csv(destination)
                continue

        df = download_file(url, destination)
        validate_download(df, filename)
        dataframes[filename] = df

    logger.info(f"Ingestion complete — {len(dataframes)} files loaded")
    return dataframes