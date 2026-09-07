from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "house_price_modeling_spatial.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "reference"
OUTPUT_FILE = OUTPUT_DIR / "location_lookup.csv"


def main():
    print(f"Reading: {SOURCE_FILE}")

    df = pd.read_csv(
        SOURCE_FILE,
        usecols=[
            "parcelid",
            "latitude",
            "longitude",
            "neighborhood",
        ],
    )

    print(f"Rows loaded: {len(df):,}")

    df = df.dropna(
        subset=[
            "parcelid",
            "latitude",
            "longitude",
            "neighborhood",
        ]
    )

    df = df.drop_duplicates(
        subset=[
            "parcelid",
            "latitude",
            "longitude",
            "neighborhood",
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Rows saved: {len(df):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()