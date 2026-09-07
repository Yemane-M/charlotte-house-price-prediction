from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/mecklenburg")

FILES = [
    "Parcel_Sales_Table.csv",
    "Cama_Table.csv",
]


def inspect_csv(filename: str) -> None:
    file_path = DATA_DIR / filename

    print("=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    # Read only a small sample first.
    df = pd.read_csv(
        file_path,
        nrows=5,
        low_memory=False,
    )

    print(f"\nNumber of columns: {len(df.columns)}")

    print("\nCOLUMN NAMES:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nDATA TYPES:")
    print(df.dtypes.to_string())

    print("\nFIRST 5 ROWS:")
    print(df.to_string())

    print()


def main() -> None:
    for filename in FILES:
        inspect_csv(filename)


if __name__ == "__main__":
    main()