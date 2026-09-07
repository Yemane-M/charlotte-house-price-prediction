import pandas as pd


INPUT_FILE = "data/processed/house_price_modeling.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("=" * 60)
    print("CLEANED DATASET VERIFICATION")
    print("=" * 60)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")
    for column in [
        "bedrooms",
        "bathroom_equivalents",
        "fingarage",
        "finattic",
        "heatedarea",
        "finisharea",
        "yearbuilt",
        "effyearblt",
    ]:
        print(f"  {column:25s}: {df[column].isna().sum():,}")

    print("\nRemaining extreme values:")

    checks = {
        "heatedarea > 10,000": (df["heatedarea"] > 10000).sum(),
        "finisharea > 10,000": (df["finisharea"] > 10000).sum(),
        "bedrooms > 10": (df["bedrooms"] > 10).sum(),
        "bathroom_equivalents > 10": (
            df["bathroom_equivalents"] > 10
        ).sum(),
        "fingarage > 2,000": (df["fingarage"] > 2000).sum(),
        "finattic > 2,000": (df["finattic"] > 2000).sum(),
    }

    for description, count in checks.items():
        print(f"  {description:30s}: {count:,}")

    print("\nFireplaces:")
    print(f"  Maximum: {df['fireplaces'].max()}")
    print(f"  Values > 5: {(df['fireplaces'] > 5).sum():,}")

    print("\nArea statistics:")
    print(df[["heatedarea", "finisharea"]].describe())

    print("\nVerification complete.")


if __name__ == "__main__":
    main()