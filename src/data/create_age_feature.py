import pandas as pd


INPUT_FILE = "data/processed/house_price_modeling.csv"
OUTPUT_FILE = "data/processed/house_price_modeling.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)

    # Create property age at the time of sale.
    df["property_age"] = df["sale_year"] - df["yearbuilt"]

    # Invalid ages should not be used.
    # A negative age means the recorded construction year
    # is after the sale year.
    invalid_age = df["property_age"] < 0

    print("=" * 70)
    print("PROPERTY AGE FEATURE")
    print("=" * 70)

    print(f"\nOriginal rows: {original_rows:,}")

    print(
        f"Negative property ages found: "
        f"{invalid_age.sum():,}"
    )

    if invalid_age.sum() > 0:
        df.loc[invalid_age, "property_age"] = pd.NA

    print("\nProperty age statistics:")
    print(
        df["property_age"]
        .describe()
        .to_string()
    )

    print(
        f"\nMissing property_age: "
        f"{df['property_age'].isna().sum():,}"
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if len(df) == original_rows:
        print("Row count preserved: YES")
    else:
        print("WARNING: Row count changed!")

    print("\nFeature creation complete.")


if __name__ == "__main__":
    main()