from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/house_price_dataset.csv")
OUTPUT_FILE = Path("data/processed/house_price_modeling.csv")


def main():
    print("=" * 100)
    print("STEP 21F - CONSERVATIVE FEATURE CLEANING")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Load the original dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    print(f"\nOriginal rows: {len(df):,}")

    # Keep original values for reporting
    original = df.copy()

    # ------------------------------------------------------------------
    # 2. Convert clearly invalid feature values to NaN
    #
    # We do NOT delete the transaction.
    # The model pipeline can later impute these missing values.
    # ------------------------------------------------------------------

    cleaning_rules = {
        "bedrooms": 10,
        "bathroom_equivalents": 10,
        "fingarage": 2000,
        "finattic": 2000,
        "heatedarea": 10000,
        "finisharea": 10000,
    }

    print("\nInvalid values converted to NaN:")

    for column, maximum in cleaning_rules.items():
        mask = df[column] > maximum
        count = mask.sum()

        print(f"  {column:25s} > {maximum:8} : {count:6,}")

        df.loc[mask, column] = np.nan

    # ------------------------------------------------------------------
    # 3. Fireplaces
    #
    # We are NOT removing values > 5 yet because the diagnostic showed
    # several 6-fireplace properties that otherwise look plausible.
    # ------------------------------------------------------------------

    print("\nFireplaces:")
    print(
        f"  Values > 5 retained for now: "
        f"{(df['fireplaces'] > 5).sum():,}"
    )

    # ------------------------------------------------------------------
    # 4. Save the cleaned modeling dataset
    # ------------------------------------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 100)
    print("CLEANING COMPLETE")
    print("=" * 100)

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # ------------------------------------------------------------------
    # 5. Verify that the original row count was preserved
    # ------------------------------------------------------------------

    assert len(df) == len(original)

    print("\nRow count preserved: YES")
    print("Original dataset was NOT modified.")

    # ------------------------------------------------------------------
    # 6. Show missing values created by this cleaning step
    # ------------------------------------------------------------------

    print("\nMissing values after cleaning:")

    for column in cleaning_rules:
        missing = df[column].isna().sum()
        print(f"  {column:25s}: {missing:6,}")

    # ------------------------------------------------------------------
    # 7. Verify the extreme values are gone from these features
    # ------------------------------------------------------------------

    print("\nRemaining extreme values:")

    for column, maximum in cleaning_rules.items():
        remaining = (df[column] > maximum).sum()
        print(f"  {column:25s} > {maximum:8} : {remaining:6,}")

    print("\nStep 21F finished successfully.")


if __name__ == "__main__":
    main()