import pandas as pd


INPUT_FILE = "data/processed/house_price_modeling.csv"

TARGET = "saleprice"

FEATURES = [
    "sale_year",
    "gisacres",
    "heatedarea",
    "finisharea",
    "bedrooms",
    "bathroom_equivalents",
    "fireplaces",
    "fingarage",
    "finattic",
    "property_age",
    "landusefulldescription",
]


def main():
    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("FINAL FEATURE REVIEW")
    print("=" * 80)

    print(f"\nDataset rows: {len(df):,}")
    print(f"Dataset columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # Check target
    # ---------------------------------------------------------
    print("\nTARGET")
    print(f"  {TARGET}")
    print(f"  Data type: {df[TARGET].dtype}")
    print(f"  Missing: {df[TARGET].isna().sum():,}")

    # ---------------------------------------------------------
    # Check features
    # ---------------------------------------------------------
    print("\nFEATURES")

    missing_columns = []

    for feature in FEATURES:
        if feature not in df.columns:
            missing_columns.append(feature)
            continue

        print(
            f"  {feature:28s}"
            f" dtype={str(df[feature].dtype):10s}"
            f" missing={df[feature].isna().sum():,}"
        )

    # ---------------------------------------------------------
    # Check missing columns
    # ---------------------------------------------------------
    if missing_columns:
        print("\nWARNING - MISSING FEATURES:")

        for column in missing_columns:
            print(f"  {column}")

    else:
        print("\nAll planned features are present.")

    # ---------------------------------------------------------
    # Feature count
    # ---------------------------------------------------------
    print(
        f"\nNumber of planned predictors: "
        f"{len(FEATURES)}"
    )

    # ---------------------------------------------------------
    # Numeric summary
    # ---------------------------------------------------------
    numeric_features = [
        feature
        for feature in FEATURES
        if feature in df.columns
        and pd.api.types.is_numeric_dtype(df[feature])
    ]

    print("\nNUMERIC FEATURE SUMMARY")

    print(
        df[numeric_features]
        .describe()
        .T
        .to_string()
    )

    # ---------------------------------------------------------
    # Categorical summary
    # ---------------------------------------------------------
    categorical_features = [
        feature
        for feature in FEATURES
        if feature in df.columns
        and not pd.api.types.is_numeric_dtype(df[feature])
    ]

    print("\nCATEGORICAL FEATURES")

    for feature in categorical_features:
        print(f"\n{feature}:")
        print(
            df[feature]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nFinal feature review complete.")


if __name__ == "__main__":
    main()