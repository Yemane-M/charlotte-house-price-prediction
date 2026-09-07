import pandas as pd


INPUT_FILE = "data/processed/house_price_modeling.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("PROPERTY AGE ANALYSIS")
    print("=" * 80)

    print(f"\nTotal rows: {len(df):,}")

    # ---------------------------------------------------------
    # 1. Property age summary
    # ---------------------------------------------------------
    print("\nPROPERTY AGE SUMMARY")

    print(
        df["property_age"]
        .describe()
        .to_string()
    )

    # ---------------------------------------------------------
    # 2. Age distribution
    # ---------------------------------------------------------
    print("\nPROPERTY AGE DISTRIBUTION")

    age_distribution = (
        df["property_age"]
        .value_counts()
        .sort_index()
        .head(50)
    )

    print(age_distribution.to_string())

    # ---------------------------------------------------------
    # 3. Age groups
    # ---------------------------------------------------------
    df["age_group"] = pd.cut(
        df["property_age"],
        bins=[-1, 0, 5, 10, 20, 30, 50, 100, float("inf")],
        labels=[
            "0",
            "1-5",
            "6-10",
            "11-20",
            "21-30",
            "31-50",
            "51-100",
            "100+",
        ],
    )

    print("\nPROPERTY AGE GROUPS")

    print(
        df["age_group"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # 4. Sale price by age group
    # ---------------------------------------------------------
    print("\nSALE PRICE BY PROPERTY AGE GROUP")

    age_price = (
        df.groupby(
            "age_group",
            observed=False
        )["saleprice"]
        .agg(
            count="count",
            median="median",
            mean="mean",
        )
    )

    print(age_price.to_string())

    # ---------------------------------------------------------
    # 5. Correlations
    # ---------------------------------------------------------
    print("\nCORRELATION WITH SALE PRICE")

    correlation_columns = [
        "saleprice",
        "property_age",
        "yearbuilt",
        "effyearblt",
        "sale_year",
    ]

    print(
        df[correlation_columns]
        .corr()["saleprice"]
        .sort_values(ascending=False)
        .to_string()
    )

    # ---------------------------------------------------------
    # 6. Missing property age
    # ---------------------------------------------------------
    missing_age = df[df["property_age"].isna()].copy()

    print("\nMISSING PROPERTY AGE")

    print(
        f"Records with missing property_age: "
        f"{len(missing_age):,}"
    )

    print(
        f"Percentage missing: "
        f"{len(missing_age) / len(df) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # 7. Missing age by sale year
    # ---------------------------------------------------------
    print("\nMISSING PROPERTY AGE BY SALE YEAR")

    missing_by_year = (
        missing_age["sale_year"]
        .value_counts()
        .sort_index()
    )

    print(missing_by_year.to_string())

    # ---------------------------------------------------------
    # 8. Very old properties
    # ---------------------------------------------------------
    print("\nVERY OLD PROPERTIES")

    old_properties = (
        df[df["property_age"] >= 100]
        .sort_values("property_age", ascending=False)
        .head(20)
    )

    columns = [
        "parcelid",
        "saleprice",
        "sale_year",
        "yearbuilt",
        "effyearblt",
        "property_age",
        "heatedarea",
        "finisharea",
        "bedrooms",
        "bathroom_equivalents",
        "landusefulldescription",
        "neighborhood",
    ]

    columns = [
        column
        for column in columns
        if column in old_properties.columns
    ]

    print(
        old_properties[columns]
        .to_string(index=False)
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()