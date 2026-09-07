import pandas as pd


INPUT_FILE = "data/processed/house_price_modeling.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("YEAR FEATURE ANALYSIS")
    print("=" * 80)

    print(f"\nTotal rows: {len(df):,}")

    # ---------------------------------------------------------
    # 1. Missing values
    # ---------------------------------------------------------
    print("\nMISSING VALUES")

    print(f"  yearbuilt   : {df['yearbuilt'].isna().sum():,}")
    print(f"  effyearblt  : {df['effyearblt'].isna().sum():,}")

    # ---------------------------------------------------------
    # 2. Calculate effective-year difference
    # ---------------------------------------------------------
    df["effective_year_difference"] = (
        df["effyearblt"] - df["yearbuilt"]
    )

    print("\nYEAR DIFFERENCE (effyearblt - yearbuilt)")

    print(
        df["effective_year_difference"]
        .describe()
        .to_string()
    )

    # ---------------------------------------------------------
    # 3. Effective year after sale year
    # ---------------------------------------------------------
    future_effective = df[
        df["effyearblt"] > df["sale_year"]
    ].copy()

    print("\nEFFECTIVE YEAR AFTER SALE YEAR")

    print(
        f"  Records where effyearblt > sale_year: "
        f"{len(future_effective):,}"
    )

    # ---------------------------------------------------------
    # 4. Difference between effective year and sale year
    # ---------------------------------------------------------
    future_effective["years_after_sale"] = (
        future_effective["effyearblt"]
        - future_effective["sale_year"]
    )

    print("\nHOW FAR AFTER THE SALE YEAR?")

    print(
        future_effective["years_after_sale"]
        .describe()
        .to_string()
    )

    # ---------------------------------------------------------
    # 5. Distribution by sale year
    # ---------------------------------------------------------
    print("\nFUTURE EFFECTIVE YEARS BY SALE YEAR")

    by_sale_year = (
        future_effective
        .groupby("sale_year")
        .size()
        .sort_index()
    )

    print(by_sale_year.to_string())

    # ---------------------------------------------------------
    # 6. Percentage by sale year
    # ---------------------------------------------------------
    print("\nPERCENTAGE OF SALES WITH effyearblt > sale_year")

    total_by_sale_year = (
        df.groupby("sale_year")
        .size()
        .sort_index()
    )

    percentage_by_year = (
        future_effective
        .groupby("sale_year")
        .size()
        .div(total_by_sale_year)
        .mul(100)
    )

    percentage_by_year = percentage_by_year.fillna(0)

    for year, percentage in percentage_by_year.items():
        print(f"  {int(year)}: {percentage:.2f}%")

    # ---------------------------------------------------------
    # 7. Inspect the largest inconsistencies
    # ---------------------------------------------------------
    print("\nLARGEST effyearblt - sale_year DIFFERENCES")

    columns = [
        "parcelid",
        "saleprice",
        "sale_year",
        "yearbuilt",
        "effyearblt",
        "effective_year_difference",
        "gisacres",
        "heatedarea",
        "finisharea",
        "bedrooms",
        "bathroom_equivalents",
        "landusefulldescription",
        "neighborhood",
        "cama_record_count",
    ]

    columns = [
        column
        for column in columns
        if column in future_effective.columns
    ]

    largest_differences = (
        future_effective
        .sort_values(
            "years_after_sale",
            ascending=False
        )
        .head(20)
    )

    print(
        largest_differences[columns]
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 8. Check whether effyearblt is before yearbuilt
    # ---------------------------------------------------------
    invalid_order = df[
        (df["yearbuilt"].notna())
        & (df["effyearblt"].notna())
        & (df["effyearblt"] < df["yearbuilt"])
    ]

    print("\nEFFYEARBLT BEFORE YEARBUILT")

    print(
        f"  Records where effyearblt < yearbuilt: "
        f"{len(invalid_order):,}"
    )

    # ---------------------------------------------------------
    # 9. Check year ranges
    # ---------------------------------------------------------
    print("\nYEAR RANGES")

    print(
        f"  yearbuilt  : "
        f"{df['yearbuilt'].min()} - {df['yearbuilt'].max()}"
    )

    print(
        f"  effyearblt : "
        f"{df['effyearblt'].min()} - {df['effyearblt'].max()}"
    )

    print(
        f"  sale_year  : "
        f"{df['sale_year'].min()} - {df['sale_year'].max()}"
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()