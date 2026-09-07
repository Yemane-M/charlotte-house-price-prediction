import pandas as pd


INPUT_FILE = "data/processed/house_price_modeling.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    invalid = df[
        df["yearbuilt"] > df["sale_year"]
    ].copy()

    invalid["years_after_sale"] = (
        invalid["yearbuilt"] - invalid["sale_year"]
    )

    print("=" * 80)
    print("INVALID YEARBUILT ANALYSIS")
    print("=" * 80)

    print(f"\nTotal records: {len(df):,}")
    print(
        f"yearbuilt > sale_year: "
        f"{len(invalid):,}"
    )

    print("\nYEARS AFTER SALE")

    print(
        invalid["years_after_sale"]
        .describe()
        .to_string()
    )

    print("\nDISTRIBUTION BY YEARS AFTER SALE")

    print(
        invalid["years_after_sale"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nINVALID RECORDS BY SALE YEAR")

    print(
        invalid["sale_year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nINVALID RECORDS BY SALE YEAR (%)")

    total_by_year = (
        df.groupby("sale_year")
        .size()
    )

    invalid_by_year = (
        invalid.groupby("sale_year")
        .size()
    )

    percentage = (
        invalid_by_year
        .div(total_by_year)
        .mul(100)
        .fillna(0)
    )

    for year, value in percentage.items():
        print(f"  {int(year)}: {value:.2f}%")

    print("\nLARGEST YEARBUILT - SALE YEAR DIFFERENCES")

    columns = [
        "parcelid",
        "saleprice",
        "sale_year",
        "yearbuilt",
        "effyearblt",
        "gisacres",
        "heatedarea",
        "finisharea",
        "bedrooms",
        "bathroom_equivalents",
        "fireplaces",
        "fingarage",
        "finattic",
        "landusefulldescription",
        "neighborhood",
        "bldgtype",
        "cama_record_count",
    ]

    columns = [
        column
        for column in columns
        if column in invalid.columns
    ]

    largest = (
        invalid
        .sort_values(
            "years_after_sale",
            ascending=False
        )
        .head(30)
    )

    print(
        largest[columns]
        .to_string(index=False)
    )

    print("\nCAMA RECORD COUNT FOR INVALID YEARS")

    print(
        invalid["cama_record_count"]
        .value_counts()
        .sort_index()
        .head(30)
        .to_string()
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()