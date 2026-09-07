import pandas as pd


INPUT_FILE = "data/processed/house_price_modeling.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    outliers = df[df["fireplaces"] > 5].copy()

    columns = [
        "parcelid",
        "saleprice",
        "sale_year",
        "landusefulldescription",
        "gisacres",
        "heatedarea",
        "finisharea",
        "bedrooms",
        "bathroom_equivalents",
        "fireplaces",
        "fingarage",
        "finattic",
        "yearbuilt",
        "effyearblt",
        "neighborhood",
        "bldgtype",
        "cama_record_count",
    ]

    columns = [c for c in columns if c in outliers.columns]

    print("=" * 80)
    print("FIREPLACE OUTLIER ANALYSIS")
    print("=" * 80)

    print(f"\nNumber of records with fireplaces > 5: {len(outliers)}")

    print("\nOutlier records:")
    print(
        outliers[columns]
        .sort_values("fireplaces", ascending=False)
        .to_string(index=False)
    )

    print("\nFireplace distribution:")
    print(df["fireplaces"].value_counts().sort_index())

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()