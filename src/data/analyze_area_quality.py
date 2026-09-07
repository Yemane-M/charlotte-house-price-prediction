from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/house_price_modeling.csv")


def main():
    print("=" * 110)
    print("STEP 21G - PROPERTY AREA QUALITY ANALYSIS")
    print("=" * 110)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nTotal rows: {len(df):,}")

    # ---------------------------------------------------------------
    # 1. Basic area statistics
    # ---------------------------------------------------------------

    area_cols = [
        "heatedarea",
        "finisharea",
        "gisacres",
    ]

    print("\n" + "=" * 110)
    print("AREA SUMMARY")
    print("=" * 110)

    print(
        df[area_cols].describe(
            percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
        ).T.to_string()
    )

    # ---------------------------------------------------------------
    # 2. Extreme area counts
    # ---------------------------------------------------------------

    print("\n" + "=" * 110)
    print("EXTREME AREA COUNTS")
    print("=" * 110)

    thresholds = [5000, 7500, 10000, 15000, 25000, 50000]

    for threshold in thresholds:
        heated_count = (df["heatedarea"] > threshold).sum()
        finish_count = (df["finisharea"] > threshold).sum()

        print(
            f"> {threshold:>6,} sq ft | "
            f"heatedarea: {heated_count:>6,} | "
            f"finisharea: {finish_count:>6,}"
        )

    # ---------------------------------------------------------------
    # 3. Compare heatedarea and finisharea
    # ---------------------------------------------------------------

    print("\n" + "=" * 110)
    print("HEATEDAREA / FINISHAREA RELATIONSHIP")
    print("=" * 110)

    valid_area = (
        df["heatedarea"].notna()
        & df["finisharea"].notna()
        & (df["heatedarea"] > 0)
        & (df["finisharea"] > 0)
    )

    ratio = (
        df.loc[valid_area, "heatedarea"]
        / df.loc[valid_area, "finisharea"]
    )

    print(f"\nRows with both areas > 0: {valid_area.sum():,}")

    print("\nHeatedarea / Finisharea ratio:")
    print(
        ratio.describe(
            percentiles=[0.01, 0.05, 0.50, 0.95, 0.99, 0.999]
        ).to_string()
    )

    print(f"\nRatio < 0.80 : {(ratio < 0.80).sum():,}")
    print(f"Ratio > 1.20 : {(ratio > 1.20).sum():,}")
    print(f"Ratio > 2.00 : {(ratio > 2.00).sum():,}")

    # ---------------------------------------------------------------
    # 4. Extreme heatedarea records
    # ---------------------------------------------------------------

    display_cols = [
        "parcelid",
        "saleprice",
        "sale_year",
        "gisacres",
        "heatedarea",
        "finisharea",
        "bedrooms",
        "bathroom_equivalents",
        "fingarage",
        "yearbuilt",
        "effyearblt",
        "cama_record_count",
        "landusefulldescription",
    ]

    print("\n" + "=" * 110)
    print("HEATEDAREA > 10,000")
    print("=" * 110)

    extreme_heated = (
        df[df["heatedarea"] > 10000]
        .sort_values("heatedarea", ascending=False)
    )

    print(
        extreme_heated[display_cols]
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # 5. Extreme finisharea records
    # ---------------------------------------------------------------

    print("\n" + "=" * 110)
    print("FINISHAREA > 10,000")
    print("=" * 110)

    extreme_finish = (
        df[df["finisharea"] > 10000]
        .sort_values("finisharea", ascending=False)
    )

    print(
        extreme_finish[display_cols]
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # 6. Relationship with CAMA record count
    # ---------------------------------------------------------------

    print("\n" + "=" * 110)
    print("AREA EXTREMES BY CAMA RECORD COUNT")
    print("=" * 110)

    area_extreme = df[df["heatedarea"] > 10000].copy()

    if len(area_extreme) > 0:
        summary = (
            area_extreme
            .groupby("cama_record_count")
            .agg(
                rows=("parcelid", "size"),
                median_heatedarea=("heatedarea", "median"),
                max_heatedarea=("heatedarea", "max"),
                median_finisharea=("finisharea", "median"),
                max_finisharea=("finisharea", "max"),
            )
            .reset_index()
            .sort_values("cama_record_count")
        )

        print(summary.to_string(index=False))

    # ---------------------------------------------------------------
    # 7. Normal-size properties
    # ---------------------------------------------------------------

    print("\n" + "=" * 110)
    print("NORMAL PROPERTY SUBSET")
    print("=" * 110)

    normal = df[
        (df["heatedarea"] > 0)
        & (df["heatedarea"] <= 10000)
        & (df["finisharea"] > 0)
        & (df["finisharea"] <= 10000)
    ].copy()

    print(f"\nNormal rows: {len(normal):,}")
    print(f"Excluded rows: {len(df) - len(normal):,}")

    print("\nNormal-property correlations with saleprice:")

    correlation_cols = [
        "saleprice",
        "heatedarea",
        "finisharea",
        "gisacres",
        "bedrooms",
        "bathroom_equivalents",
        "fingarage",
    ]

    print(
        normal[correlation_cols]
        .corr()["saleprice"]
        .sort_values(ascending=False)
        .to_string()
    )

    print("\n" + "=" * 110)
    print("STEP 21G DIAGNOSTIC COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()