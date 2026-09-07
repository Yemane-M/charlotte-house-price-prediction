from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SALES_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "model_sales.csv"
)

CAMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mecklenburg"
    / "Cama_Table.csv"
)


# ------------------------------------------------------------------
# CAMA columns needed for property validation
# ------------------------------------------------------------------

CAMA_COLUMNS = [
    "parcelid",
    "gispid",
    "lusecode",
    "landuse_description",
    "neighborhood",
    "gisacres",
    "yearid",
    "heatedarea",
    "yearbuilt",
    "effyearblt",
    "fullbath",
    "halfbath",
    "threequabath",
    "fireplaces",
    "basegarage",
    "storyheight",
    "bldgtype",
    "finisharea",
    "totalarea",
    "bedrooms",
    "grade",
    "basearea",
    "fingarage",
    "unfingarag",
    "totgararea",
    "finattic",
    "totcrpoarea",
    "condo_town_flag",
    "parcel_type",
]


# ------------------------------------------------------------------
# Helper function
# ------------------------------------------------------------------

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    print_section("STEP 18 - SALES TO CAMA PROPERTY VALIDATION")

    # ------------------------------------------------------------------
    # Step 18.1: Check that the cleaned sales file exists
    # ------------------------------------------------------------------

    if not SALES_FILE.exists():
        raise FileNotFoundError(
            f"\nCleaned sales file was not found:\n{SALES_FILE}\n\n"
            "Run filter_sales.py first and make sure it creates "
            "data/interim/model_sales.csv."
        )

    if not CAMA_FILE.exists():
        raise FileNotFoundError(
            f"\nCAMA file was not found:\n{CAMA_FILE}"
        )

    # ------------------------------------------------------------------
    # Step 18.2: Load cleaned sales
    # ------------------------------------------------------------------

    print("\nLoading cleaned sales data...")

    sales = pd.read_csv(
        SALES_FILE,
        low_memory=False
    )

    print(f"Sales records loaded: {len(sales):,}")
    print(f"Unique sales parcels: {sales['parcelid'].nunique():,}")

    # Make sure parcelid has a consistent type
    sales["parcelid"] = (
        sales["parcelid"]
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------------------
    # Step 18.3: Identify low-price sales
    # ------------------------------------------------------------------

    low_price_sales = sales[sales["saleprice"] <= 100_000].copy()

    print_section("LOW-PRICE SALES")

    print(
        f"Transactions <= $100,000: "
        f"{len(low_price_sales):,}"
    )

    print(
        f"Unique parcels <= $100,000: "
        f"{low_price_sales['parcelid'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # Step 18.4: Load selected CAMA columns
    # ------------------------------------------------------------------

    print_section("LOADING CAMA PROPERTY DATA")

    print(f"CAMA file: {CAMA_FILE}")
    print("\nLoading selected CAMA columns...")
    print("This may take a little while because the source file is large.")

    cama = pd.read_csv(
        CAMA_FILE,
        usecols=CAMA_COLUMNS,
        low_memory=False
    )

    print(f"\nCAMA records loaded: {len(cama):,}")
    print(f"CAMA unique parcels: {cama['parcelid'].nunique():,}")

    # Make parcelid consistent
    cama["parcelid"] = (
        cama["parcelid"]
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------------------
    # Step 18.5: Examine CAMA duplicate parcels
    # ------------------------------------------------------------------

    print_section("CAMA DUPLICATE PARCEL INVESTIGATION")

    cama_duplicate_mask = cama.duplicated(
        subset=["parcelid"],
        keep=False
    )

    cama_duplicates = cama[cama_duplicate_mask].copy()

    print(
        f"CAMA rows belonging to duplicate parcels: "
        f"{len(cama_duplicates):,}"
    )

    print(
        f"Unique parcels with multiple CAMA records: "
        f"{cama_duplicates['parcelid'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # Step 18.6: Join cleaned sales to CAMA
    # ------------------------------------------------------------------

    print_section("SALES TO CAMA JOIN")

    # We intentionally use a LEFT JOIN.
    #
    # Every cleaned sale should remain in the dataset even if
    # corresponding CAMA characteristics are missing.

    sales_cama = sales.merge(
        cama,
        on="parcelid",
        how="left",
        indicator=True,
        suffixes=("", "_cama")
    )

    print(f"Sales records before join: {len(sales):,}")
    print(f"Records after join:        {len(sales_cama):,}")

    # ------------------------------------------------------------------
    # Important duplicate warning
    # ------------------------------------------------------------------

    if len(sales_cama) > len(sales):

        print(
            "\nWARNING:"
            "\nThe CAMA join increased the number of rows."
            "\nThis means some parcels have multiple CAMA records."
            "\nWe will NOT remove those records yet."
            "\nWe need to investigate the structure first."
        )

    # ------------------------------------------------------------------
    # Step 18.7: Match statistics
    # ------------------------------------------------------------------

    print_section("CAMA MATCH STATISTICS")

    match_counts = (
        sales_cama["_merge"]
        .value_counts()
    )

    print(match_counts)

    matched = (sales_cama["_merge"] == "both").sum()
    unmatched = (sales_cama["_merge"] == "left_only").sum()

    print(f"\nMatched sales rows:   {matched:,}")
    print(f"Unmatched sales rows: {unmatched:,}")

    matched_parcels = sales_cama.loc[
        sales_cama["_merge"] == "both",
        "parcelid"
    ].nunique()

    unmatched_parcels = sales_cama.loc[
        sales_cama["_merge"] == "left_only",
        "parcelid"
    ].nunique()

    print(f"\nMatched unique parcels:   {matched_parcels:,}")
    print(f"Unmatched unique parcels: {unmatched_parcels:,}")

    # ------------------------------------------------------------------
    # Step 18.8: Investigate low-price properties
    # ------------------------------------------------------------------

    print_section("LOW-PRICE SALES WITH CAMA CHARACTERISTICS")

    low_price_joined = sales_cama[
        sales_cama["saleprice"] <= 100_000
    ].copy()

    # Keep one row per joined CAMA record for inspection.
    low_price_columns = [
        "parcelid",
        "saleprice",
        "saledate",
        "landusefulldescription",
        "deeddescription",
        "salesvalidity",
        "lusecode",
        "landuse_description",
        "neighborhood",
        "gisacres",
        "yearid",
        "heatedarea",
        "yearbuilt",
        "effyearblt",
        "fullbath",
        "halfbath",
        "threequabath",
        "storyheight",
        "bldgtype",
        "finisharea",
        "totalarea",
        "bedrooms",
        "grade",
        "basearea",
        "fingarage",
        "unfingarag",
        "totgararea",
        "condo_town_flag",
        "parcel_type",
    ]

    low_price_columns = [
        col for col in low_price_columns
        if col in low_price_joined.columns
    ]

    low_price_view = (
        low_price_joined[
            low_price_columns
        ]
        .sort_values(
            by=["saleprice", "parcelid"]
        )
    )

    print(
        f"\nLow-price joined rows: "
        f"{len(low_price_view):,}"
    )

    print("\nLowest 30 transactions with CAMA characteristics:\n")

    print(
        low_price_view
        .head(30)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Step 18.9: Check missing property characteristics
    # ------------------------------------------------------------------

    print_section("CAMA PROPERTY CHARACTERISTIC COMPLETENESS")

    characteristics = [
        "heatedarea",
        "yearbuilt",
        "effyearblt",
        "fullbath",
        "halfbath",
        "bedrooms",
        "gisacres",
        "totalarea",
        "finisharea",
        "grade",
        "bldgtype",
    ]

    for column in characteristics:

        if column not in sales_cama.columns:
            continue

        missing = sales_cama[column].isna().sum()

        percentage = (
            missing / len(sales_cama) * 100
            if len(sales_cama) > 0
            else 0
        )

        print(
            f"{column:20s} "
            f"missing: {missing:10,} "
            f"({percentage:6.2f}%)"
        )

    # ------------------------------------------------------------------
    # Step 18.10: Save diagnostic output
    # ------------------------------------------------------------------

    output_dir = PROJECT_ROOT / "data" / "interim"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "low_price_sales_cama_diagnostic.csv"
    )

    low_price_view.to_csv(
        output_file,
        index=False
    )

    print_section("DIAGNOSTIC FILE CREATED")

    print(f"\nSaved to:")
    print(output_file)

    # ------------------------------------------------------------------
    # Step 18.11: Summary
    # ------------------------------------------------------------------

    print_section("STEP 18 SUMMARY")

    print(
        f"\nCleaned sales records: "
        f"{len(sales):,}"
    )

    print(
        f"CAMA records loaded: "
        f"{len(cama):,}"
    )

    print(
        f"CAMA unique parcels: "
        f"{cama['parcelid'].nunique():,}"
    )

    print(
        f"Joined records: "
        f"{len(sales_cama):,}"
    )

    print(
        f"Matched unique parcels: "
        f"{matched_parcels:,}"
    )

    print(
        f"Unmatched unique parcels: "
        f"{unmatched_parcels:,}"
    )

    print(
        "\nIMPORTANT:"
        "\nWe have NOT removed low-price sales."
        "\nWe have NOT removed duplicate CAMA parcels."
        "\nWe have NOT created the final modeling dataset yet."
        "\n"
        "\nNext step will be to examine the CAMA duplicate structure"
        "\nand determine the correct property-level aggregation."
    )


if __name__ == "__main__":
    main()