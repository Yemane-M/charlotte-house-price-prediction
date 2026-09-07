from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mecklenburg"
    / "Cama_Table.csv"
)


# ------------------------------------------------------------------
# CAMA columns needed for duplicate investigation
# ------------------------------------------------------------------

CAMA_COLUMNS = [
    "parcelid",
    "gispid",
    "legal_from",
    "map_book",
    "map_page",
    "map_block",
    "lot_num",
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


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    print_section("STEP 19 - CAMA DUPLICATE STRUCTURE INVESTIGATION")

    # ------------------------------------------------------------------
    # Step 19.1: Load CAMA data
    # ------------------------------------------------------------------

    print("\nLoading CAMA data...")

    cama = pd.read_csv(
        CAMA_FILE,
        usecols=CAMA_COLUMNS,
        low_memory=False
    )

    cama["parcelid"] = (
        cama["parcelid"]
        .astype(str)
        .str.strip()
    )

    print(f"CAMA rows: {len(cama):,}")
    print(f"Unique parcels: {cama['parcelid'].nunique():,}")

    # ------------------------------------------------------------------
    # Step 19.2: Find parcels with multiple CAMA records
    # ------------------------------------------------------------------

    parcel_counts = (
        cama["parcelid"]
        .value_counts()
    )

    duplicate_parcels = parcel_counts[
        parcel_counts > 1
    ]

    print_section("CAMA DUPLICATE SUMMARY")

    print(
        f"Parcels with multiple CAMA records: "
        f"{len(duplicate_parcels):,}"
    )

    print(
        f"Total rows belonging to duplicate parcels: "
        f"{parcel_counts[parcel_counts > 1].sum():,}"
    )

    print("\nDistribution of CAMA records per parcel:")

    print(
        duplicate_parcels
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Step 19.3: Extract duplicate parcels
    # ------------------------------------------------------------------

    duplicate_ids = duplicate_parcels.index

    duplicates = cama[
        cama["parcelid"].isin(duplicate_ids)
    ].copy()

    # ------------------------------------------------------------------
    # Step 19.4: Determine whether duplicate rows are identical
    # ------------------------------------------------------------------

    print_section("IDENTICAL DUPLICATE ANALYSIS")

    duplicate_without_parcel = duplicates.drop(
        columns=["parcelid"]
    )

    identical_duplicate_rows = (
        duplicate_without_parcel
        .duplicated(keep=False)
    )

    print(
        f"Rows that are identical across all selected CAMA fields: "
        f"{identical_duplicate_rows.sum():,}"
    )

    # ------------------------------------------------------------------
    # Step 19.5: Find parcels with multiple distinct records
    # ------------------------------------------------------------------

    distinct_record_counts = (
        duplicates
        .groupby("parcelid")
        .apply(
            lambda group: len(
                group.drop_duplicates()
            ),
            include_groups=False
        )
    )

    multiple_distinct = distinct_record_counts[
        distinct_record_counts > 1
    ]

    only_identical = distinct_record_counts[
        distinct_record_counts == 1
    ]

    print(
        f"\nParcels whose CAMA rows are identical: "
        f"{len(only_identical):,}"
    )

    print(
        f"Parcels with multiple DISTINCT CAMA records: "
        f"{len(multiple_distinct):,}"
    )

    # ------------------------------------------------------------------
    # Step 19.6: Show examples of distinct CAMA records
    # ------------------------------------------------------------------

    print_section("EXAMPLES OF MULTIPLE DISTINCT CAMA RECORDS")

    example_parcels = (
        multiple_distinct
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    example_columns = [
        "parcelid",
        "gispid",
        "legal_from",
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
        "bedrooms",
        "storyheight",
        "bldgtype",
        "finisharea",
        "totalarea",
        "grade",
        "basearea",
        "fingarage",
        "unfingarag",
        "totgararea",
        "parcel_type",
    ]

    examples = (
        duplicates[
            duplicates["parcelid"].isin(example_parcels)
        ][example_columns]
        .sort_values(
            by=["parcelid", "heatedarea"],
            na_position="first"
        )
    )

    print(
        examples.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Step 19.7: Analyze CAMA records by building type
    # ------------------------------------------------------------------

    print_section("BUILDING TYPE ANALYSIS")

    building_summary = (
        duplicates["bldgtype"]
        .fillna("MISSING")
        .value_counts()
        .head(30)
    )

    print(
        building_summary.to_string()
    )

    # ------------------------------------------------------------------
    # Step 19.8: Analyze yearid
    # ------------------------------------------------------------------

    print_section("CAMA YEARID ANALYSIS")

    print(
        duplicates["yearid"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Step 19.9: Analyze number of buildings/records
    # ------------------------------------------------------------------

    print_section("RECORD COUNT PER PARCEL")

    record_count_summary = (
        duplicate_parcels
        .describe()
    )

    print(
        record_count_summary.to_string()
    )

    # ------------------------------------------------------------------
    # Step 19.10: Save detailed duplicate data
    # ------------------------------------------------------------------

    output_dir = PROJECT_ROOT / "data" / "interim"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "cama_duplicate_parcels.csv"
    )

    duplicates.sort_values(
        by="parcelid"
    ).to_csv(
        output_file,
        index=False
    )

    print_section("STEP 19 SUMMARY")

    print(
        f"\nTotal CAMA records: "
        f"{len(cama):,}"
    )

    print(
        f"Unique CAMA parcels: "
        f"{cama['parcelid'].nunique():,}"
    )

    print(
        f"Parcels with multiple CAMA records: "
        f"{len(duplicate_parcels):,}"
    )

    print(
        f"Parcels with identical duplicate records: "
        f"{len(only_identical):,}"
    )

    print(
        f"Parcels with multiple distinct records: "
        f"{len(multiple_distinct):,}"
    )

    print(
        f"\nDetailed duplicate data saved to:"
        f"\n{output_file}"
    )

    print(
        "\nIMPORTANT:"
        "\nDo NOT aggregate or remove CAMA records yet."
        "\nThe next step depends on these results."
    )


if __name__ == "__main__":
    main()