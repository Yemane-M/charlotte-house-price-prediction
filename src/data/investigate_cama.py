from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/mecklenburg")
CAMA_FILE = DATA_DIR / "Cama_Table.csv"


def main() -> None:
    print("=" * 80)
    print("CAMA DUPLICATE PARCEL INVESTIGATION")
    print("=" * 80)

    columns = [
        "parcelid",
        "gispid",
        "nc_pin",
        "landuse_description",
        "heatedarea",
        "bedrooms",
        "fullbath",
        "halfbath",
        "yearbuilt",
        "saleprice",
        "saledate",
        "validsale",
        "bldgtype",
        "resunits",
        "comunits",
        "finisharea",
        "totalarea",
    ]

    print("\nReading CAMA data...")

    cama = pd.read_csv(
        CAMA_FILE,
        usecols=columns,
        low_memory=False,
    )

    print(f"Total CAMA rows: {len(cama):,}")
    print(f"Unique parcels: {cama['parcelid'].nunique():,}")

    # ------------------------------------------------------------------
    # Find duplicate parcels
    # ------------------------------------------------------------------

    duplicate_mask = cama["parcelid"].duplicated(
        keep=False
    )

    duplicates = cama[duplicate_mask].copy()

    print("\n" + "-" * 80)
    print("DUPLICATE PARCELS")
    print("-" * 80)

    print(f"Rows belonging to duplicate parcels: {len(duplicates):,}")
    print(
        f"Unique parcels with multiple rows: "
        f"{duplicates['parcelid'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # Number of CAMA rows per parcel
    # ------------------------------------------------------------------

    rows_per_parcel = cama.groupby("parcelid").size()

    print("\n" + "-" * 80)
    print("ROWS PER PARCEL")
    print("-" * 80)

    print(
        rows_per_parcel
        .value_counts()
        .sort_index()
        .head(20)
    )

    print(
        f"\nMaximum CAMA rows for one parcel: "
        f"{rows_per_parcel.max()}"
    )

    # ------------------------------------------------------------------
    # Show examples
    # ------------------------------------------------------------------

    example_parcels = (
        rows_per_parcel[rows_per_parcel > 1]
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    print("\n" + "-" * 80)
    print("EXAMPLE DUPLICATE PARCELS")
    print("-" * 80)

    example_columns = [
        "parcelid",
        "gispid",
        "nc_pin",
        "landuse_description",
        "heatedarea",
        "bedrooms",
        "fullbath",
        "yearbuilt",
        "saleprice",
        "saledate",
        "validsale",
        "bldgtype",
        "resunits",
        "comunits",
        "finisharea",
        "totalarea",
    ]

    print(
        duplicates[
            duplicates["parcelid"].isin(example_parcels)
        ][example_columns]
        .sort_values(["parcelid", "heatedarea"])
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Property type distribution among duplicates
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("PROPERTY TYPES AMONG DUPLICATE PARCELS")
    print("-" * 80)

    print(
        duplicates["landuse_description"]
        .value_counts(dropna=False)
        .head(30)
    )

    # ------------------------------------------------------------------
    # Building type distribution
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("BUILDING TYPES AMONG DUPLICATE PARCELS")
    print("-" * 80)

    print(
        duplicates["bldgtype"]
        .value_counts(dropna=False)
        .head(30)
    )


if __name__ == "__main__":
    main()