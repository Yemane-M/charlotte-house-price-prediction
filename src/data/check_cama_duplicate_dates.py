from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMA_FILE = PROJECT_ROOT / "data" / "raw" / "mecklenburg" / "Cama_Table.csv"


def main():
    print("\n" + "=" * 80)
    print("STEP 21 - CHECK CAMA DUPLICATE RECORD DATES")
    print("=" * 80)

    cama = pd.read_csv(
        CAMA_FILE,
        usecols=[
            "parcelid",
            "gispid",
            "legal_from",
            "yearid",
            "lusecode",
            "landuse_description",
            "bldgtype",
            "yearbuilt",
            "effyearblt",
            "heatedarea",
            "totalarea",
            "bedrooms",
            "fullbath",
            "halfbath",
            "gisacres",
        ],
        low_memory=False,
    )

    cama["parcelid"] = cama["parcelid"].astype(str).str.strip()
    cama["legal_from"] = pd.to_datetime(
        cama["legal_from"],
        errors="coerce"
    )

    # --------------------------------------------------------------
    # Find parcels with multiple CAMA records
    # --------------------------------------------------------------

    counts = cama["parcelid"].value_counts()

    duplicate_parcels = counts[counts > 1].index

    duplicates = cama[
        cama["parcelid"].isin(duplicate_parcels)
    ].copy()

    print(f"\nTotal CAMA records: {len(cama):,}")
    print(f"Unique parcels: {cama['parcelid'].nunique():,}")
    print(f"Parcels with multiple records: {len(duplicate_parcels):,}")

    # --------------------------------------------------------------
    # Look at parcels with multiple DISTINCT records
    # --------------------------------------------------------------

    distinct_columns = [
        "parcelid",
        "lusecode",
        "landuse_description",
        "bldgtype",
        "yearbuilt",
        "effyearblt",
        "heatedarea",
        "totalarea",
        "bedrooms",
        "fullbath",
        "halfbath",
        "gisacres",
    ]

    distinct = duplicates.drop_duplicates(
        subset=distinct_columns
    )

    distinct_counts = distinct["parcelid"].value_counts()

    multi_distinct_parcels = distinct_counts[
        distinct_counts > 1
    ].index

    multi_distinct = distinct[
        distinct["parcelid"].isin(multi_distinct_parcels)
    ].copy()

    print(
        f"Parcels with multiple DISTINCT records: "
        f"{len(multi_distinct_parcels):,}"
    )

    # --------------------------------------------------------------
    # Legal_from analysis
    # --------------------------------------------------------------

    legal_counts = (
        multi_distinct
        .groupby("parcelid")["legal_from"]
        .nunique()
    )

    print("\n" + "=" * 80)
    print("LEGAL_FROM PATTERNS")
    print("=" * 80)

    print(
        f"\nParcels with multiple distinct records: "
        f"{len(multi_distinct_parcels):,}"
    )

    print(
        f"Parcels where all records share the same legal_from: "
        f"{(legal_counts == 1).sum():,}"
    )

    print(
        f"Parcels where records have different legal_from dates: "
        f"{(legal_counts > 1).sum():,}"
    )

    # --------------------------------------------------------------
    # Show examples
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXAMPLES OF MULTIPLE DISTINCT CAMA RECORDS")
    print("=" * 80)

    example_parcels = (
        multi_distinct["parcelid"]
        .drop_duplicates()
        .head(20)
        .tolist()
    )

    examples = (
        multi_distinct[
            multi_distinct["parcelid"].isin(example_parcels)
        ]
        .sort_values(["parcelid", "legal_from"])
    )

    print(
        examples.to_string(index=False)
    )

    # --------------------------------------------------------------
    # Save diagnostic
    # --------------------------------------------------------------

    output_file = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "cama_duplicate_date_diagnostic.csv"
    )

    examples.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("STEP 21 COMPLETE")
    print("=" * 80)

    print(
        f"\nDiagnostic saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()