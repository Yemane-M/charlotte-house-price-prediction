from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mecklenburg"
    / "Cama_Table.csv"
)

AGGREGATED_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cama_parcel_features.csv"
)


def main():

    print("\n" + "=" * 80)
    print("STEP 24 - INVESTIGATE CAMA GRADE")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Raw CAMA
    # ------------------------------------------------------------------

    print("\nLoading raw CAMA grade...")

    cama = pd.read_csv(
        CAMA_FILE,
        usecols=[
            "parcelid",
            "grade",
            "landuse_description",
            "bldgtype",
        ],
        low_memory=False,
    )

    print(f"Raw CAMA records: {len(cama):,}")

    print("\nRaw grade data type:")
    print(cama["grade"].dtype)

    print("\nRaw grade missing:")
    print(
        f"{cama['grade'].isna().sum():,}"
    )

    print("\nRaw grade non-missing:")
    print(
        f"{cama['grade'].notna().sum():,}"
    )

    print("\nRaw grade unique values:")

    print(
        cama["grade"]
        .value_counts(dropna=False)
        .head(30)
        .to_string()
    )

    # ------------------------------------------------------------------
    # Aggregated CAMA
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("AGGREGATED CAMA GRADE")
    print("=" * 80)

    aggregated = pd.read_csv(
        AGGREGATED_FILE,
        low_memory=False,
    )

    print(
        f"\nAggregated records: "
        f"{len(aggregated):,}"
    )

    print("\nAggregated grade data type:")
    print(aggregated["grade"].dtype)

    print("\nAggregated grade missing:")
    print(
        f"{aggregated['grade'].isna().sum():,}"
    )

    print("\nAggregated grade non-missing:")
    print(
        f"{aggregated['grade'].notna().sum():,}"
    )

    print("\nAggregated grade unique values:")

    print(
        aggregated["grade"]
        .value_counts(dropna=False)
        .head(30)
        .to_string()
    )

    # ------------------------------------------------------------------
    # Grade by residential property type
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("GRADE BY LAND USE")
    print("=" * 80)

    grade_by_landuse = (
        cama.groupby("landuse_description")["grade"]
        .agg(
            records="size",
            non_missing="count",
        )
        .sort_values(
            "records",
            ascending=False,
        )
        .head(30)
    )

    grade_by_landuse["missing_pct"] = (
        100
        * (
            1
            - grade_by_landuse["non_missing"]
            / grade_by_landuse["records"]
        )
    )

    print(
        grade_by_landuse.to_string()
    )

    # ------------------------------------------------------------------
    # Grade by building type
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("GRADE BY BUILDING TYPE")
    print("=" * 80)

    grade_by_bldgtype = (
        cama.groupby("bldgtype")["grade"]
        .agg(
            records="size",
            non_missing="count",
        )
        .sort_values(
            "records",
            ascending=False,
        )
        .head(30)
    )

    grade_by_bldgtype["missing_pct"] = (
        100
        * (
            1
            - grade_by_bldgtype["non_missing"]
            / grade_by_bldgtype["records"]
        )
    )

    print(
        grade_by_bldgtype.to_string()
    )

    print("\n" + "=" * 80)
    print("STEP 24 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()