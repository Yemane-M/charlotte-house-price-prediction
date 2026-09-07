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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cama_parcel_features.csv"
)


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


def clean_text(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )


def main():

    print("\n" + "=" * 80)
    print("STEP 22 - AGGREGATE CAMA TO ONE ROW PER PARCEL")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load CAMA
    # ------------------------------------------------------------------

    print("\nLoading CAMA data...")

    cama = pd.read_csv(
        CAMA_FILE,
        usecols=CAMA_COLUMNS,
        low_memory=False,
    )

    print(f"CAMA records loaded: {len(cama):,}")

    # ------------------------------------------------------------------
    # Clean parcel ID
    # ------------------------------------------------------------------

    cama["parcelid"] = (
        cama["parcelid"]
        .astype(str)
        .str.strip()
    )

    print(
        f"Unique CAMA parcels: "
        f"{cama['parcelid'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # Clean text columns
    # ------------------------------------------------------------------

    text_columns = [
        "gispid",
        "lusecode",
        "landuse_description",
        "neighborhood",
        "bldgtype",
         "grade",
        "condo_town_flag",
        "parcel_type",
    ]

    for column in text_columns:
        cama[column] = clean_text(cama[column])

    # ------------------------------------------------------------------
    # Numeric columns
    # ------------------------------------------------------------------

    numeric_columns = [
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
        "finisharea",
        "totalarea",
        "bedrooms",
        "basearea",
        "fingarage",
        "unfingarag",
        "totgararea",
        "finattic",
        "totcrpoarea",
    ]

    for column in numeric_columns:
        cama[column] = pd.to_numeric(
            cama[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------
    # Diagnostics before aggregation
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CAMA RECORD STRUCTURE")
    print("=" * 80)

    parcel_counts = cama["parcelid"].value_counts()

    print(
        f"\nParcels with exactly one CAMA record: "
        f"{(parcel_counts == 1).sum():,}"
    )

    print(
        f"Parcels with multiple CAMA records: "
        f"{(parcel_counts > 1).sum():,}"
    )

    print(
        f"Maximum CAMA records for one parcel: "
        f"{parcel_counts.max():,}"
    )

    # ------------------------------------------------------------------
    # Aggregate numeric characteristics
    # ------------------------------------------------------------------

    print("\nAggregating parcel characteristics...")

    aggregated = (
        cama.groupby("parcelid", as_index=False)
        .agg(
            # Identity / categorical characteristics
            gispid=("gispid", "first"),
            neighborhood=("neighborhood", "first"),
            condo_town_flag=("condo_town_flag", "first"),
            parcel_type=("parcel_type", "first"),

            # Land
            gisacres=("gisacres", "max"),

            # Assessment snapshot year
            yearid=("yearid", "max"),

            # Building characteristics
            heatedarea=("heatedarea", "sum"),
            finisharea=("finisharea", "sum"),
            totalarea=("totalarea", "sum"),
            basearea=("basearea", "sum"),

            # Bedrooms / bathrooms
            bedrooms=("bedrooms", "sum"),
            fullbath=("fullbath", "sum"),
            halfbath=("halfbath", "sum"),
            threequabath=("threequabath", "sum"),

            # Other characteristics
            fireplaces=("fireplaces", "sum"),
            basegarage=("basegarage", "sum"),
            fingarage=("fingarage", "sum"),
            unfingarag=("unfingarag", "sum"),
            totgararea=("totgararea", "sum"),
            finattic=("finattic", "sum"),
            totcrpoarea=("totcrpoarea", "sum"),

            # Year-built characteristics
            yearbuilt=("yearbuilt", "min"),
            effyearblt=("effyearblt", "max"),

            # Story height
            storyheight=("storyheight", "max"),

            
        )
    )

    # ------------------------------------------------------------------
    # Add building / record counts
    # ------------------------------------------------------------------

    record_counts = (
        cama.groupby("parcelid")
        .size()
        .reset_index(name="cama_record_count")
    )

    aggregated = aggregated.merge(
        record_counts,
        on="parcelid",
        how="left",
    )

    # ------------------------------------------------------------------
    # Identify whether parcel has multiple CAMA records
    # ------------------------------------------------------------------

    aggregated["has_multiple_cama_records"] = (
        aggregated["cama_record_count"] > 1
    )

    # ------------------------------------------------------------------
    # Dominant land-use description
    # ------------------------------------------------------------------

    def most_common_value(series):
        series = series.dropna()

        if len(series) == 0:
            return pd.NA

        return series.value_counts().index[0]

    dominant_landuse = (
        cama.groupby("parcelid")["landuse_description"]
        .agg(most_common_value)
        .reset_index(name="landuse_description")
    )

    aggregated = aggregated.merge(
        dominant_landuse,
        on="parcelid",
        how="left",
    )

    # ------------------------------------------------------------------
    # Dominant building type
    # ------------------------------------------------------------------

    dominant_bldgtype = (
        cama.groupby("parcelid")["bldgtype"]
        .agg(most_common_value)
        .reset_index(name="bldgtype")
    )

    aggregated = aggregated.merge(
        dominant_bldgtype,
        on="parcelid",
        how="left",
    )

    # ------------------------------------------------------------------
    # Dominant grade
    # ------------------------------------------------------------------

    dominant_grade = (
        cama.groupby("parcelid")["grade"]
        .agg(most_common_value)
        .reset_index(name="grade")
    )

    aggregated = aggregated.merge(
        dominant_grade,
        on="parcelid",
        how="left",
    )


    # ------------------------------------------------------------------
    # Number of distinct building types
    # ------------------------------------------------------------------

    distinct_bldgtypes = (
        cama.groupby("parcelid")["bldgtype"]
        .nunique(dropna=True)
        .reset_index(name="distinct_bldgtype_count")
    )

    aggregated = aggregated.merge(
        distinct_bldgtypes,
        on="parcelid",
        how="left",
    )

    # ------------------------------------------------------------------
    # Number of distinct land-use descriptions
    # ------------------------------------------------------------------

    distinct_landuses = (
        cama.groupby("parcelid")["landuse_description"]
        .nunique(dropna=True)
        .reset_index(name="distinct_landuse_count")
    )

    aggregated = aggregated.merge(
        distinct_landuses,
        on="parcelid",
        how="left",
    )

    # ------------------------------------------------------------------
    # Replace zero/invalid values where appropriate
    # ------------------------------------------------------------------

    aggregated["bedrooms"] = aggregated["bedrooms"].fillna(0)
    aggregated["fullbath"] = aggregated["fullbath"].fillna(0)
    aggregated["halfbath"] = aggregated["halfbath"].fillna(0)
    aggregated["threequabath"] = aggregated["threequabath"].fillna(0)

    # ------------------------------------------------------------------
    # Calculate total bathroom equivalents
    # ------------------------------------------------------------------

    aggregated["bathroom_equivalents"] = (
        aggregated["fullbath"]
        + 0.5 * aggregated["halfbath"]
        + 0.75 * aggregated["threequabath"]
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    aggregated.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Final diagnostics
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("AGGREGATION RESULTS")
    print("=" * 80)

    print(
        f"\nOriginal CAMA records: "
        f"{len(cama):,}"
    )

    print(
        f"Original unique parcels: "
        f"{cama['parcelid'].nunique():,}"
    )

    print(
        f"Aggregated parcel records: "
        f"{len(aggregated):,}"
    )

    print(
        f"Duplicate parcel records removed through aggregation: "
        f"{len(cama) - len(aggregated):,}"
    )

    print(
        f"\nParcels with multiple CAMA records: "
        f"{aggregated['has_multiple_cama_records'].sum():,}"
    )

    print("\nFinal columns:")

    for column in aggregated.columns:
        print(f"  - {column}")

    print("\nSample aggregated records:")

    print(
        aggregated.head(10).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("STEP 22 COMPLETE")
    print("=" * 80)

    print(
        f"\nOutput saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()