from pathlib import Path

import pandas as pd


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
    / "interim"
    / "cama_parcel_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "house_price_dataset.csv"
)


def main():

    print("\n" + "=" * 80)
    print("STEP 23 - BUILD FINAL ML DATASET")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load cleaned sales
    # ------------------------------------------------------------------

    print("\nLoading cleaned sales...")

    sales = pd.read_csv(
        SALES_FILE,
        low_memory=False,
    )

    sales["parcelid"] = (
        sales["parcelid"]
        .astype(str)
        .str.strip()
    )

    sales["saledate"] = pd.to_datetime(
        sales["saledate"],
        errors="coerce",
    )

    print(f"Sales records: {len(sales):,}")
    print(
        f"Unique sales parcels: "
        f"{sales['parcelid'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # Load aggregated CAMA
    # ------------------------------------------------------------------

    print("\nLoading parcel-level CAMA features...")

    cama = pd.read_csv(
        CAMA_FILE,
        low_memory=False,
    )

    cama["parcelid"] = (
        cama["parcelid"]
        .astype(str)
        .str.strip()
    )

    print(f"CAMA feature rows: {len(cama):,}")
    print(
        f"Unique CAMA parcels: "
        f"{cama['parcelid'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # Verify CAMA is truly one row per parcel
    # ------------------------------------------------------------------

    cama_duplicate_count = (
        cama["parcelid"].duplicated().sum()
    )

    print(
        f"\nDuplicate parcel IDs in CAMA features: "
        f"{cama_duplicate_count:,}"
    )

    if cama_duplicate_count > 0:
        raise ValueError(
            "CAMA feature dataset is not one row per parcel."
        )

    # ------------------------------------------------------------------
    # Check sales parcels against CAMA
    # ------------------------------------------------------------------

    sales_parcels = set(
        sales["parcelid"].dropna()
    )

    cama_parcels = set(
        cama["parcelid"].dropna()
    )

    unmatched_sales_parcels = (
        sales_parcels - cama_parcels
    )

    print(
        f"Sales parcels not found in CAMA: "
        f"{len(unmatched_sales_parcels):,}"
    )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    print("\nJoining sales with parcel features...")

    ml_data = sales.merge(
        cama,
        on="parcelid",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # ------------------------------------------------------------------
    # Join diagnostics
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("JOIN RESULTS")
    print("=" * 80)

    print(
        f"\nSales records before join: "
        f"{len(sales):,}"
    )

    print(
        f"ML records after join: "
        f"{len(ml_data):,}"
    )

    print("\nMerge status:")

    print(
        ml_data["_merge"]
        .value_counts()
        .to_string()
    )

    matched = (
        ml_data["_merge"] == "both"
    ).sum()

    unmatched = (
        ml_data["_merge"] == "left_only"
    ).sum()

    print(
        f"\nMatched sales: {matched:,}"
    )

    print(
        f"Unmatched sales: {unmatched:,}"
    )

    # ------------------------------------------------------------------
    # Ensure no rows were unexpectedly lost or multiplied
    # ------------------------------------------------------------------

    if len(ml_data) != len(sales):
        raise ValueError(
            "Join changed the number of sales records. "
            "Investigate before continuing."
        )

    if unmatched > 0:
        raise ValueError(
            "Some sales transactions do not have CAMA features."
        )

    # ------------------------------------------------------------------
    # Remove merge indicator
    # ------------------------------------------------------------------

    ml_data = ml_data.drop(
        columns=["_merge"]
    )

    # ------------------------------------------------------------------
    # Basic ML dataset diagnostics
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ML DATASET SUMMARY")
    print("=" * 80)

    print(
        f"\nRows: {len(ml_data):,}"
    )

    print(
        f"Columns: {len(ml_data.columns):,}"
    )

    print(
        f"Unique parcels: "
        f"{ml_data['parcelid'].nunique():,}"
    )

    print("\nSale price summary:")

    print(
        ml_data["saleprice"]
        .describe()
        .to_string()
    )

    print("\nMissing values in important features:")

    important_columns = [
        "saleprice",
        "saledate",
        "heatedarea",
        "finisharea",
        "totalarea",
        "bedrooms",
        "fullbath",
        "halfbath" if "halfbath" in ml_data.columns else "halfbath",
        "gisacres",
        "yearbuilt",
        "effyearblt",
        "grade",
        "bldgtype",
        "landuse_description",
        "neighborhood",
    ]

    important_columns = [
        column
        for column in important_columns
        if column in ml_data.columns
    ]

    missing = (
        ml_data[important_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing.to_string())

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    ml_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 80)
    print("STEP 23 COMPLETE")
    print("=" * 80)

    print(
        f"\nFinal ML dataset saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()