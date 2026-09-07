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

SALES_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "model_sales.csv"
)


def main():

    print("\n" + "=" * 80)
    print("STEP 20 - CHECK CAMA LEGAL_FROM DATES")
    print("=" * 80)

    # --------------------------------------------------------------
    # Load sales
    # --------------------------------------------------------------

    print("\nLoading cleaned sales...")

    sales = pd.read_csv(
        SALES_FILE,
        low_memory=False
    )

    sales["parcelid"] = (
        sales["parcelid"]
        .astype(str)
        .str.strip()
    )

    sales["saledate"] = pd.to_datetime(
        sales["saledate"],
        errors="coerce"
    )

    print(f"Sales records: {len(sales):,}")

    # --------------------------------------------------------------
    # Select a sample of parcels from the sales data
    # --------------------------------------------------------------

    sample_parcels = (
        sales["parcelid"]
        .drop_duplicates()
        .head(100)
        .tolist()
    )

    # --------------------------------------------------------------
    # Load CAMA
    # --------------------------------------------------------------

    print("\nLoading CAMA legal_from data...")

    cama = pd.read_csv(
        CAMA_FILE,
        usecols=[
            "parcelid",
            "gispid",
            "legal_from",
            "yearid",
            "yearbuilt",
            "effyearblt",
            "heatedarea",
            "bldgtype",
            "landuse_description",
        ],
        low_memory=False
    )

    cama["parcelid"] = (
        cama["parcelid"]
        .astype(str)
        .str.strip()
    )

    cama["legal_from"] = pd.to_datetime(
        cama["legal_from"],
        errors="coerce"
    )

    # --------------------------------------------------------------
    # Show legal_from statistics
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LEGAL_FROM SUMMARY")
    print("=" * 80)

    print(
        f"\nCAMA records: {len(cama):,}"
    )

    print(
        f"Missing legal_from: "
        f"{cama['legal_from'].isna().sum():,}"
    )

    print(
        f"Non-missing legal_from: "
        f"{cama['legal_from'].notna().sum():,}"
    )

    print("\nEarliest legal_from:")

    print(
        cama["legal_from"]
        .min()
    )

    print("\nLatest legal_from:")

    print(
        cama["legal_from"]
        .max()
    )

    print("\nLegal_from year distribution:")

    print(
        cama["legal_from"]
        .dt.year
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------------
    # Compare sale date with legal_from
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAMPLE SALES VS CAMA LEGAL_FROM")
    print("=" * 80)

    sample_sales = (
        sales[
            sales["parcelid"].isin(sample_parcels)
        ][
            [
                "parcelid",
                "saleprice",
                "saledate",
            ]
        ]
        .drop_duplicates()
        .head(30)
    )

    sample_joined = sample_sales.merge(
        cama,
        on="parcelid",
        how="left"
    )

    print(
        sample_joined.to_string(index=False)
    )

    # --------------------------------------------------------------
    # Find CAMA records where legal_from is after sale date
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEMPORAL CHECK")
    print("=" * 80)

    temporal = sales[
        [
            "parcelid",
            "saleprice",
            "saledate",
        ]
    ].merge(
        cama[
            [
                "parcelid",
                "legal_from",
                "yearid",
                "yearbuilt",
                "effyearblt",
                "heatedarea",
                "bldgtype",
            ]
        ],
        on="parcelid",
        how="left"
    )

    temporal["legal_after_sale"] = (
        temporal["legal_from"]
        > temporal["saledate"]
    )

    print(
        f"\nJoined rows: "
        f"{len(temporal):,}"
    )

    print(
        f"CAMA legal_from AFTER sale date: "
        f"{temporal['legal_after_sale'].sum():,}"
    )

    print(
        f"CAMA legal_from ON/BEFORE sale date: "
        f"{(~temporal['legal_after_sale']).sum():,}"
    )

    # --------------------------------------------------------------
    # Show examples where legal_from is after sale
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXAMPLES WHERE LEGAL_FROM IS AFTER SALE DATE")
    print("=" * 80)

    after_sale = (
        temporal[
            temporal["legal_after_sale"]
        ]
        .sort_values(
            by="saledate"
        )
        .head(30)
    )

    if len(after_sale) == 0:

        print(
            "\nNo examples found."
        )

    else:

        print(
            after_sale.to_string(index=False)
        )

    # --------------------------------------------------------------
    # Save diagnostic
    # --------------------------------------------------------------

    output_file = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "cama_date_diagnostic.csv"
    )

    temporal.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("STEP 20 COMPLETE")
    print("=" * 80)

    print(
        f"\nDiagnostic saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()