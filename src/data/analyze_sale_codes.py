from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/mecklenburg")

SALES_FILE = DATA_DIR / "Parcel_Sales_Table.csv"
CAMA_FILE = DATA_DIR / "Cama_Table.csv"


def main() -> None:
    print("=" * 80)
    print("MECKLENBURG COUNTY SALE CODE ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # SALES DATA
    # ------------------------------------------------------------------

    print("\nLoading Sales data...")

    sales = pd.read_csv(
        SALES_FILE,
        usecols=[
            "parcelid",
            "saleprice",
            "saledate",
            "landusefulldescription",
            "landuse",
            "deeddescription",
            "salesvalidity",
        ],
        low_memory=False,
    )

    print(f"Sales records: {len(sales):,}")

    # Parse dates
    sales["saledate"] = pd.to_datetime(
        sales["saledate"],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # SALES VALIDITY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("SALES VALIDITY CODES")
    print("-" * 80)

    validity = (
        sales.groupby("salesvalidity", dropna=False)
        .agg(
            records=("parcelid", "size"),
            positive_price=(
                "saleprice",
                lambda x: (x > 0).sum(),
            ),
            median_price=(
                "saleprice",
                lambda x: x[x > 0].median()
                if (x > 0).any()
                else 0,
            ),
        )
        .sort_values("records", ascending=False)
    )

    print(validity.to_string())

    # ------------------------------------------------------------------
    # VALIDITY + PROPERTY TYPE
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("VALIDITY BY MAJOR PROPERTY TYPE")
    print("-" * 80)

    residential_mask = sales[
        "landusefulldescription"
    ].fillna("").str.contains(
        "SINGLE FAMILY RESIDENTIAL",
        case=False,
        regex=False,
    )

    residential_sales = sales[residential_mask].copy()

    print(
        f"Single-family-like records: "
        f"{len(residential_sales):,}"
    )

    residential_validity = (
        residential_sales["salesvalidity"]
        .value_counts(dropna=False)
    )

    print(residential_validity.to_string())

    # ------------------------------------------------------------------
    # POSITIVE-PRICE RESIDENTIAL SALES
    # ------------------------------------------------------------------

    residential_positive = residential_sales[
        residential_sales["saleprice"] > 0
    ].copy()

    print("\n" + "-" * 80)
    print("POSITIVE-PRICE SINGLE-FAMILY SALES")
    print("-" * 80)

    print(
        f"Records: {len(residential_positive):,}"
    )

    print("\nValidity codes:")

    print(
        residential_positive[
            "salesvalidity"
        ]
        .value_counts(dropna=False)
        .to_string()
    )

    # ------------------------------------------------------------------
    # PRICE DISTRIBUTION BY VALIDITY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("PRICE STATISTICS BY VALIDITY")
    print("-" * 80)

    price_by_validity = (
        residential_positive
        .groupby("salesvalidity", dropna=False)
        ["saleprice"]
        .agg(
            count="count",
            min="min",
            median="median",
            mean="mean",
            max="max",
        )
        .sort_values("count", ascending=False)
    )

    print(price_by_validity.to_string())

    # ------------------------------------------------------------------
    # DEED TYPES
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("TOP DEED DESCRIPTIONS FOR RESIDENTIAL SALES")
    print("-" * 80)

    print(
        residential_positive[
            "deeddescription"
        ]
        .value_counts(dropna=False)
        .head(40)
        .to_string()
    )

    # ------------------------------------------------------------------
    # EXTREME PRICES
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("HIGHEST SINGLE-FAMILY SALE PRICES")
    print("-" * 80)

    extreme = (
        residential_positive
        .sort_values("saleprice", ascending=False)
        .head(30)
    )

    print(
        extreme[
            [
                "parcelid",
                "saleprice",
                "saledate",
                "landusefulldescription",
                "salesvalidity",
                "deeddescription",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()