from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/mecklenburg")
SALES_FILE = DATA_DIR / "Parcel_Sales_Table.csv"


# Official Mecklenburg County sale-validity codes that we
# do NOT want to use as normal market transactions.
EXCLUDED_VALIDITY_CODES = {
    "A",    # Conveyance of 3 or more parcels
    "B",    # Improvements not included in sale
    "C",    # Transaction is for $3,000 or less
    "D",    # Deed date outside study period
    "E",    # Transaction between relatives/related party
    "F",    # Grantor conveying a fraction interest
    "FC",   # Foreclosure
    "G",    # Grantor reserves interest
    "H",    # Grantor reserves possession
    "I",    # Government/public utility/lending institution
    "J",    # Tax exempt property/cemetery
    "K",    # Church/school/lodge/nonprofit
    "L",    # Deed of trust indicates price discrepancy
    "M",    # Property situated in more than one county
    "N",    # Minerals/timber/other rights
    "O",    # Personal property included
    "P",    # Forced sale/auction
    "PB",   # Probate
    "Q",    # Contract made before study period
    "R",    # Trade/exchange/loan assumption
    "S",    # Real property not identifiable
    "TEMP", # Temporarily disqualified
    "U",    # Parcels assembled for development
    "UC",   # Under construction/remodeling incomplete
    "UR",   # Under review
    "V",    # Condominium declaration
    "X",    # Other
    "Y",    # Demo sale
    "Z",    # Builder sale or other miscellaneous
    "CD",  # Condo
}


def main() -> None:
    print("=" * 80)
    print("MECKLENBURG COUNTY SALES FILTER")
    print("=" * 80)

    print("\nLoading sales data...")

    sales = pd.read_csv(
        SALES_FILE,
        usecols=[
            "parcelid",
            "transferid",
            "propertyid",
            "saleprice",
            "saledate",
            "landusefulldescription",
            "landuse",
            "deeddescription",
            "salesvalidity",
        ],
        low_memory=False,
    )

    print(f"Total sales records loaded: {len(sales):,}")

    # ------------------------------------------------------------------
    # Step 1: Convert sale date
    # ------------------------------------------------------------------

    sales["saledate"] = pd.to_datetime(
        sales["saledate"],
        errors="coerce",
    )

    print(
        f"Valid sale dates: "
        f"{sales['saledate'].notna().sum():,}"
    )

    # ------------------------------------------------------------------
    # Step 2: Keep positive sale prices
    # ------------------------------------------------------------------

    positive_price = sales["saleprice"] > 0

    sales_positive = sales[positive_price].copy()

    print("\n" + "-" * 80)
    print("POSITIVE-PRICE SALES")
    print("-" * 80)

    print(
        f"Positive-price records: "
        f"{len(sales_positive):,}"
    )

    print(
        f"Median sale price: "
        f"${sales_positive['saleprice'].median():,.0f}"
    )

    # ------------------------------------------------------------------
    # Step 3: Identify single-family residential properties
    # ------------------------------------------------------------------

    single_family_mask = (
        sales_positive["landusefulldescription"]
        .fillna("")
        .str.contains(
            "SINGLE FAMILY RESIDENTIAL",
            case=False,
            regex=False,
        )
    )

    single_family = sales_positive[
        single_family_mask
    ].copy()

    print("\n" + "-" * 80)
    print("SINGLE-FAMILY RESIDENTIAL")
    print("-" * 80)

    print(
        f"Single-family records: "
        f"{len(single_family):,}"
    )

    # ------------------------------------------------------------------
    # Step 4: Show validity-code distribution
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("VALIDITY CODES BEFORE FILTERING")
    print("-" * 80)

    validity_counts = (
        single_family["salesvalidity"]
        .fillna("[BLANK]")
        .value_counts()
    )

    print(validity_counts.to_string())

    # ------------------------------------------------------------------
    # Step 5: Filter validity codes
    # ------------------------------------------------------------------

    # Blank validity is intentionally retained.
    valid_market_mask = (
        single_family["salesvalidity"].isna()
        | ~single_family["salesvalidity"].isin(
            EXCLUDED_VALIDITY_CODES
        )
    )

    filtered_sales = single_family[
        valid_market_mask
    ].copy()

    # ------------------------------------------------------------------
    # Step 6: Results
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FILTERED CANDIDATE SALES")
    print("=" * 80)

    print(
        f"Candidate records: "
        f"{len(filtered_sales):,}"
    )

    print(
        f"Removed records: "
        f"{len(single_family) - len(filtered_sales):,}"
    )

    print(
        f"Median sale price: "
        f"${filtered_sales['saleprice'].median():,.0f}"
    )

    print(
        f"Mean sale price: "
        f"${filtered_sales['saleprice'].mean():,.0f}"
    )

    print(
        f"Minimum sale price: "
        f"${filtered_sales['saleprice'].min():,.0f}"
    )

    print(
        f"Maximum sale price: "
        f"${filtered_sales['saleprice'].max():,.0f}"
    )

    # ------------------------------------------------------------------
    # Step 7: Remaining validity codes
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("VALIDITY CODES AFTER FILTERING")
    print("-" * 80)

    remaining_codes = (
        filtered_sales["salesvalidity"]
        .fillna("[BLANK]")
        .value_counts()
    )

    print(remaining_codes.to_string())

    # ------------------------------------------------------------------
    # Step 8: Sale-date range
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("SALE DATE RANGE")
    print("-" * 80)

    print(
        f"Earliest sale: "
        f"{filtered_sales['saledate'].min()}"
    )

    print(
        f"Latest sale: "
        f"{filtered_sales['saledate'].max()}"
    )

    # ------------------------------------------------------------------
    # Step 9: Number of unique parcels
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("PARCEL COUNTS")
    print("-" * 80)

    print(
        f"Unique parcels: "
        f"{filtered_sales['parcelid'].nunique():,}"
    )

    print(
        f"Sales per parcel - median: "
        f"{filtered_sales.groupby('parcelid').size().median():.0f}"
    )

    # ------------------------------------------------------------------
    # Step 10: Sales by year
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("SALES BY YEAR")
    print("-" * 80)

    filtered_sales["sale_year"] = (
        filtered_sales["saledate"].dt.year
    )

    yearly_summary = (
        filtered_sales
        .groupby("sale_year")
        .agg(
            sales=("parcelid", "size"),
            unique_parcels=("parcelid", "nunique"),
            median_price=("saleprice", "median"),
            mean_price=("saleprice", "mean"),
            min_price=("saleprice", "min"),
            max_price=("saleprice", "max"),
        )
        .sort_index()
    )

    print(yearly_summary.to_string())

    # ------------------------------------------------------------------
    # Step 11: Investigate AS and AG
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("UNDEFINED VALIDITY CODES: AS AND AG")
    print("-" * 80)

    for code in ["AS", "AG"]:
        code_sales = single_family[
            single_family["salesvalidity"] == code
        ].copy()

        print(f"\nValidity code: {code}")
        print(f"Records: {len(code_sales):,}")

        print(
            f"Median price: "
            f"${code_sales['saleprice'].median():,.0f}"
        )

        print(
            f"Mean price: "
            f"${code_sales['saleprice'].mean():,.0f}"
        )

        print(
            f"Minimum price: "
            f"${code_sales['saleprice'].min():,.0f}"
        )

        print(
            f"Maximum price: "
            f"${code_sales['saleprice'].max():,.0f}"
        )

        print("\nLand-use descriptions:")
        print(
            code_sales["landusefulldescription"]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

        print("\nDeed descriptions:")
        print(
            code_sales["deeddescription"]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

        print("\nSample transactions:")
        print(
            code_sales[
                [
                    "parcelid",
                    "saleprice",
                    "saledate",
                    "landusefulldescription",
                    "landuse",
                    "deeddescription",
                    "salesvalidity",
                ]
            ]
            .sort_values("saleprice", ascending=False)
            .head(20)
            .to_string(index=False)
        )
    
    # ------------------------------------------------------------------
    # Step 12: Define modeling dataset
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BUILDING INITIAL MODELING DATASET")
    print("=" * 80)

    # Normalize sales validity codes.
    # Missing/blank values will be represented as an empty string.

    validity = (
        single_family["salesvalidity"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print("\nValidity codes in single-family dataset:")
    print(
        validity.value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    # Keep only transactions with no sales-validity exclusion code.
    # AS and AG are intentionally excluded because their definitions
    # have not been established.

    model_sales = single_family[validity == ""].copy()

    print("\nModel sales after blank-validity filtering:")
    print(f"Records: {len(model_sales):,}")

    # ------------------------------------------------------------------
    # Study period
    # ------------------------------------------------------------------

    START_DATE = "2018-01-01"
    END_DATE = "2026-12-31"

    model_sales = model_sales[
        (model_sales["saledate"] >= START_DATE)
        & (model_sales["saledate"] <= END_DATE)
    ].copy()

    print(f"Study period: {START_DATE} to {END_DATE}")
    print(f"Records after date filtering: {len(model_sales):,}")

    # ------------------------------------------------------------------
    # Price distribution
    # ------------------------------------------------------------------

    print("\nPrice distribution:")
    print(
        model_sales["saleprice"]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
    )

    print("\nUnique parcels:")
    print(model_sales["parcelid"].nunique())

    # ------------------------------------------------------------------
    # Annual summary
    # ------------------------------------------------------------------

    model_sales["sale_year"] = model_sales["saledate"].dt.year

    annual_summary = (
        model_sales
        .groupby("sale_year")
        .agg(
            sales=("saleprice", "count"),
            unique_parcels=("parcelid", "nunique"),
            median_price=("saleprice", "median"),
            mean_price=("saleprice", "mean"),
            min_price=("saleprice", "min"),
            max_price=("saleprice", "max"),
        )
    )

    print("\nAnnual summary:")
    print(annual_summary.to_string())

    # ------------------------------------------------------------------
    # Highest-price transactions
    # ------------------------------------------------------------------

    print("\nTop 50 highest-price transactions:")

    print(
        model_sales[
            [
                "parcelid",
                "saleprice",
                "saledate",
                "landusefulldescription",
                "landuse",
                "deeddescription",
                "salesvalidity",
            ]
        ]
        .sort_values("saleprice", ascending=False)
        .head(50)
        .to_string(index=False)
    )
   
    # ------------------------------------------------------------------
    # Step 13: Investigate duplicate sales records
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DUPLICATE SALES INVESTIGATION")
    print("=" * 80)

    # Count exact duplicates across the entire modeling dataset.
    duplicate_count = model_sales.duplicated().sum()

    print(f"\nExact duplicate rows: {duplicate_count:,}")

    # Investigate duplicates using the core transaction fields.
    transaction_columns = [
        "parcelid",
        "transferid",
        "propertyid",
        "saleprice",
        "saledate",
        "deeddescription",
        "salesvalidity",
    ]

    duplicate_transactions = (
        model_sales[
            model_sales.duplicated(
                subset=transaction_columns,
                keep=False
            )
        ]
        .sort_values(
            ["saleprice", "saledate", "parcelid"],
            ascending=[False, True, True]
        )
    )

    print(
        f"Rows involved in duplicate transactions: "
        f"{len(duplicate_transactions):,}"
    )

    print("\nTop duplicate transaction groups:")

    duplicate_groups = (
        duplicate_transactions
        .groupby(transaction_columns, dropna=False)
        .size()
        .reset_index(name="duplicate_count")
        .sort_values("duplicate_count", ascending=False)
    )

    print(
        duplicate_groups
        .head(30)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Investigate the highest-price duplicate transactions
    # ------------------------------------------------------------------

    print("\nHighest-price duplicate transactions:")

    print(
        duplicate_transactions[
            [
                "parcelid",
                "transferid",
                "propertyid",
                "saleprice",
                "saledate",
                "landusefulldescription",
                "landuse",
                "deeddescription",
                "salesvalidity",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Step 14: Remove duplicate transaction records
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DEDUPLICATING SALES TRANSACTIONS")
    print("=" * 80)

    before_dedup = len(model_sales)

    transaction_key = [
        "parcelid",
        "transferid",
        "propertyid",
        "saleprice",
        "saledate",
    ]

    model_sales = model_sales.drop_duplicates(
        subset=transaction_key,
        keep="first"
    ).copy()

    after_dedup = len(model_sales)

    print(f"\nRecords before deduplication: {before_dedup:,}")
    print(f"Records after deduplication:  {after_dedup:,}")
    print(f"Duplicate records removed:   {before_dedup - after_dedup:,}")

    print(
        f"Unique parcels after deduplication: "
        f"{model_sales['parcelid'].nunique():,}"
    )

    print(
        f"Unique transactions after deduplication: "
        f"{model_sales['transferid'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # Step 15: Analyze price distribution after deduplication
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PRICE DISTRIBUTION AFTER DEDUPLICATION")
    print("=" * 80)

    print("\nPrice statistics:")

    print(
        model_sales["saleprice"].describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
    )

    print("\nSelected price percentiles:")

    percentiles = model_sales["saleprice"].quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    )

    for percentile, value in percentiles.items():
        print(f"{percentile * 100:6.2f}% : ${value:,.0f}")
    
    # ------------------------------------------------------------------
    # Step 16: Investigate extreme-price transactions
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXTREME-PRICE TRANSACTIONS")
    print("=" * 80)

    extreme_prices = model_sales[
        model_sales["saleprice"] >= 5_000_000
    ].copy()

    print(
        f"\nTransactions with sale price >= $5M: "
        f"{len(extreme_prices):,}"
    )

    print(
        f"Unique parcels with sale price >= $5M: "
        f"{extreme_prices['parcelid'].nunique():,}"
    )

    print("\nExtreme-price transactions:")

    extreme_columns = [
        "parcelid",
        "transferid",
        "propertyid",
        "saleprice",
        "saledate",
        "landusefulldescription",
        "landuse",
        "deeddescription",
        "salesvalidity",
    ]

    print(
        extreme_prices[
            extreme_columns
        ]
        .sort_values("saleprice", ascending=False)
        .to_string(index=False)
    )
    
    # ------------------------------------------------------------------
    # Step 17: Investigate unusually low-price transactions
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOW-PRICE TRANSACTIONS")
    print("=" * 80)

    low_prices = model_sales[
        model_sales["saleprice"] <= 100_000
    ].copy()

    print(
        f"\nTransactions with sale price <= $100K: "
        f"{len(low_prices):,}"
    )

    print(
        f"Unique parcels with sale price <= $100K: "
        f"{low_prices['parcelid'].nunique():,}"
    )

    print("\nLowest-price transactions:")

    low_price_columns = [
        "parcelid",
        "transferid",
        "propertyid",
        "saleprice",
        "saledate",
        "landusefulldescription",
        "landuse",
        "deeddescription",
        "salesvalidity",
    ]

    print(
        low_prices[
            low_price_columns
        ]
        .sort_values("saleprice", ascending=True)
        .head(100)
        .to_string(index=False)
    )

    output_file = Path("data/interim/model_sales.csv")

    model_sales.to_csv(
        output_file,
        index=False
    )

    # ------------------------------------------------------------------
    # Filtering complete
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FILTERING STEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
