import pandas as pd


CAMA_PATH = "data/raw/mecklenburg/Cama_Table.csv"


print("Loading CAMA coordinates...")

df = pd.read_csv(
    CAMA_PATH,
    usecols=[
        "parcelid",
        "xcoord",
        "ycoord",
    ],
    low_memory=False,
)

print(f"Rows: {len(df):,}")
print(f"Unique parcels: {df['parcelid'].nunique():,}")


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

df["xcoord"] = pd.to_numeric(
    df["xcoord"],
    errors="coerce"
)

df["ycoord"] = pd.to_numeric(
    df["ycoord"],
    errors="coerce"
)


# ============================================================
# BASIC QUALITY
# ============================================================

print("\n" + "=" * 60)
print("COORDINATE QUALITY")
print("=" * 60)

print("\nMissing values:")

print(
    df[
        ["xcoord", "ycoord"]
    ].isna().sum()
)

print("\nZero values:")

print(
    (df[
        ["xcoord", "ycoord"]
    ] == 0).sum()
)


# ============================================================
# DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("COORDINATE DISTRIBUTION")
print("=" * 60)

print(
    df[
        ["xcoord", "ycoord"]
    ].describe(
        percentiles=[
            0.001,
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
            0.999,
        ]
    )
)


# ============================================================
# COORDINATES PER PARCEL
# ============================================================

print("\n" + "=" * 60)
print("COORDINATES PER PARCEL")
print("=" * 60)

coord_counts = (
    df.dropna(
        subset=["xcoord", "ycoord"]
    )
    .groupby("parcelid")
    .agg(
        x_values=("xcoord", "nunique"),
        y_values=("ycoord", "nunique"),
    )
)

multiple_coords = coord_counts[
    (coord_counts["x_values"] > 1)
    | (coord_counts["y_values"] > 1)
]

print(
    "Parcels with multiple coordinate values: "
    f"{len(multiple_coords):,}"
)


# ============================================================
# EXACT DUPLICATE COORDINATES
# ============================================================

coordinate_frequency = (
    df.dropna(
        subset=["xcoord", "ycoord"]
    )
    .groupby(
        ["xcoord", "ycoord"]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print("\nMost frequently repeated coordinate pairs:")

print(
    coordinate_frequency
    .head(20)
    .to_string()
)


# ============================================================
# SAMPLE COORDINATES
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE COORDINATES")
print("=" * 60)

print(
    df[
        [
            "parcelid",
            "xcoord",
            "ycoord",
        ]
    ]
    .dropna()
    .head(30)
    .to_string(index=False)
)


print("\nCoordinate analysis completed.")