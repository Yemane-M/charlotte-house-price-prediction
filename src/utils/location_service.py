from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODELING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "location_lookup.csv"
)


# ============================================================
# CHARLOTTE / MODEL CONSTANTS
# ============================================================

UPTOWN_LATITUDE = 35.2271
UPTOWN_LONGITUDE = -80.8431

EARTH_RADIUS_MILES = 3958.8


# Common residential Charlotte ZIP codes.
#
# We are deliberately using these for ADDRESS validation only.
# ZIP code is NOT currently a feature in the ML model.
CHARLOTTE_ZIP_CODES = {
    "28202",
    "28203",
    "28204",
    "28205",
    "28206",
    "28207",
    "28208",
    "28209",
    "28210",
    "28211",
    "28212",
    "28213",
    "28214",
    "28215",
    "28216",
    "28217",
    "28226",
    "28227",
    "28262",
    "28269",
    "28270",
    "28273",
    "28277",
    "28278",
}


# Broad geographic limits for the Mecklenburg / Charlotte area.
#
# These provide a safety check after geocoding.
MIN_LATITUDE = 35.00
MAX_LATITUDE = 35.52

MIN_LONGITUDE = -81.07
MAX_LONGITUDE = -80.54


# ============================================================
# CENSUS GEOCODER
# ============================================================

CENSUS_GEOCODER_URL = (
    "https://geocoding.geo.census.gov/"
    "geocoder/geographies/onelineaddress"
)


# ============================================================
# CACHE
# ============================================================

_location_lookup = None
_location_tree = None


# ============================================================
# ZIP VALIDATION
# ============================================================

def validate_zipcode(zipcode):
    """
    Validate a Charlotte residential ZIP code.

    ZIP is used for address lookup only.
    It is not passed into the ML model.
    """

    zipcode = str(zipcode).strip()

    if not re.fullmatch(r"\d{5}", zipcode):
        raise ValueError(
            "ZIP code must contain exactly 5 digits."
        )

    if zipcode not in CHARLOTTE_ZIP_CODES:
        raise ValueError(
            f"{zipcode} is not currently recognized as a "
            "supported Charlotte residential ZIP code."
        )

    return zipcode


# ============================================================
# ADDRESS VALIDATION
# ============================================================

def validate_address(
    street,
    city,
    state,
    zipcode,
):
    """
    Validate basic user-entered address information.
    """

    street = str(street).strip()
    city = str(city).strip()
    state = str(state).strip().upper()

    zipcode = validate_zipcode(
        zipcode
    )

    if not street:
        raise ValueError(
            "Street address is required."
        )

    if not city:
        raise ValueError(
            "City is required."
        )

    if city.lower() != "charlotte":
        raise ValueError(
            "This model currently supports "
            "Charlotte, North Carolina addresses."
        )

    if state != "NC":
        raise ValueError(
            "State must be NC."
        )

    return {
        "street": street,
        "city": "Charlotte",
        "state": "NC",
        "zipcode": zipcode,
    }


# ============================================================
# DISTANCE TO UPTOWN
# ============================================================

def calculate_distance_to_uptown(
    latitude,
    longitude,
):
    """
    Calculate Haversine distance to Uptown Charlotte.

    Returns miles.
    """

    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )

    lat1 = np.radians(
        UPTOWN_LATITUDE
    )

    lon1 = np.radians(
        UPTOWN_LONGITUDE
    )

    lat2 = np.radians(
        latitude
    )

    lon2 = np.radians(
        longitude
    )

    delta_lat = (
        lat2 - lat1
    )

    delta_lon = (
        lon2 - lon1
    )

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return float(
        EARTH_RADIUS_MILES * c
    )


# ============================================================
# LOAD LOCATION LOOKUP
# ============================================================

def load_location_lookup():
    """
    Load unique known property coordinates and neighborhood
    codes from the processed modeling dataset.

    A BallTree is created so nearest-property searches are fast.

    This data is loaded only once and cached.
    """

    global _location_lookup
    global _location_tree

    if (
        _location_lookup is not None
        and _location_tree is not None
    ):
        return (
            _location_lookup,
            _location_tree,
        )

    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError(
            "Modeling dataset was not found at: "
            f"{MODELING_DATA_PATH}"
        )

    print(
        "Loading Mecklenburg location lookup..."
    )

    df = pd.read_csv(
        MODELING_DATA_PATH,
        usecols=[
            "parcelid",
            "latitude",
            "longitude",
            "neighborhood",
        ],
        low_memory=False,
        dtype={
            "parcelid": str,
            "neighborhood": str,
        },
    )


    # --------------------------------------------------------
    # REMOVE INVALID LOCATION RECORDS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "neighborhood",
        ]
    ).copy()

    df["neighborhood"] = (
        df["neighborhood"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["neighborhood"] != ""
    ].copy()


    # --------------------------------------------------------
    # ONE LOCATION RECORD PER PARCEL
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "parcelid",
            "latitude",
            "longitude",
            "neighborhood",
        ]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # BUILD HAVERSINE BALL TREE
    # --------------------------------------------------------

    coordinates_radians = np.radians(
        df[
            [
                "latitude",
                "longitude",
            ]
        ].to_numpy()
    )

    tree = BallTree(
        coordinates_radians,
        metric="haversine",
    )

    _location_lookup = df
    _location_tree = tree

    print(
        f"Location records loaded: "
        f"{len(df):,}"
    )

    return (
        _location_lookup,
        _location_tree,
    )


# ============================================================
# FIND NEAREST KNOWN PROPERTY
# ============================================================

def find_nearest_neighborhood(
    latitude,
    longitude,
):
    """
    Find the nearest known parcel from the modeling dataset.

    Returns its neighborhood code and distance from the
    geocoded location.
    """

    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )

    lookup_df, tree = (
        load_location_lookup()
    )

    query_point = np.radians(
        [
            [
                latitude,
                longitude,
            ]
        ]
    )

    distances, indexes = tree.query(
        query_point,
        k=1,
    )

    nearest_index = int(
        indexes[0][0]
    )

    angular_distance = float(
        distances[0][0]
    )

    distance_miles = (
        angular_distance
        * EARTH_RADIUS_MILES
    )

    # --------------------------------------------------------
    # LOCATION CONFIDENCE
    # --------------------------------------------------------

    if distance_miles <= 0.10:
        location_confidence = "high"

    elif distance_miles <= 0.25:
        location_confidence = "medium"

    elif distance_miles <= 0.50:
        location_confidence = "low"

    else:
        location_confidence = "unreliable"

    nearest_record = (
        lookup_df.iloc[
            nearest_index
        ]
    )

    return {
        "neighborhood":
            str(
                nearest_record[
                    "neighborhood"
                ]
            ),

        "nearest_parcelid":
            str(
                nearest_record[
                    "parcelid"
                ]
            ),

        "nearest_parcel_latitude":
            float(
                nearest_record[
                    "latitude"
                ]
            ),

        "nearest_parcel_longitude":
            float(
                nearest_record[
                    "longitude"
                ]
            ),

        "nearest_parcel_distance_miles":
            float(
                distance_miles
            ),

        "location_confidence":
            location_confidence,
    }


# ============================================================
# GEOCODE ADDRESS
# ============================================================

def geocode_address(
    street,
    city="Charlotte",
    state="NC",
    zipcode=None,
):
    """
    Convert a Charlotte street address into latitude
    and longitude using the U.S. Census geocoder.
    """

    validated = validate_address(
        street=street,
        city=city,
        state=state,
        zipcode=zipcode,
    )

    one_line_address = (
        f"{validated['street']}, "
        f"{validated['city']}, "
        f"{validated['state']} "
        f"{validated['zipcode']}"
    )

    params = {
        "address": one_line_address,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }

    try:

        response = requests.get(
            CENSUS_GEOCODER_URL,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        raise RuntimeError(
            "The address lookup service is currently "
            "unavailable. Please try again."
        ) from exc


    # --------------------------------------------------------
    # READ RESPONSE
    # --------------------------------------------------------

    try:

        data = response.json()

        matches = (
            data[
                "result"
            ][
                "addressMatches"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:

        raise RuntimeError(
            "Unexpected response from the address "
            "lookup service."
        ) from exc


    # --------------------------------------------------------
    # NO MATCH
    # --------------------------------------------------------

    if not matches:
        raise ValueError(
            "The address could not be found. "
            "Check the street address and ZIP code "
            "and try again."
        )


    # --------------------------------------------------------
    # FIRST / BEST MATCH
    # --------------------------------------------------------

    match = matches[0]

    coordinates = (
        match[
            "coordinates"
        ]
    )

    latitude = float(
        coordinates[
            "y"
        ]
    )

    longitude = float(
        coordinates[
            "x"
        ]
    )

    matched_address = (
        match.get(
            "matchedAddress",
            one_line_address,
        )
    )


    # --------------------------------------------------------
    # GEOGRAPHIC SAFETY CHECK
    # --------------------------------------------------------

    if not (
        MIN_LATITUDE
        <= latitude
        <= MAX_LATITUDE
    ):
        raise ValueError(
            "The address lookup returned a location "
            "outside the supported Charlotte area."
        )

    if not (
        MIN_LONGITUDE
        <= longitude
        <= MAX_LONGITUDE
    ):
        raise ValueError(
            "The address lookup returned a location "
            "outside the supported Charlotte area."
        )


    return {
        "input_address":
            one_line_address,

        "matched_address":
            matched_address,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "zipcode":
            validated[
                "zipcode"
            ],
    }


# ============================================================
# COMPLETE LOCATION RESOLUTION
# ============================================================

def resolve_property_location(
    street,
    city="Charlotte",
    state="NC",
    zipcode=None,
):
    """
    Resolve a normal user-entered property address into the
    location features required by the ML model.
    """

    # --------------------------------------------------------
    # GEOCODE
    # --------------------------------------------------------

    geocoded = geocode_address(
        street=street,
        city=city,
        state=state,
        zipcode=zipcode,
    )

    latitude = geocoded[
        "latitude"
    ]

    longitude = geocoded[
        "longitude"
    ]


    # --------------------------------------------------------
    # FIND MODEL NEIGHBORHOOD
    # --------------------------------------------------------

    neighborhood_result = (
        find_nearest_neighborhood(
            latitude=latitude,
            longitude=longitude,
        )
    )


    # --------------------------------------------------------
    # DISTANCE TO UPTOWN
    # --------------------------------------------------------

    uptown_distance = (
        calculate_distance_to_uptown(
            latitude=latitude,
            longitude=longitude,
        )
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "input_address":
            geocoded[
                "input_address"
            ],

        "matched_address":
            geocoded[
                "matched_address"
            ],

        "zipcode":
            geocoded[
                "zipcode"
            ],

        "latitude":
            latitude,

        "longitude":
            longitude,

        "neighborhood":
            neighborhood_result[
                "neighborhood"
            ],

        "distance_to_uptown_miles":
            round(
                uptown_distance,
                2,
            ),

        "nearest_parcelid":
            neighborhood_result[
                "nearest_parcelid"
            ],

        "nearest_parcel_distance_miles":
            round(
                neighborhood_result[
                    "nearest_parcel_distance_miles"
                ],
                4,
            ),

        "location_confidence":
            neighborhood_result[
                "location_confidence"
        ],
    }


# ============================================================
# BASIC MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Charlotte property location service"
    )

    print(
        "\nLocation service module loaded successfully."
    )

    print(
        f"\nModeling data path:\n"
        f"{MODELING_DATA_PATH}"
    )

    print(
        "\nSupported residential Charlotte ZIP codes:"
    )

    for zipcode in sorted(
        CHARLOTTE_ZIP_CODES
    ):
        print(
            f"  - {zipcode}"
        )

    print(
        "\nLocation service is ready."
    )