from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT / MODEL PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "xgboost_production.joblib"
)


# ============================================================
# CONSTANTS
# ============================================================

# Same Uptown Charlotte reference point used
# during feature engineering.
UPTOWN_LATITUDE = 35.2271
UPTOWN_LONGITUDE = -80.8431

EARTH_RADIUS_MILES = 3958.8


# Exact feature order used during model training.
FEATURES = [
    "sale_year",
    "gisacres",
    "heatedarea",
    "finisharea",
    "bedrooms",
    "bathroom_equivalents",
    "fireplaces",
    "fingarage",
    "finattic",
    "property_age",
    "landusefulldescription",
    "neighborhood",
    "latitude",
    "longitude",
    "distance_to_uptown_miles",
]


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained production model.

    The model is loaded only once and then cached.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Production model was not found at: "
                f"{MODEL_PATH}"
            )

        _model = joblib.load(
            MODEL_PATH
        )

    return _model


# ============================================================
# DISTANCE TO UPTOWN
# ============================================================

def calculate_distance_to_uptown(
    latitude,
    longitude,
):
    """
    Calculate straight-line distance from a property
    to Uptown Charlotte using the Haversine formula.

    Returns distance in miles.
    """

    latitude_radians = np.radians(
        float(latitude)
    )

    longitude_radians = np.radians(
        float(longitude)
    )

    uptown_latitude_radians = np.radians(
        UPTOWN_LATITUDE
    )

    uptown_longitude_radians = np.radians(
        UPTOWN_LONGITUDE
    )

    delta_latitude = (
        latitude_radians
        - uptown_latitude_radians
    )

    delta_longitude = (
        longitude_radians
        - uptown_longitude_radians
    )

    a = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(uptown_latitude_radians)
        * np.cos(latitude_radians)
        * np.sin(delta_longitude / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    distance = (
        EARTH_RADIUS_MILES
        * c
    )

    return float(distance)


# ============================================================
# HELPER FOR OPTIONAL NUMERIC VALUES
# ============================================================

def optional_float(value):
    """
    Convert optional numeric input to float.

    None or an empty string becomes NaN so the
    model's existing imputer can handle it.
    """

    if value is None:
        return np.nan

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return np.nan

    return float(value)


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(
    sale_year,
    year_built,
    latitude,
    longitude,
    gisacres,
    heatedarea,
    finisharea,
    bedrooms,
    bathroom_equivalents,
    fireplaces,
    fingarage,
    finattic,
    landusefulldescription,
    neighborhood,
):
    """
    Perform basic validation before prediction.
    """

    sale_year = int(sale_year)
    year_built = int(year_built)

    latitude = float(latitude)
    longitude = float(longitude)

    if sale_year < 1900 or sale_year > 2100:
        raise ValueError(
            "sale_year must be between "
            "1900 and 2100."
        )

    if year_built < 1700 or year_built > 2100:
        raise ValueError(
            "year_built must be between "
            "1700 and 2100."
        )

    if not (
        34.5 <= latitude <= 36.0
    ):
        raise ValueError(
            "Latitude does not appear to be "
            "within the Charlotte region."
        )

    if not (
        -81.5 <= longitude <= -80.0
    ):
        raise ValueError(
            "Longitude does not appear to be "
            "within the Charlotte region."
        )

    if not landusefulldescription:
        raise ValueError(
            "landusefulldescription is required."
        )

    if not neighborhood:
        raise ValueError(
            "neighborhood is required."
        )

    numeric_values = {
        "gisacres": gisacres,
        "heatedarea": heatedarea,
        "finisharea": finisharea,
        "bedrooms": bedrooms,
        "bathroom_equivalents":
            bathroom_equivalents,
        "fireplaces": fireplaces,
        "fingarage": fingarage,
        "finattic": finattic,
    }

    for name, value in numeric_values.items():

        converted = optional_float(
            value
        )

        if (
            not np.isnan(converted)
            and converted < 0
        ):
            raise ValueError(
                f"{name} cannot be negative."
            )


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_prediction_input(
    sale_year,
    year_built,
    gisacres,
    heatedarea,
    finisharea,
    bedrooms,
    bathroom_equivalents,
    fireplaces,
    fingarage,
    finattic,
    landusefulldescription,
    neighborhood,
    latitude,
    longitude,
):
    """
    Convert raw property information into the exact
    15 features expected by the production model.
    """

    validate_input(
        sale_year=sale_year,
        year_built=year_built,
        latitude=latitude,
        longitude=longitude,
        gisacres=gisacres,
        heatedarea=heatedarea,
        finisharea=finisharea,
        bedrooms=bedrooms,
        bathroom_equivalents=
            bathroom_equivalents,
        fireplaces=fireplaces,
        fingarage=fingarage,
        finattic=finattic,
        landusefulldescription=
            landusefulldescription,
        neighborhood=neighborhood,
    )

    sale_year = int(
        sale_year
    )

    year_built = int(
        year_built
    )

    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )


    # --------------------------------------------------------
    # PROPERTY AGE
    # --------------------------------------------------------

    property_age = (
        sale_year
        - year_built
    )

    # Match the training-data cleaning rule:
    # negative property ages were converted to missing.
    if property_age < 0:
        property_age = np.nan


    # --------------------------------------------------------
    # DISTANCE TO UPTOWN
    # --------------------------------------------------------

    distance_to_uptown = (
        calculate_distance_to_uptown(
            latitude,
            longitude,
        )
    )


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "sale_year":
                    sale_year,

                "gisacres":
                    optional_float(
                        gisacres
                    ),

                "heatedarea":
                    optional_float(
                        heatedarea
                    ),

                "finisharea":
                    optional_float(
                        finisharea
                    ),

                "bedrooms":
                    optional_float(
                        bedrooms
                    ),

                "bathroom_equivalents":
                    optional_float(
                        bathroom_equivalents
                    ),

                "fireplaces":
                    optional_float(
                        fireplaces
                    ),

                "fingarage":
                    optional_float(
                        fingarage
                    ),

                "finattic":
                    optional_float(
                        finattic
                    ),

                "property_age":
                    property_age,

                "landusefulldescription":
                    str(
                        landusefulldescription
                    ).strip(),

                "neighborhood":
                    str(
                        neighborhood
                    ).strip(),

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "distance_to_uptown_miles":
                    distance_to_uptown,
            }
        ]
    )

    # Guarantee exact feature order.
    input_data = input_data[
        FEATURES
    ]

    return input_data


# ============================================================
# PREDICT HOUSE PRICE
# ============================================================

def predict_house_price(
    sale_year,
    year_built,
    gisacres,
    heatedarea,
    finisharea,
    bedrooms,
    bathroom_equivalents,
    fireplaces,
    fingarage,
    finattic,
    landusefulldescription,
    neighborhood,
    latitude,
    longitude,
):
    """
    Predict a Mecklenburg County property sale price.

    Returns a dictionary containing:
        predicted_price
        property_age
        distance_to_uptown_miles
    """

    model = load_model()

    input_data = build_prediction_input(
        sale_year=sale_year,
        year_built=year_built,
        gisacres=gisacres,
        heatedarea=heatedarea,
        finisharea=finisharea,
        bedrooms=bedrooms,
        bathroom_equivalents=
            bathroom_equivalents,
        fireplaces=fireplaces,
        fingarage=fingarage,
        finattic=finattic,
        landusefulldescription=
            landusefulldescription,
        neighborhood=neighborhood,
        latitude=latitude,
        longitude=longitude,
    )


    # Model was trained on log1p(saleprice).
    predicted_log_price = (
        model.predict(
            input_data
        )[0]
    )


    # Convert prediction back to dollars.
    predicted_price = np.expm1(
        predicted_log_price
    )


    # Price cannot be negative.
    predicted_price = max(
        float(predicted_price),
        0.0,
    )


    return {
        "predicted_price":
            round(
                predicted_price,
                2
            ),

        "property_age":
            input_data[
                "property_age"
            ].iloc[0],

        "distance_to_uptown_miles":
            round(
                float(
                    input_data[
                        "distance_to_uptown_miles"
                    ].iloc[0]
                ),
                2,
            ),
    }


# ============================================================
# BASIC MODEL LOAD TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Checking production "
        "prediction module..."
    )

    print(
        f"\nModel path:\n{MODEL_PATH}"
    )

    model = load_model()

    print(
        "\nProduction model loaded "
        "successfully."
    )

    print(
        f"\nExpected model features: "
        f"{len(FEATURES)}"
    )

    for feature in FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        "\nPrediction module is ready."
    )