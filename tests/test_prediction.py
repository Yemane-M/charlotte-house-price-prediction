import math

from src.models.predict_house_price import (
    calculate_distance_to_uptown,
    build_prediction_input,
    predict_house_price,
)


def test_distance_to_uptown_is_zero_at_reference_point():

    distance = calculate_distance_to_uptown(
        latitude=35.2271,
        longitude=-80.8431,
    )

    assert distance < 0.001


def test_build_prediction_input_creates_expected_features():

    input_data = build_prediction_input(
        sale_year=2026,
        year_built=2000,
        gisacres=0.25,
        heatedarea=2500,
        finisharea=2500,
        bedrooms=4,
        bathroom_equivalents=2.5,
        fireplaces=1,
        fingarage=400,
        finattic=0,
        landusefulldescription="SINGLE FAMILY RESIDENTIAL",
        neighborhood="A516",
        latitude=35.2271,
        longitude=-80.8431,
    )

    assert len(input_data) == 1

    assert input_data["property_age"].iloc[0] == 26

    assert math.isclose(
        input_data["distance_to_uptown_miles"].iloc[0],
        0.0,
        abs_tol=0.001,
    )

    assert input_data["neighborhood"].iloc[0] == "A516"


def test_prediction_returns_valid_positive_price():

    result = predict_house_price(
        sale_year=2019,
        year_built=2000,
        gisacres=0.3269094,
        heatedarea=2856,
        finisharea=2856,
        bedrooms=4,
        bathroom_equivalents=2.5,
        fireplaces=1,
        fingarage=400,
        finattic=0,
        landusefulldescription="SINGLE FAMILY RESIDENTIAL",
        neighborhood="A516",
        latitude=35.462958,
        longitude=-80.86250035,
    )

    assert "predicted_price" in result
    assert "property_age" in result
    assert "distance_to_uptown_miles" in result

    assert result["predicted_price"] > 0
    assert result["property_age"] == 19


def test_negative_numeric_input_is_rejected():

    try:

        build_prediction_input(
            sale_year=2026,
            year_built=2000,
            gisacres=-1,
            heatedarea=2500,
            finisharea=2500,
            bedrooms=4,
            bathroom_equivalents=2.5,
            fireplaces=1,
            fingarage=400,
            finattic=0,
            landusefulldescription="SINGLE FAMILY RESIDENTIAL",
            neighborhood="A516",
            latitude=35.2271,
            longitude=-80.8431,
        )

        assert False, "Expected ValueError"

    except ValueError as exc:

        assert "gisacres cannot be negative" in str(exc)