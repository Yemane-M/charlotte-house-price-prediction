from unittest.mock import patch

from app.app import app


def test_home_page_loads():

    app.config["TESTING"] = True

    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200

    assert b"Charlotte House Price Prediction" in response.data


def test_prediction_form_submission():

    app.config["TESTING"] = True

    client = app.test_client()

    mock_location_result = {
        "input_address": "4122 CENTRAL AVE, Charlotte, NC 28205",
        "matched_address": "4122 CENTRAL AVE, CHARLOTTE, NC, 28205",
        "zipcode": "28205",
        "latitude": 35.214,
        "longitude": -80.777,
        "neighborhood": "N109",
        "distance_to_uptown_miles": 3.93,
        "nearest_parcelid": "TEST123",
        "nearest_parcel_distance_miles": 0.05,
        "location_confidence": "high",
    }

    mock_prediction_result = {
        "predicted_price": 664959.62,
        "property_age": 58,
        "distance_to_uptown_miles": 3.93,
    }

    with patch(
        "app.app.resolve_property_location",
        return_value=mock_location_result,
    ), patch(
        "app.app.predict_house_price",
        return_value=mock_prediction_result,
    ):

        response = client.post(
            "/",
            data={
                "street": "4122 CENTRAL AVE",
                "city": "Charlotte",
                "state": "NC",
                "zipcode": "28205",
                "sale_year": "2026",
                "year_built": "1968",
                "gisacres": "0.25",
                "heatedarea": "2200",
                "finisharea": "2200",
                "bedrooms": "3",
                "bathroom_equivalents": "2.0",
                "fireplaces": "1",
                "fingarage": "400",
                "finattic": "0",
                "landusefulldescription":
                    "SINGLE FAMILY RESIDENTIAL",
            },
        )

    assert response.status_code == 200

    assert b"664,959.62" in response.data

    assert b"N109" in response.data

    assert b"High" in response.data


def test_prediction_error_is_displayed():

    app.config["TESTING"] = True

    client = app.test_client()

    with patch(
        "app.app.resolve_property_location",
        side_effect=ValueError(
            "The address could not be found."
        ),
    ):

        response = client.post(
            "/",
            data={
                "street": "Invalid Address",
                "city": "Charlotte",
                "state": "NC",
                "zipcode": "28205",
                "sale_year": "2026",
                "year_built": "2000",
                "gisacres": "0.25",
                "heatedarea": "2000",
                "finisharea": "2000",
                "bedrooms": "3",
                "bathroom_equivalents": "2",
                "fireplaces": "1",
                "fingarage": "400",
                "finattic": "0",
                "landusefulldescription":
                    "SINGLE FAMILY RESIDENTIAL",
            },
        )

    assert response.status_code == 200

    assert b"Unable to Complete Prediction" in response.data

    assert b"The address could not be found." in response.data