from src.utils.location_service import (
    calculate_distance_to_uptown,
    validate_zipcode,
    find_nearest_neighborhood,
)


def test_location_distance_is_zero_at_uptown():

    distance = calculate_distance_to_uptown(
        latitude=35.2271,
        longitude=-80.8431,
    )

    assert distance < 0.001


def test_valid_charlotte_zipcode():

    zipcode = validate_zipcode(
        "28205"
    )

    assert zipcode == "28205"


def test_invalid_zipcode_format_is_rejected():

    try:

        validate_zipcode(
            "2820"
        )

        assert False, "Expected ValueError"

    except ValueError as exc:

        assert (
            "exactly 5 digits"
            in str(exc)
        )


def test_unsupported_zipcode_is_rejected():

    try:

        validate_zipcode(
            "10001"
        )

        assert False, "Expected ValueError"

    except ValueError as exc:

        assert (
            "not currently recognized"
            in str(exc)
        )


def test_nearest_neighborhood_lookup_returns_valid_result():

    result = find_nearest_neighborhood(
        latitude=35.22201511475,
        longitude=-80.83921414192,
    )

    assert "neighborhood" in result
    assert "nearest_parcelid" in result
    assert "nearest_parcel_distance_miles" in result
    assert "location_confidence" in result

    assert result["neighborhood"]
    assert result["nearest_parcelid"]

    assert (
        result[
            "nearest_parcel_distance_miles"
        ]
        >= 0
    )

    assert result["location_confidence"] in {
        "high",
        "medium",
        "low",
        "unreliable",
    }