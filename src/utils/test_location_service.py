from src.utils.location_service import resolve_property_location


print("Testing Charlotte property location lookup...\n")

result = resolve_property_location(
    street="600 E 4th St",
    city="Charlotte",
    state="NC",
    zipcode="28202",
)


print("=" * 65)
print("LOCATION LOOKUP RESULT")
print("=" * 65)

print(
    f"Input address              : "
    f"{result['input_address']}"
)

print(
    f"Matched address            : "
    f"{result['matched_address']}"
)

print(
    f"ZIP code                   : "
    f"{result['zipcode']}"
)

print(
    f"Latitude                   : "
    f"{result['latitude']}"
)

print(
    f"Longitude                  : "
    f"{result['longitude']}"
)

print(
    f"Model neighborhood         : "
    f"{result['neighborhood']}"
)

print(
    f"Distance to Uptown         : "
    f"{result['distance_to_uptown_miles']} miles"
)

print(
    f"Nearest known parcel       : "
    f"{result['nearest_parcelid']}"
)

print(
    f"Distance to nearest parcel : "
    f"{result['nearest_parcel_distance_miles']} miles"
)

print(
    f"Location confidence        : "
    f"{result['location_confidence']}"
)

print("\nLocation lookup completed successfully.")