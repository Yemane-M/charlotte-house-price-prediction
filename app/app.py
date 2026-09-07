from flask import Flask, render_template, request

from src.models.predict_house_price import predict_house_price
from src.utils.location_service import resolve_property_location


app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    location_result = None
    error_message = None

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # RESOLVE USER ADDRESS
            # ------------------------------------------------

            location_result = resolve_property_location(
                street=request.form.get(
                    "street"
                ),
                city=request.form.get(
                    "city"
                ),
                state=request.form.get(
                    "state"
                ),
                zipcode=request.form.get(
                    "zipcode"
                ),
            )


            # ------------------------------------------------
            # RUN MODEL PREDICTION
            # ------------------------------------------------

            prediction = predict_house_price(
                sale_year=request.form.get(
                    "sale_year"
                ),
                year_built=request.form.get(
                    "year_built"
                ),
                gisacres=request.form.get(
                    "gisacres"
                ),
                heatedarea=request.form.get(
                    "heatedarea"
                ),
                finisharea=request.form.get(
                    "finisharea"
                ),
                bedrooms=request.form.get(
                    "bedrooms"
                ),
                bathroom_equivalents=request.form.get(
                    "bathroom_equivalents"
                ),
                fireplaces=request.form.get(
                    "fireplaces"
                ),
                fingarage=request.form.get(
                    "fingarage"
                ),
                finattic=request.form.get(
                    "finattic"
                ),
                landusefulldescription=request.form.get(
                    "landusefulldescription"
                ),

                # Automatically resolved location features
                neighborhood=location_result[
                    "neighborhood"
                ],

                latitude=location_result[
                    "latitude"
                ],

                longitude=location_result[
                    "longitude"
                ],
            )

        except Exception as exc:

            error_message = str(
                exc
            )


    return render_template(
        "index.html",
        prediction=prediction,
        location_result=location_result,
        error_message=error_message,
    )


# ============================================================
# RUN FLASK APP
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )