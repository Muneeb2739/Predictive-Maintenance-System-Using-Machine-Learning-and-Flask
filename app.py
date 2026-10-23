
import os
import joblib
import pandas as pd

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename


# ==================================================
# 1. Initialize Flask application
# ==================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ==================================================
# 2. Model input features
# ==================================================

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M"
]


# ==================================================
# 3. Load trained machine learning model
# ==================================================

model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully:", type(model))

except Exception as exception:
    print("Error loading model:", exception)


# ==================================================
# 4. Individual prediction
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None
    probability = None
    risk_category = None

    if request.method == "POST":

        try:
            if model is None:
                raise RuntimeError(
                    "The trained model could not be loaded."
                )

            # Read form values using index.html field names

            machine_type = request.form["machine_type"]

            air_temperature = float(
                request.form["air_temp"]
            )

            process_temperature = float(
                request.form["process_temp"]
            )

            rotational_speed = float(
                request.form["rotational_speed"]
            )

            torque = float(
                request.form["torque"]
            )

            tool_wear = float(
                request.form["tool_wear"]
            )

            if machine_type not in ["L", "M", "H"]:
                raise ValueError(
                    "Machine type must be L, M, or H."
                )

            numeric_values = [
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear
            ]

            if not all(
                pd.notna(value) for value in numeric_values
            ):
                raise ValueError(
                    "Please enter valid numeric values."
                )

            if not all(
                value >= 0 for value in numeric_values
            ):
                raise ValueError(
                    "Input values cannot be negative."
                )

            # Encode machine type

            input_data = pd.DataFrame(
                [[
                    air_temperature,
                    process_temperature,
                    rotational_speed,
                    torque,
                    tool_wear,
                    1 if machine_type == "L" else 0,
                    1 if machine_type == "M" else 0
                ]],
                columns=FEATURE_COLUMNS
            )

            # Predict failure

            result = int(model.predict(input_data)[0])

            if result == 1:
                prediction = "Machine Failure Predicted"
            else:
                prediction = "No Machine Failure Predicted"

            # Calculate failure probability if available

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    probability = round(
                        float(
                            probabilities[classes.index(1)] * 100
                        ),
                        2
                    )

                    if probability >= 70:
                        risk_category = "High"
                    elif probability >= 30:
                        risk_category = "Medium"
                    else:
                        risk_category = "Low"

        except (ValueError, KeyError) as exception:
            error = (
                "Please check your input values: "
                + str(exception)
            )

        except Exception as exception:
            error = "Prediction failed: " + str(exception)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        probability=probability,
        risk_category=risk_category
    )


# ==================================================
# 5. Batch prediction from CSV
# ==================================================

@app.route("/batch-predict", methods=["GET", "POST"])
def batch_predict():

    results = None
    error = None
    total_records = 0

    if request.method == "POST":

        try:
            if model is None:
                raise RuntimeError(
                    "The trained model could not be loaded."
                )

            # Validate uploaded file

            if "file" not in request.files:
                raise ValueError(
                    "No file was included in the upload."
                )

            file = request.files["file"]

            if file.filename == "":
                raise ValueError(
                    "Please select a CSV file."
                )

            if not file.filename.lower().endswith(".csv"):
                raise ValueError(
                    "Please upload a CSV file."
                )

            filename = secure_filename(file.filename)

            if not filename:
                filename = "uploaded_data.csv"

            upload_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(upload_path)

            # Read uploaded CSV

            df = pd.read_csv(upload_path)

            if df.empty:
                raise ValueError(
                    "The uploaded CSV file is empty."
                )

            # Support a Type column containing L, M, or H

            if "Type" in df.columns:

                df["Type"] = df["Type"].astype(str).str.strip()

                if not df["Type"].isin(["L", "M", "H"]).all():
                    raise ValueError(
                        "The Type column must contain L, M, or H."
                    )

                df["Type_L"] = (
                    df["Type"] == "L"
                ).astype(int)

                df["Type_M"] = (
                    df["Type"] == "M"
                ).astype(int)

            # Support machine_type as an alternative column name

            elif "machine_type" in df.columns:

                df["machine_type"] = (
                    df["machine_type"].astype(str).str.strip()
                )

                if not df["machine_type"].isin(
                    ["L", "M", "H"]
                ).all():
                    raise ValueError(
                        "The machine_type column must contain L, M, or H."
                    )

                df["Type_L"] = (
                    df["machine_type"] == "L"
                ).astype(int)

                df["Type_M"] = (
                    df["machine_type"] == "M"
                ).astype(int)

            # Check required feature columns

            missing_columns = [
                column
                for column in FEATURE_COLUMNS
                if column not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            # Prepare features in the expected order

            input_data = df[FEATURE_COLUMNS].copy()

            # Convert feature values to numbers

            for column in FEATURE_COLUMNS:
                input_data[column] = pd.to_numeric(
                    input_data[column],
                    errors="raise"
                )

            if input_data.isna().any().any():
                raise ValueError(
                    "The CSV contains missing feature values."
                )

            # Predict every record

            predictions = model.predict(input_data)

            df["Predicted Failure"] = predictions.astype(int)

            df["Prediction Result"] = df[
                "Predicted Failure"
            ].map({
                0: "No Machine Failure",
                1: "Machine Failure"
            })

            # Calculate failure probability if supported

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)
                classes = list(model.classes_)

                if 1 in classes:
                    failure_index = classes.index(1)

                    df["Failure Probability (%)"] = (
                        probabilities[:, failure_index] * 100
                    ).round(2)

            # Count all processed records

            total_records = len(df)

            # Save the COMPLETE result set for download

            results_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                "batch_prediction_results.csv"
            )

            df.to_csv(results_path, index=False)

            # IMPORTANT:
            # Pass a list of dictionaries, not an HTML string.
            # This matches results[0].keys() in the HTML template.

            results = df.head(100).to_dict(
                orient="records"
            )

            print(
                f"Batch prediction successful: {total_records} records"
            )

        except Exception as exception:
            error = str(exception)
            print("Batch prediction error:", error)

    # Keep this return OUTSIDE the try/except block.
    # This also allows the page to open before a file is uploaded.

    return render_template(
        "batch_prediction.html",
        results=results,
        error=error,
        total_records=total_records
    )


# ==================================================
# 6. Download batch prediction CSV
# ==================================================

@app.route("/download-results")
def download_results():

    results_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "batch_prediction_results.csv"
    )

    if not os.path.isfile(results_path):
        return (
            "No batch prediction results found. "
            "Upload a CSV file and run predictions first.",
            404
        )

    return send_file(
        results_path,
        as_attachment=True,
        download_name="batch_prediction_results.csv",
        mimetype="text/csv"
    )


# ==================================================
# 7. Model dashboard
# ==================================================

@app.route("/dashboard")
def dashboard():

    metrics = {
        "accuracy": "98.20%",
        "precision": "75.00%",
        "recall": "70.59%",
        "f1_score": "72.73%",
        "roc_auc": "96.38%",
        "pr_auc": "78.68%"
    }

    return render_template(
        "dashboard.html",
        metrics=metrics
    )


# ==================================================
# 8. Handle oversized CSV uploads
# ==================================================

@app.errorhandler(413)
def file_too_large(exception):

    return (
        "The uploaded file is too large. "
        "Maximum file size is 50 MB.",
        413
    )


# ==================================================
# 9. Start Flask application
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)
