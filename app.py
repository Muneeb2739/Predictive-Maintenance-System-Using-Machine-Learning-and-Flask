
import os
import math
import joblib
import pandas as pd

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest


# ==================================================
# 1. Initialize Flask application
# ==================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "uploads"
)

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

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ==================================================
# 3. Load trained machine learning model
# ==================================================

model = None
model_error = None

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully:", type(model))

except Exception as exception:
    model_error = str(exception)
    print("Error loading model:", model_error)


# ==================================================
# 4. Helper functions
# ==================================================

def check_model():
    """Check whether the trained model is available."""

    if model is None:
        raise RuntimeError(
            "The trained model could not be loaded. "
            "Check models/best_model.pkl and your installed "
            "scikit-learn version."
        )


def validate_sensor_values(data):
    """Check that sensor values are numeric, finite and nonnegative."""

    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise"
        )

    if data[SENSOR_COLUMNS].isna().any().any():
        raise ValueError(
            "Sensor values cannot be missing."
        )

    if not data[SENSOR_COLUMNS].map(
        lambda value: math.isfinite(float(value))
    ).all().all():
        raise ValueError(
            "Sensor values must be finite numbers."
        )

    if (data[SENSOR_COLUMNS] < 0).any().any():
        raise ValueError(
            "Sensor values cannot be negative."
        )


def assign_risk_category(probability):
    """Convert failure probability (%) into an application risk category."""

    if probability < 30:
        return "Low"
    elif probability < 70:
        return "Medium"
    else:
        return "High"


def add_prediction_results(data, input_data):
    """Add predicted class, probability and risk category to a DataFrame."""

    predictions = model.predict(input_data)

    data["Predicted Failure"] = predictions.astype(int)

    data["Prediction Result"] = data[
        "Predicted Failure"
    ].map({
        0: "No Machine Failure",
        1: "Machine Failure"
    })

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)
        classes = list(model.classes_)

        if 1 in classes:
            failure_index = classes.index(1)

            failure_probability = (
                probabilities[:, failure_index] * 100
            )

            data["Failure Probability (%)"] = (
                failure_probability.round(2)
            )

            data["Risk Category"] = [
                assign_risk_category(float(probability))
                for probability in failure_probability
            ]

        else:
            data["Failure Probability (%)"] = None
            data["Risk Category"] = "Unavailable"

    else:
        data["Failure Probability (%)"] = None
        data["Risk Category"] = "Unavailable"

    return data


# ==================================================
# 5. Individual prediction
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None
    probability = None
    risk_category = None

    if request.method == "POST":

        try:
            check_model()

            # Read form values

            machine_type = request.form.get(
                "machine_type", ""
            ).strip().upper()

            if machine_type not in ["L", "M", "H"]:
                raise ValueError(
                    "Machine type must be L, M, or H."
                )

            form_values = {
                "Air temperature [K]":
                    request.form.get("air_temp", ""),

                "Process temperature [K]":
                    request.form.get("process_temp", ""),

                "Rotational speed [rpm]":
                    request.form.get("rotational_speed", ""),

                "Torque [Nm]":
                    request.form.get("torque", ""),

                "Tool wear [min]":
                    request.form.get("tool_wear", "")
            }

            input_data = pd.DataFrame([form_values])

            # Validate sensor values

            validate_sensor_values(input_data)

            # Encode machine type

            input_data["Type_L"] = (
                1 if machine_type == "L" else 0
            )

            input_data["Type_M"] = (
                1 if machine_type == "M" else 0
            )

            input_data = input_data[FEATURE_COLUMNS]

            # Predict failure

            result = int(model.predict(input_data)[0])

            if result == 1:
                prediction = "Machine Failure Predicted"
            else:
                prediction = "No Machine Failure Predicted"

            # Calculate failure probability

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)
                classes = list(model.classes_)

                if 1 in classes:
                    failure_index = classes.index(1)

                    probability = round(
                        float(
                            probabilities[0][failure_index] * 100
                        ),
                        2
                    )

                    risk_category = assign_risk_category(
                        probability
                    )

                else:
                    risk_category = "Unavailable"

            else:
                risk_category = "Unavailable"

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
# 6. Batch prediction from CSV
# ==================================================

@app.route("/batch-predict", methods=["GET", "POST"])
def batch_predict():

    results = None
    error = None
    total_records = 0

    if request.method == "POST":

        try:
            check_model()

            # Validate uploaded file

            if "file" not in request.files:
                raise ValueError(
                    "No file was included in the upload."
                )

            file = request.files["file"]

            if not file.filename:
                raise ValueError(
                    "Please select a CSV file."
                )

            if not file.filename.lower().endswith(".csv"):
                raise ValueError(
                    "Please upload a CSV file."
                )

            # Read CSV safely without relying on its original filename

            try:
                df = pd.read_csv(file)
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                UnicodeDecodeError
            ) as exception:
                raise ValueError(
                    "The uploaded file is empty or is not a valid CSV."
                ) from exception

            if df.empty:
                raise ValueError(
                    "The uploaded CSV file contains no records."
                )

            if len(df.columns) == 0:
                raise ValueError(
                    "The CSV file does not contain column headers."
                )

            # Remove accidental spaces from column names

            df.columns = df.columns.astype(str).str.strip()

            if df.columns.duplicated().any():
                raise ValueError(
                    "The CSV contains duplicate column names."
                )

            # Support a Type column containing L, M or H

            if "Type" in df.columns:

                df["Type"] = (
                    df["Type"].astype("string").str.strip().str.upper()
                )

                if df["Type"].isna().any() or not df[
                    "Type"
                ].isin(["L", "M", "H"]).all():
                    raise ValueError(
                        "The Type column must contain only L, M, or H."
                    )

                df["Type_L"] = (
                    df["Type"] == "L"
                ).astype(int)

                df["Type_M"] = (
                    df["Type"] == "M"
                ).astype(int)

            # Support machine_type as an alternative column

            elif "machine_type" in df.columns:

                df["machine_type"] = (
                    df["machine_type"]
                    .astype("string")
                    .str.strip()
                    .str.upper()
                )

                if df["machine_type"].isna().any() or not df[
                    "machine_type"
                ].isin(["L", "M", "H"]).all():
                    raise ValueError(
                        "The machine_type column must contain "
                        "only L, M, or H."
                    )

                df["Type_L"] = (
                    df["machine_type"] == "L"
                ).astype(int)

                df["Type_M"] = (
                    df["machine_type"] == "M"
                ).astype(int)

            # Check required features

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

            # Validate sensor values

            validate_sensor_values(input_data)

            # Validate encoded machine type columns

            for column in ["Type_L", "Type_M"]:
                input_data[column] = pd.to_numeric(
                    input_data[column],
                    errors="raise"
                )

            if input_data[["Type_L", "Type_M"]].isna().any().any():
                raise ValueError(
                    "Machine type encoding cannot contain missing values."
                )

            if not input_data[["Type_L", "Type_M"]].isin(
                [0, 1]
            ).all().all():
                raise ValueError(
                    "Type_L and Type_M must contain only 0 or 1."
                )

            if (
                input_data["Type_L"] + input_data["Type_M"] > 1
            ).any():
                raise ValueError(
                    "A record cannot have both Type_L and Type_M set to 1."
                )

            # Predict all records and add result columns

            df = add_prediction_results(
                df,
                input_data[FEATURE_COLUMNS]
            )

            total_records = len(df)

            # Save the complete results for download

            results_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                "batch_prediction_results.csv"
            )

            df.to_csv(results_path, index=False)

            # Display at most 100 records on the webpage

            results = df.head(100).to_dict(
                orient="records"
            )

            print(
                f"Batch prediction successful: {total_records} records"
            )

        except Exception as exception:
            error = str(exception)
            print("Batch prediction error:", error)

    return render_template(
        "batch_prediction.html",
        results=results,
        error=error,
        total_records=total_records
    )


# ==================================================
# 7. Download batch prediction CSV
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
# 8. Model dashboard
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
# 9. Handle oversized uploads
# ==================================================

@app.errorhandler(413)
def file_too_large(exception):

    return (
        "The uploaded file is too large. "
        "Maximum file size is 50 MB.",
        413
    )


# ==================================================
# 10. Start Flask application
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)
