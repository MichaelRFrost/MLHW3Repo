import joblib
import os
import pandas as pd
import json
import numpy as np
from io import BytesIO, StringIO


def model_fn(model_dir):
    """Load the trained pipeline from the specified directory."""
    path = os.path.join(model_dir, "finalized_loan_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

    model = joblib.load(path)
    print("Model loaded successfully.")
    return model


def input_fn(request_body, request_content_type):
    print(f"Receiving data of type: {request_content_type}")

    if request_content_type == "application/x-npy":
        data = np.load(BytesIO(request_body), allow_pickle=True)
        return pd.DataFrame(data)

    elif request_content_type == "application/json":
        return pd.read_json(StringIO(request_body))

    elif request_content_type == "text/csv":
        return pd.read_csv(StringIO(request_body))

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_df, model):
    """Apply the trained pipeline to parsed input data."""
    print("Running prediction pipeline...")

    input_df = pd.DataFrame(input_df)
    prediction = model.predict(input_df)
    return prediction


def output_fn(prediction, content_type):
    """Format model output as JSON."""
    print("Formatting output...")
    res = prediction.tolist() if isinstance(prediction, (np.ndarray, np.generic)) else prediction
    return json.dumps(res), "application/json"
