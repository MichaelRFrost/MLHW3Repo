import warnings
import pandas as pd
import streamlit as st
import boto3
import sagemaker

from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

warnings.simplefilter("ignore")

st.set_page_config(page_title="Loan Default Prediction App", layout="wide")
st.title("Loan Default Prediction")
st.write("Enter borrower information below to predict whether a loan is likely to default.")

# Load sample data so the app can mirror the training feature structure
@st.cache_data
def load_sample_dataset():
    possible_paths = [
        "X_train.csv",
        "Portfolio/X_train.csv"
    ]
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
            return df
        except Exception:
            continue
    return pd.DataFrame()

sample_dataset = load_sample_dataset()

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    boto_session = boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )
    return boto_session, sagemaker.Session(boto_session=boto_session)

boto_session, sm_session = get_session(aws_id, aws_secret, aws_token)

predictor = Predictor(
    endpoint_name=aws_endpoint,
    sagemaker_session=sm_session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

st.subheader("Borrower Inputs")

if sample_dataset.empty:
    st.warning("Sample feature file not found. Add X_train.csv to the app folder for fully dynamic inputs.")
    manual_defaults = {
        "loan_amnt": 10000.0,
        "int_rate": 12.0,
        "annual_inc": 60000.0,
        "dti": 15.0,
        "installment": 300.0,
        "fico_range_low": 680.0
    }
    input_row = {}
    cols = st.columns(2)
    for i, (feature, default_value) in enumerate(manual_defaults.items()):
        with cols[i % 2]:
            input_row[feature] = st.number_input(feature, value=float(default_value))
else:
    base_row = sample_dataset.iloc[0].copy()
    input_row = {}
    cols = st.columns(2)

    for i, feature in enumerate(sample_dataset.columns):
        value = base_row[feature]

        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(sample_dataset[feature]):
                default_value = float(value) if pd.notna(value) else 0.0
                input_row[feature] = st.number_input(
                    feature,
                    value=default_value
                )
            else:
                text_value = "" if pd.isna(value) else str(value)
                input_row[feature] = st.text_input(
                    feature,
                    value=text_value
                )

if st.button("Run Prediction"):
    input_df = pd.DataFrame([input_row])

    try:
        prediction = predictor.predict(input_df.to_dict(orient="records"))
        pred_value = prediction[0] if isinstance(prediction, list) else prediction

        if int(pred_value) == 1:
            st.error("Prediction: Likely Default")
        else:
            st.success("Prediction: Likely Fully Paid")

        st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
