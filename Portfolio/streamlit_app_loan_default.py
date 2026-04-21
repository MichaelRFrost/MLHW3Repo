import os
import sys
import tempfile
import warnings
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Optional path setup so local src modules remain importable if present.
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


# -----------------------------------------------------------------------------
# Custom transformer copied into the app so joblib artifacts saved from the
# notebook can be loaded successfully during Streamlit execution.
# -----------------------------------------------------------------------------
class LendingClubCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans selected LendingClub variables and creates engineered features.
    This matches the transformer used in the Milestone 4 notebook.
    """

    def __init__(self, rare_cutoff: float = 0.01):
        self.rare_cutoff = rare_cutoff
        self.numeric_output_columns_ = None
        self.categorical_output_columns_ = None
        self.keep_levels_ = {}

    def _emp_to_num(self, s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        s = s.str.replace("10+ years", "10", regex=False)
        s = s.str.replace("< 1 year", "0", regex=False)
        s = s.str.extract(r"(\d+)")[0]
        return pd.to_numeric(s, errors="coerce")

    def _percent_to_num(self, s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")

    def _term_to_num(self, s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    def _year_from_credit_line(self, s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, format="%b-%Y", errors="coerce").dt.year

    def fit(self, X: pd.DataFrame, y=None):
        Xc = self.transform(X, fit_mode=True)
        cat_cols = Xc.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            freq = Xc[col].fillna("Missing").astype(str).value_counts(normalize=True)
            keep_levels = freq[freq >= self.rare_cutoff].index.tolist()
            self.keep_levels_[col] = set(keep_levels)

        Xc = self._collapse_rare_levels(Xc)
        self.numeric_output_columns_ = Xc.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_output_columns_ = Xc.select_dtypes(exclude=[np.number]).columns.tolist()
        return self

    def _collapse_rare_levels(self, Xc: pd.DataFrame) -> pd.DataFrame:
        Xc = Xc.copy()
        for col, keep_levels in self.keep_levels_.items():
            Xc[col] = Xc[col].fillna("Missing").astype(str)
            Xc[col] = np.where(Xc[col].isin(keep_levels), Xc[col], "Other")
        return Xc

    def transform(self, X: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        Xc = X.copy()

        Xc["term_num"] = self._term_to_num(Xc["term"])
        Xc["int_rate_num"] = self._percent_to_num(Xc["int_rate"])
        Xc["emp_length_num"] = self._emp_to_num(Xc["emp_length"])
        Xc["revol_util_num"] = self._percent_to_num(Xc["revol_util"])

        Xc["earliest_cr_year"] = self._year_from_credit_line(Xc["earliest_cr_line"])
        issue_year = pd.to_datetime(Xc["issue_d"], format="%b-%Y", errors="coerce").dt.year
        Xc["credit_history_length"] = issue_year - Xc["earliest_cr_year"]

        Xc["fico_avg"] = (Xc["fico_range_low"] + Xc["fico_range_high"]) / 2.0
        Xc["loan_to_income"] = Xc["loan_amnt"] / (Xc["annual_inc"] + 1)
        Xc["installment_to_income"] = Xc["installment"] / (Xc["annual_inc"] + 1)
        Xc["revol_bal_to_income"] = Xc["revol_bal"] / (Xc["annual_inc"] + 1)
        Xc["inq_per_open_acc"] = Xc["inq_last_6mths"] / (Xc["open_acc"] + 1)
        Xc["delinq_per_total_acc"] = Xc["delinq_2yrs"] / (Xc["total_acc"] + 1)
        Xc["pub_rec_per_total_acc"] = Xc["pub_rec"] / (Xc["total_acc"] + 1)
        Xc["log_annual_inc"] = np.log1p(Xc["annual_inc"].clip(lower=0))
        Xc["log_revol_bal"] = np.log1p(Xc["revol_bal"].clip(lower=0))

        drop_cols = [
            "loan_status",
            "term",
            "int_rate",
            "emp_length",
            "revol_util",
            "earliest_cr_line",
            "issue_d",
        ]
        Xc = Xc.drop(columns=[c for c in drop_cols if c in Xc.columns], errors="ignore")
        Xc = Xc.replace([np.inf, -np.inf], np.nan)

        if not fit_mode and self.keep_levels_:
            Xc = self._collapse_rare_levels(Xc)

        return Xc


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
APP_TITLE = "Loan Default Risk Prediction App"
LOCAL_ARTIFACTS = "final_loan_default_model_artifacts.joblib"
LOCAL_EXPLAINER = "final_loan_default_shap_explainer.joblib"

STATE_OPTIONS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
    "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
    "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
    "WI", "WV", "WY"
]

PURPOSE_OPTIONS = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business",
    "car", "medical", "moving", "vacation", "house", "wedding", "renewable_energy", "other"
]

HOME_OWNERSHIP_OPTIONS = ["RENT", "MORTGAGE", "OWN", "ANY"]
VERIFICATION_OPTIONS = ["Verified", "Source Verified", "Not Verified"]
APPLICATION_TYPE_OPTIONS = ["Individual", "Joint App"]
GRADE_OPTIONS = list("ABCDEFG")
SUB_GRADE_OPTIONS = [f"{g}{i}" for g in GRADE_OPTIONS for i in range(1, 6)]
TERM_OPTIONS = ["36 months", "60 months"]
EMP_LENGTH_OPTIONS = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years",
    "7 years", "8 years", "9 years", "10+ years"
]
ISSUE_MONTH_OPTIONS = [
    "Jan-2018", "Feb-2018", "Mar-2018", "Apr-2018", "May-2018", "Jun-2018",
    "Jul-2018", "Aug-2018", "Sep-2018", "Oct-2018", "Nov-2018", "Dec-2018"
]
EARLIEST_CR_LINE_OPTIONS = [f"Jan-{year}" for year in range(1965, 2019)]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def maybe_download_from_s3(filename: str, secret_key_name: str) -> str:
    """Download a file from S3 if it is not available locally and Streamlit secrets exist."""
    local_path = os.path.join(current_dir, filename)
    if os.path.exists(local_path):
        return local_path

    try:
        import boto3  # imported lazily to keep local app simple when not needed
    except Exception:
        return local_path

    if "aws_credentials" not in st.secrets:
        return local_path

    creds = st.secrets["aws_credentials"]
    required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_BUCKET", secret_key_name]
    if not all(k in creds for k in required):
        return local_path

    temp_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(temp_path):
        return temp_path

    session = boto3.Session(
        aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=creds["AWS_SESSION_TOKEN"],
        region_name=creds.get("AWS_REGION", "us-east-1"),
    )
    s3_client = session.client("s3")
    s3_client.download_file(creds["AWS_BUCKET"], creds[secret_key_name], temp_path)
    return temp_path


@st.cache_resource(show_spinner=False)
def load_artifacts():
    artifacts_path = maybe_download_from_s3(LOCAL_ARTIFACTS, "AWS_LOAN_ARTIFACTS_KEY")
    explainer_path = maybe_download_from_s3(LOCAL_EXPLAINER, "AWS_LOAN_EXPLAINER_KEY")

    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(
            "Could not find final_loan_default_model_artifacts.joblib locally or via Streamlit secrets."
        )

    artifacts = joblib.load(artifacts_path)
    explainer = joblib.load(explainer_path) if os.path.exists(explainer_path) else None
    return artifacts, explainer


def build_input_row(user_inputs: Dict[str, object]) -> pd.DataFrame:
    row = {
        "loan_amnt": float(user_inputs["loan_amnt"]),
        "term": str(user_inputs["term"]),
        "int_rate": f"{float(user_inputs['int_rate']):.2f}%",
        "installment": float(user_inputs["installment"]),
        "grade": str(user_inputs["grade"]),
        "sub_grade": str(user_inputs["sub_grade"]),
        "emp_length": str(user_inputs["emp_length"]),
        "home_ownership": str(user_inputs["home_ownership"]),
        "annual_inc": float(user_inputs["annual_inc"]),
        "verification_status": str(user_inputs["verification_status"]),
        "purpose": str(user_inputs["purpose"]),
        "addr_state": str(user_inputs["addr_state"]),
        "dti": float(user_inputs["dti"]),
        "delinq_2yrs": float(user_inputs["delinq_2yrs"]),
        "earliest_cr_line": str(user_inputs["earliest_cr_line"]),
        "fico_range_low": float(user_inputs["fico_range_low"]),
        "fico_range_high": float(user_inputs["fico_range_high"]),
        "inq_last_6mths": float(user_inputs["inq_last_6mths"]),
        "open_acc": float(user_inputs["open_acc"]),
        "pub_rec": float(user_inputs["pub_rec"]),
        "revol_bal": float(user_inputs["revol_bal"]),
        "revol_util": f"{float(user_inputs['revol_util']):.2f}%",
        "total_acc": float(user_inputs["total_acc"]),
        "application_type": str(user_inputs["application_type"]),
        "mort_acc": float(user_inputs["mort_acc"]),
        "pub_rec_bankruptcies": float(user_inputs["pub_rec_bankruptcies"]),
        "issue_d": str(user_inputs["issue_d"]),
    }
    return pd.DataFrame([row])


def get_probability_and_prediction(input_df: pd.DataFrame, artifacts: dict) -> Tuple[float, int]:
    pipeline = artifacts["model_pipeline"]
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(input_df)[:, 1][0])
    else:
        probability = float(pipeline.predict(input_df)[0])
    pred_class = int(probability >= 0.50)
    return probability, pred_class


def risk_bucket(probability: float) -> Tuple[str, str]:
    if probability < 0.20:
        return "Low Risk", "Proceed through normal underwriting."
    if probability < 0.45:
        return "Moderate Risk", "Consider manual review or adjusted pricing."
    return "High Risk", "Recommend heightened review and tighter approval standards."


def compute_local_shap(input_df: pd.DataFrame, artifacts: dict, explainer):
    pipeline = artifacts["model_pipeline"]
    selected_feature_names = artifacts["selected_feature_names"]

    fitted_cleaner = pipeline.named_steps["cleaner"]
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_selector = pipeline.named_steps["selector"]
    final_model = pipeline.named_steps["model"]

    cleaned = fitted_cleaner.transform(input_df)
    transformed = fitted_preprocessor.transform(cleaned)
    selected = fitted_selector.transform(transformed)
    selected_df = pd.DataFrame(selected, columns=selected_feature_names)

    if explainer is None:
        explainer = shap.Explainer(final_model, selected_df)

    shap_values = explainer(selected_df)
    return shap_values, selected_df


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🏦 Loan Default Risk Prediction")
st.write(
    "This app estimates the probability that a borrower will default on a loan using the final "
    "Milestone 4 model. Enter a borrower profile below to generate a risk score and a local SHAP explanation."
)

with st.expander("What the app is doing"):
    st.markdown(
        "- Uses the final saved loan-default model artifacts from the notebook  \n"
        "- Predicts a default probability for one applicant  \n"
        "- Places the borrower into a practical risk tier  \n"
        "- Shows a SHAP waterfall chart for local explainability"
    )

try:
    artifacts, saved_explainer = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.form("loan_form"):
    st.subheader("Borrower and loan inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = st.number_input("Loan Amount", min_value=500.0, max_value=100000.0, value=15000.0, step=500.0)
        term = st.selectbox("Term", TERM_OPTIONS, index=0)
        int_rate = st.number_input("Interest Rate (%)", min_value=0.10, max_value=40.00, value=12.50, step=0.10)
        installment = st.number_input("Installment", min_value=10.0, max_value=5000.0, value=450.0, step=10.0)
        grade = st.selectbox("Grade", GRADE_OPTIONS, index=2)
        sub_grade = st.selectbox("Sub Grade", SUB_GRADE_OPTIONS, index=12)
        purpose = st.selectbox("Purpose", PURPOSE_OPTIONS, index=0)
        application_type = st.selectbox("Application Type", APPLICATION_TYPE_OPTIONS, index=0)
        issue_d = st.selectbox("Issue Date", ISSUE_MONTH_OPTIONS, index=11)

    with col2:
        emp_length = st.selectbox("Employment Length", EMP_LENGTH_OPTIONS, index=5)
        home_ownership = st.selectbox("Home Ownership", HOME_OWNERSHIP_OPTIONS, index=0)
        annual_inc = st.number_input("Annual Income", min_value=1000.0, max_value=1000000.0, value=75000.0, step=1000.0)
        verification_status = st.selectbox("Verification Status", VERIFICATION_OPTIONS, index=1)
        addr_state = st.selectbox("State", STATE_OPTIONS, index=43 if "TX" in STATE_OPTIONS else 0)
        dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=60.0, value=18.0, step=0.1)
        earliest_cr_line = st.selectbox("Earliest Credit Line", EARLIEST_CR_LINE_OPTIONS, index=len(EARLIEST_CR_LINE_OPTIONS) - 15)
        fico_range_low = st.number_input("FICO Range Low", min_value=300.0, max_value=850.0, value=680.0, step=1.0)
        fico_range_high = st.number_input("FICO Range High", min_value=300.0, max_value=850.0, value=684.0, step=1.0)

    with col3:
        delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
        inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        open_acc = st.number_input("Open Accounts", min_value=0.0, max_value=80.0, value=10.0, step=1.0)
        pub_rec = st.number_input("Public Records", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
        revol_bal = st.number_input("Revolving Balance", min_value=0.0, max_value=500000.0, value=12000.0, step=500.0)
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=42.0, step=0.1)
        total_acc = st.number_input("Total Accounts", min_value=1.0, max_value=150.0, value=24.0, step=1.0)
        mort_acc = st.number_input("Mortgage Accounts", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", min_value=0.0, max_value=10.0, value=0.0, step=1.0)

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    if fico_range_high < fico_range_low:
        st.error("FICO Range High must be greater than or equal to FICO Range Low.")
        st.stop()

    user_inputs = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "addr_state": addr_state,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "earliest_cr_line": earliest_cr_line,
        "fico_range_low": fico_range_low,
        "fico_range_high": fico_range_high,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "application_type": application_type,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "issue_d": issue_d,
    }

    input_df = build_input_row(user_inputs)
    probability, pred_class = get_probability_and_prediction(input_df, artifacts)
    bucket, recommendation = risk_bucket(probability)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted Default Probability", f"{probability:.1%}")
    metric_col2.metric("Predicted Class", "Default" if pred_class == 1 else "Non-Default")
    metric_col3.metric("Risk Tier", bucket)

    st.info(recommendation)

    with st.expander("Submitted input row", expanded=False):
        st.dataframe(input_df, use_container_width=True)

    try:
        shap_values, transformed_df = compute_local_shap(input_df, artifacts, saved_explainer)
        st.subheader("Local explanation for this borrower")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        st.pyplot(fig, clear_figure=True)

        top_feature = pd.Series(
            shap_values[0].values,
            index=shap_values[0].feature_names
        ).abs().idxmax()
        st.success(f"Most influential feature for this prediction: {top_feature}")

        top_contributions = pd.DataFrame({
            "Feature": shap_values[0].feature_names,
            "SHAP Value": shap_values[0].values
        })
        top_contributions["Abs SHAP"] = top_contributions["SHAP Value"].abs()
        top_contributions = top_contributions.sort_values("Abs SHAP", ascending=False).drop(columns="Abs SHAP")

        st.subheader("Top feature contributions")
        st.dataframe(top_contributions.head(10), use_container_width=True)
    except Exception as exc:
        st.warning(f"Prediction succeeded, but SHAP could not be displayed: {exc}")

st.caption(
    "Deployment note: place final_loan_default_model_artifacts.joblib and "
    "final_loan_default_shap_explainer.joblib alongside this app file, or provide S3 keys in Streamlit secrets."
)
