
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("Attrition.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Attrition.csv was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()


# ==========================================
# 3. PAGE HEADER
# ==========================================

st.title("Employee Attrition Prediction")
st.write(
    "An interactive machine learning application "
    "for predicting employee attrition."
)

st.divider()


# ==========================================
# 4. DEFINE FEATURES AND TARGET
# ==========================================

numeric_features = [
    "Age",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "YearsAtCompany"
]

categorical_features = [
    "Department",
    "Gender",
    "JobRole",
    "MaritalStatus"
]

target = "Attrition"

required_columns = (
    numeric_features + categorical_features + [target]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Your dataset is missing these columns: "
        f"{', '.join(missing_columns)}"
    )
    st.stop()


# Keep only the columns required for this model.
data = df[required_columns].copy()

# Normalize target labels.
data[target] = (
    data[target]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Remove rows without a valid target.
data = data[
    data[target].isin(["yes", "no"])
].copy()

# Convert target to binary values.
data[target] = data[target].map({
    "yes": 1,
    "no": 0
})

# Convert numeric features to numeric data types.
for col in numeric_features:
    data[col] = pd.to_numeric(
        data[col], errors="coerce"
    )

# Handle missing values in categorical columns.
for col in categorical_features:
    data[col] = data[col].replace(
        r"^\s*$", np.nan, regex=True
    )

if data.empty:
    st.error(
        "No valid employee records were found. "
        "Check the Attrition column in your dataset."
    )
    st.stop()

if data[target].nunique() < 2:
    st.error(
        "The dataset must contain both Yes and No "
        "values in the Attrition column."
    )
    st.stop()


X = data[numeric_features + categorical_features]
y = data[target].astype(int)


# ==========================================
# 5. DISPLAY DATASET OVERVIEW
# ==========================================

total_employees = len(data)
employees_left = int(y.sum())
employees_stayed = total_employees - employees_left

attrition_rate = (
    employees_left / total_employees * 100
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Employees",
    f"{total_employees:,}"
)

col2.metric(
    "Employees Who Left",
    f"{employees_left:,}"
)

col3.metric(
    "Employees Who Stayed",
    f"{employees_stayed:,}"
)

col4.metric(
    "Attrition Rate",
    f"{attrition_rate:.2f}%"
)

st.divider()


# ==========================================
# 6. MODEL SELECTION
# ==========================================

st.subheader("Machine Learning Model")

model_choice = st.selectbox(
    "Select a model",
    ["Random Forest", "Logistic Regression"]
)


# ==========================================
# 7. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 8. SELECT MODEL
# ==========================================

if model_choice == "Random Forest":
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
else:
    classifier = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=2000
    )


# ==========================================
# 9. TRAIN THE MODEL
# ==========================================

@st.cache_resource
def train_model(_X, _y, selected_model):
    # Split before fitting preprocessing to prevent
    # information leakage from the test set.
    X_train, X_test, y_train, y_test = (
        train_test_split(
            _X,
            _y,
            test_size=0.2,
            random_state=42,
            stratify=_y
        )
    )

    if selected_model == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    else:
        model = LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=2000
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Confusion Matrix": confusion_matrix(
            y_test, predictions
        ),
        "Classification Report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Stayed", "Left"],
            zero_division=0
        )
    }

    return pipeline, metrics


try:
    with st.spinner("Training the machine learning model..."):
        model, metrics = train_model(
            X,
            y,
            model_choice
        )

    st.success(
        f"{model_choice} model trained successfully."
    )

except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()


# ==========================================
# 10. MODEL PERFORMANCE
# ==========================================

st.subheader("Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{metrics['Accuracy']:.2%}"
)

m2.metric(
    "Precision",
    f"{metrics['Precision']:.2%}"
)

m3.metric(
    "Recall",
    f"{metrics['Recall']:.2%}"
)

m4.metric(
    "F1 Score",
    f"{metrics['F1 Score']:.2%}"
)

with st.expander("View Confusion Matrix"):
    matrix = pd.DataFrame(
        metrics["Confusion Matrix"],
        index=["Actual Stayed", "Actual Left"],
        columns=["Predicted Stayed", "Predicted Left"]
    )

    st.dataframe(matrix)

with st.expander("View Classification Report"):
    st.text(metrics["Classification Report"])


# ==========================================
# 11. EMPLOYEE ATTRITION PREDICTION
# ==========================================

st.divider()

st.header("Predict Employee Attrition")

st.write(
    "Enter the employee's details below to estimate "
    "the likelihood of attrition."
)

with st.form("prediction_form"):

    st.subheader("Employee Information")

    input_values = {}

    col1, col2 = st.columns(2)

    # Numeric input fields.
    for index, feature in enumerate(numeric_features):

        column = col1 if index % 2 == 0 else col2

        valid_values = data[feature].dropna()

        if valid_values.empty:
            default_value = 0.0
        else:
            default_value = float(
                valid_values.median()
            )

        if feature in [
            "Age",
            "DistanceFromHome",
            "Education",
            "EnvironmentSatisfaction",
            "JobInvolvement",
            "JobLevel",
            "JobSatisfaction",
            "YearsAtCompany"
        ]:
            input_values[feature] = column.number_input(
                feature,
                min_value=0,
                max_value=100,
                value=int(default_value),
                step=1
            )
        else:
            input_values[feature] = column.number_input(
                feature,
                min_value=0.0,
                value=max(0.0, default_value),
                step=1.0
            )

    st.subheader("Categorical Information")

    col3, col4 = st.columns(2)

    for index, feature in enumerate(categorical_features):

        column = col3 if index % 2 == 0 else col4

        options = sorted(
            data[feature]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not options:
            st.error(
                f"No valid values found for {feature}."
            )
            st.stop()

        input_values[feature] = column.selectbox(
            feature,
            options
        )

    submitted = st.form_submit_button(
        "Predict Attrition",
        use_container_width=True
    )


# ==========================================
# 12. DISPLAY PREDICTION
# ==========================================

if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=numeric_features + categorical_features
    )

    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.named_steps["classifier"].classes_)

    probability_left = float(
        probabilities[classes.index(1)]
    )

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.error(
                "Prediction: Employee Likely to Leave"
            )
        else:
            st.success(
                "Prediction: Employee Likely to Stay"
            )

    with result_col2:
        st.metric(
            "Estimated Attrition Probability",
            f"{probability_left:.2%}"
        )

    st.progress(probability_left)

    if prediction == 1:
        st.warning(
            "The model classified this employee as "
            "likely to leave. Consider reviewing "
            "workplace conditions, job satisfaction, "
            "and other relevant factors."
        )
    else:
        st.info(
            "The model classified this employee as "
            "likely to stay. This prediction does not "
            "guarantee that the employee will remain."
        )

    st.caption(
        "This is a machine learning estimate, not a "
        "guarantee of future employee behaviour. "
        "Use predictions alongside human judgment."
    )


# ==========================================
# 13. DATA PREVIEW
# ==========================================

st.divider()

with st.expander("Preview Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

    st.write("Dataset dimensions:", df.shape)

st.caption(
    "Employee Attrition Prediction | "
    "Machine Learning Project"
)
