import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Server Infrastructure Monitoring",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ Server Infrastructure Monitoring Dashboard")
st.markdown(
    "### Server Performance Analysis & Performance Issue Prediction"
)

st.sidebar.header("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard Overview",
        "Model Performance",
        "Feature Importance",
        "Predict Performance Issue",
        "Dataset Explorer"
    ]
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload server performance CSV",
    type=["csv"]
)

DEFAULT_FILE = "server_infrastructure_performance_5000.csv"

@st.cache_data
def load_data(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

if uploaded_file is not None:
    df = load_data(uploaded_file.getvalue())
else:
    try:
        with open(DEFAULT_FILE, "rb") as f:
            df = load_data(f.read())
    except FileNotFoundError:
        st.warning(
            "Please upload your server infrastructure CSV file "
            "using the sidebar."
        )
        st.stop()

# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------

TARGET = "Performance_Issue"

features = [
    "Server_Type",
    "Operating_System",
    "CPU_Usage",
    "Memory_Usage",
    "Disk_Usage",
    "Disk_IO",
    "Network_In_Mbps",
    "Network_Out_Mbps",
    "Network_Latency_ms",
    "Response_Time_ms",
    "Active_Connections",
    "Request_Count",
    "Error_Count",
    "Process_Count",
    "Uptime_Hours",
    "Temperature_C",
    "Hour"
]

required_columns = features + [TARGET]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    st.stop()

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Fill missing numerical values using median
numeric_cols = df[features].select_dtypes(
    include=np.number
).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values
categorical_cols = ["Server_Type", "Operating_System"]

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown").astype(str)

# Remove records with missing target
df = df.dropna(subset=[TARGET])

# Select features and target
X = df[features].copy()
y = df[TARGET].astype(int)

# Convert categorical variables to numeric
X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# TRAIN MODELS WITHOUT PICKLE FILES
# --------------------------------------------------

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    }

    results = []
    predictions = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, y_pred, zero_division=0
            )
        })

    return models, pd.DataFrame(results), predictions


with st.spinner("Training machine learning models..."):
    models, results_df, predictions = train_models(
        X_train, y_train, X_test, y_test
    )

# XGBoost is used for prediction, matching your notebook
selected_model = models["XGBoost"]

# --------------------------------------------------
# DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.header("📊 Server Infrastructure Overview")

    total_records = len(df)

    avg_cpu = df["CPU_Usage"].mean()
    avg_memory = df["Memory_Usage"].mean()
    avg_response = df["Response_Time_ms"].mean()

    total_issues = int((df[TARGET] == 1).sum())

    issue_rate = (
        total_issues / len(df) * 100
        if len(df) > 0 else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Avg CPU Usage", f"{avg_cpu:.2f}%")
    col3.metric("Avg Memory Usage", f"{avg_memory:.2f}%")
    col4.metric("Avg Response Time", f"{avg_response:.2f} ms")
    col5.metric("Performance Issue Rate", f"{issue_rate:.2f}%")

    st.divider()

    col1, col2 = st.columns(2)

    # Performance Issue Distribution
    with col1:
        st.subheader("Performance Issue Distribution")

        issue_counts = df[TARGET].value_counts().sort_index()

        issue_counts.index = [
            "Normal" if i == 0 else "Performance Issue"
            for i in issue_counts.index
        ]

        st.bar_chart(issue_counts)

    # CPU Usage by Server Type
    with col2:
        st.subheader("Average CPU Usage by Server Type")

        cpu_data = df.groupby(
            "Server_Type"
        )["CPU_Usage"].mean()

        st.bar_chart(cpu_data)

    col1, col2 = st.columns(2)

    # Memory Usage
    with col1:
        st.subheader("Average Memory Usage by Server Type")

        memory_data = df.groupby(
            "Server_Type"
        )["Memory_Usage"].mean()

        st.bar_chart(memory_data)

    # Response Time
    with col2:
        st.subheader("Average Response Time by Server Type")

        response_data = df.groupby(
            "Server_Type"
        )["Response_Time_ms"].mean()

        st.bar_chart(response_data)

    # Performance issues by operating system
    st.subheader("Performance Issues by Operating System")

    os_issues = df.groupby(
        "Operating_System"
    )[TARGET].sum()

    st.bar_chart(os_issues)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("🤖 Machine Learning Model Performance")

    st.write(
        "The following models are trained using the "
        "server infrastructure dataset."
    )

    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}"
        }),
        use_container_width=True
    )

    st.subheader("Model Comparison")

    st.bar_chart(
        results_df.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        ]
    )

    # Cross-validation
    st.subheader("5-Fold Stratified Cross-Validation")

    if st.button("Run Cross-Validation"):

        kfold = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        cv_results = []

        progress = st.progress(0)

        for i, (name, model) in enumerate(models.items()):

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=kfold,
                scoring="f1"
            )

            cv_results.append({
                "Model": name,
                "Mean F1 Score": scores.mean(),
                "Standard Deviation": scores.std()
            })

            progress.progress((i + 1) / len(models))

        cv_df = pd.DataFrame(cv_results)

        st.dataframe(
            cv_df.style.format({
                "Mean F1 Score": "{:.4f}",
                "Standard Deviation": "{:.4f}"
            }),
            use_container_width=True
        )

    # Confusion Matrix
    st.subheader("XGBoost Confusion Matrix")

    xgb_pred = predictions["XGBoost"]

    cm = confusion_matrix(
        y_test,
        xgb_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Performance Issue"],
        yticklabels=["Normal", "Performance Issue"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost Confusion Matrix")

    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        xgb_pred,
        target_names=["Normal", "Performance Issue"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

elif page == "Feature Importance":

    st.header("📌 Feature Importance Analysis")

    importance = selected_model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("Top 10 Important Features")

    top_features = feature_importance.head(10)

    st.dataframe(top_features, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 Features - XGBoost")

    plt.tight_layout()

    st.pyplot(fig)

    st.download_button(
        "Download Feature Importance CSV",
        feature_importance.to_csv(index=False),
        file_name="server_feature_importance.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# PREDICT PERFORMANCE ISSUE
# --------------------------------------------------

elif page == "Predict Performance Issue":

    st.header("🔍 Predict Server Performance Issue")

    st.write(
        "Select a server record from your dataset "
        "to predict its performance status."
    )

    # Select record
    row_number = st.number_input(
        "Select Record Number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    selected_row = df.iloc[[int(row_number)]]

    st.subheader("Selected Server Details")

    st.dataframe(
        selected_row[features],
        use_container_width=True
    )

    if st.button("Predict Performance"):

        # Apply same encoding as training data
        input_data = selected_row[features].copy()

        input_data = pd.get_dummies(
            input_data,
            columns=categorical_cols,
            drop_first=True
        )

        # Match training columns
        input_data = input_data.reindex(
            columns=X.columns,
            fill_value=0
        )

        prediction = selected_model.predict(input_data)[0]

        probability = selected_model.predict_proba(
            input_data
        )[0][1]

        if prediction == 1:
            st.error("⚠️ Predicted Status: Performance Issue")
        else:
            st.success("✅ Predicted Status: Normal")

        st.metric(
            "Predicted Performance Issue Probability",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "This probability is the model's estimated probability "
            "for the Performance Issue class, not a guarantee."
        )

# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.header("📁 Server Dataset Explorer")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    col1.write("Dataset Shape")
    col1.write(df.shape)

    col2.write("Missing Values")
    col2.dataframe(
        df.isnull().sum().to_frame("Missing Values")
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Download Dataset")

    st.download_button(
        "Download Processed Dataset",
        df.to_csv(index=False),
        file_name="server_performance_processed.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Server Infrastructure Monitoring Analysis | "
    "Machine Learning & Streamlit"
)