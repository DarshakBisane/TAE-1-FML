
import streamlit as st
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Hyperparameter Tuning",
    page_icon="⚙️",
    layout="wide"
)


# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

with open("results.json", "r") as file:
    results = json.load(file)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚙️ Hyperparameter Tuning for Classification Models")

st.write(
    "This dashboard compares classification model performance "
    "before and after hyperparameter tuning."
)

st.divider()


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.header("📊 Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Dataset", "Heart Disease")
col2.metric("Models", len(results))
col3.metric("Tuning Method", "GridSearchCV")


st.divider()


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.header("🤖 Model Performance")

performance_data = []

for model_name, data in results.items():

    before = data["before"]
    after = data["after"]

    performance_data.append({
        "Model": model_name,
        "Accuracy Before": before["accuracy"],
        "Accuracy After": after["accuracy"],
        "Precision Before": before["precision"],
        "Precision After": after["precision"],
        "Recall Before": before["recall"],
        "Recall After": after["recall"],
        "F1 Before": before["f1_score"],
        "F1 After": after["f1_score"]
    })


performance_df = pd.DataFrame(performance_data)


# Show simple table

display_df = performance_df.copy()

for column in display_df.columns:
    if column != "Model":
        display_df[column] = (
            display_df[column] * 100
        ).round(2).astype(str) + "%"


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# ACCURACY COMPARISON
# --------------------------------------------------

st.header("📈 Accuracy Before vs After Tuning")

models = performance_df["Model"]

before_accuracy = performance_df["Accuracy Before"]
after_accuracy = performance_df["Accuracy After"]

fig, ax = plt.subplots(figsize=(5, 3))

x = range(len(models))
width = 0.35

ax.bar(
    [i - width / 2 for i in x],
    before_accuracy,
    width,
    label="Before Tuning"
)

ax.bar(
    [i + width / 2 for i in x],
    after_accuracy,
    width,
    label="After Tuning"
)

ax.set_xticks(list(x))
ax.set_xticklabels(models)

ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)

ax.set_title("Model Accuracy Comparison")

ax.legend()

plt.tight_layout()

chart_col_left, chart_col, chart_col_right = st.columns([1, 2, 1])
with chart_col:
    st.pyplot(fig, use_container_width=False)


st.divider()


# --------------------------------------------------
# BEST PARAMETERS
# --------------------------------------------------

st.header("⚙️ Best Hyperparameters")

selected_model = st.selectbox(
    "Select Model",
    list(results.keys())
)


parameters = results[selected_model]["best_parameters"]


for parameter, value in parameters.items():

    st.write(
        f"**{parameter}:** `{value}`"
    )


st.divider()


# --------------------------------------------------
# IMPROVEMENT
# --------------------------------------------------

st.header("📌 Performance Improvement")

for model_name, data in results.items():

    before = data["before"]["accuracy"]
    after = data["after"]["accuracy"]

    improvement = (after - before) * 100

    st.write(
        f"**{model_name}** → "
        f"{improvement:+.2f}% accuracy change"
    )


st.divider()

st.caption(
    "Hyperparameter Tuning for Classification Models"
)

