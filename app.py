import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f8;
}
.main {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #1f4e79;
}
.metric-box {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("📊 Machine Learning Model Evaluation Dashboard")
st.write("Comparison of multiple classification models using Accuracy, F1-score, and Confusion Matrix.")

# -----------------------------
# Model Performance Summary
# -----------------------------
data = {
    "Model": [
        "Logistic Regression", "Decision Tree", "Random Forest", "KNN",
        "SVM", "XGBoost", "Naive Bayes", "Gradient Boosting", "AdaBoost", "Extra Trees"
    ],
    "Accuracy": [0.56, 0.55, 0.61, 0.58, 0.55, 0.61, 0.40, 0.62, 0.59, 0.58],
    "Weighted F1-Score": [0.53, 0.53, 0.59, 0.55, 0.51, 0.60, 0.45, 0.60, 0.55, 0.56]
}

df = pd.DataFrame(data)

st.subheader("🔍 Model Comparison Table")
st.dataframe(df, use_container_width=True)

# -----------------------------
# Confusion Matrices
# -----------------------------
st.subheader("📌 Confusion Matrices by Model")

confusion_matrices = {
    "Logistic Regression": np.array([
        [65, 6, 2, 2, 0],
        [24, 30, 0, 0, 0],
        [15, 6, 2, 2, 0],
        [11, 7, 1, 6, 1],
        [2, 1, 0, 1, 0]
    ]),
    "Random Forest": np.array([
        [69, 4, 1, 1, 0],
        [18, 30, 3, 3, 0],
        [11, 6, 6, 2, 0],
        [10, 4, 2, 7, 3],
        [1, 1, 1, 0, 1]
    ]),
    "XGBoost": np.array([
        [68, 4, 2, 1, 0],
        [18, 30, 3, 3, 0],
        [10, 6, 6, 3, 0],
        [10, 5, 3, 8, 0],
        [1, 1, 1, 0, 1]
    ]),
    "Gradient Boosting": np.array([
        [66, 6, 2, 1, 0],
        [17, 32, 3, 2, 0],
        [9, 6, 7, 3, 0],
        [8, 6, 2, 9, 1],
        [1, 1, 1, 0, 1]
    ])
}

for model_name, cm in confusion_matrices.items():
    st.markdown(f"### 🔹 {model_name}")
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

# -----------------------------
# Best Model Highlight
# -----------------------------
st.subheader("🏆 Best Performing Model")

best_model = df.sort_values(by="Accuracy", ascending=False).iloc[0]

st.success(
    f"**{best_model['Model']}** achieved the highest accuracy of **{best_model['Accuracy']:.2f}** "
    f"with a weighted F1-score of **{best_model['Weighted F1-Score']:.2f}**."
)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)
