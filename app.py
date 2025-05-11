import streamlit as st
import pickle
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize  

# Load all saved objects
with open('all_objects.pkl', 'rb') as f:
    all_objects = pickle.load(f)

model = all_objects['model']
X_train = all_objects['X_train']
X_test = all_objects['X_test']
y_test = all_objects['y_test']

# Extract real feature names from X_train
feature_names = X_train.columns.tolist()

# Build LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['Safe', 'Ransomware'],
    mode='classification'
)

# Streamlit Web Interface 
st.title("🛡️ Ransomware Detection Dashboard")
st.markdown("---")  

st.markdown("### 🔍 Make a Prediction")
st.sidebar.title("🛠️ Input Feature Values")
st.sidebar.markdown("Fill all feature inputs below for a live prediction.")

input_data = []

for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, format="%.5f")
    input_data.append(value)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Prediction: **{'Ransomware Detected' if prediction == 1 else 'Safe'}**")
    st.write(f"Confidence: **{probability:.2%}**")

    # Generate LIME Explanation
    explanation = explainer.explain_instance(
        data_row=input_array[0],
        predict_fn=model.predict_proba,
        num_features=10
    )
    st.markdown("---")
    st.markdown("## 🔎 LIME Explanation of the Prediction")

    st.subheader("🔎 LIME Local Explanation")
    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)

    # ROC Curve Section
    st.markdown("---")
    st.markdown("## 📈 Multiclass ROC-AUC Curve")

    from sklearn.preprocessing import label_binarize

# Binarize the output
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

# Predict probabilities
y_probs = model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
fig_roc, ax = plt.subplots(figsize=(8, 6))
for i in range(len(classes)):
    ax.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Multiclass ROC-AUC Curve')
ax.legend(loc="lower right")
st.pyplot(fig_roc)
