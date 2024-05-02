import streamlit as st
import pandas as pd
import numpy as np
import pickle

import subprocess
import sys

def install_packages(packages):
    for package in packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} not found, installing now...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                __import__(package)
                print(f"{package} has been successfully installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}. Error: {e}")

# List of packages to install
packages_to_install = ['joblib', 'matplotlib']

# Installing the packages
install_packages(packages_to_install)


from joblib import load
import matplotlib.pyplot as plt


st.title('Credit Risk Assessment Tool')

# Load the dataset
df = pd.read_csv("heloc_dataset_v1.csv")
df_pm = pd.read_csv("model_performance_metrics.csv")
df_pm.set_index('Models', inplace=True)

# Load the models
# best_RF_model = load("best_RF_model.joblib")
# best_Boosting_model = load("best_Boosting_model.joblib")
# best_LDA_model = load("best_LDA_model.joblib")

with open('best_RF_model.joblib', 'rb') as f:
    best_RF_model = pickle.load(f)
    
with open('best_Boosting_model.joblib', 'rb') as f:
    best_Boosting_model = pickle.load(f)
    
with open('best_LDA_model.joblib', 'rb') as f:
    best_LDA_model = pickle.load(f)

models = {
    'AdaBoost': best_Boosting_model,
    'LDA': best_LDA_model,
    'Random Forest': best_RF_model
}

# Provide options to select certain model
model_choice = st.selectbox("Choose a model to make predictions:", ("Random Forest", "AdaBoost", "LDA"))

# Get variable ranges
values = df.describe().loc["max"].apply(int)
variable_ranges = {col: (0, max_val) for col, max_val in values.items()}
special_ranges = {
    "MSinceMostRecentDelq=-7": (0, 1),
    "MSinceMostRecentInqexcl7days=-7": (0, 1),
    "MSinceOldestTradeOpen=-8": (0, 1),
    "MSinceMostRecentDelq=-8": (0, 1),
    "MSinceMostRecentInqexcl7days=-8": (0, 1),
    "NetFractionRevolvingBurden=-8": (0, 1),
    "NetFractionInstallBurden=-8": (0, 1),
    "NumRevolvingTradesWBalance=-8": (0, 1),
    "NumInstallTradesWBalance=-8": (0, 1),
    "NumBank2NatlTradesWHighUtilization=-8": (0, 1),
    "PercentTradesWBalance=-8": (0, 1)
}
variable_ranges.update(special_ranges)
feature_names = list(variable_ranges.keys())

# Prediction funtion
def predict_risk(model_choice, **inputs):
    feature_order = list(variable_ranges.keys())
    input_data = np.array([[inputs[feature] for feature in feature_order]])

    # Select the model based on user choice
    if model_choice == "AdaBoost":
        model = best_Boosting_model
        prediction = model.predict(np.array(input_data).reshape(1, -1))
    elif model_choice == "LDA":
        model = best_LDA_model
        prediction = model.predict(input_data)
    elif model_choice == "Random Forest":
        model = best_RF_model
        prediction = model.predict(np.array(input_data).reshape(1, -1))
    return prediction

# Creating a sidebar for input fields
with st.sidebar:
    st.header("Input Predictors for Risk Assessment")

    inputs = {}
    for feature, (min_val, max_val) in variable_ranges.items():
        inputs[feature] = st.slider(feature, min_value=min_val, max_value=max_val, value=(min_val + max_val) // 2, key=feature)

# Button to perform prediction
submit_button = st.button("Get Prediction")

# Initialize session state for storing predictions if not already set
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'unique_counter' not in st.session_state:
    st.session_state.unique_counter = 0  # This counter will continuously increment

def add_prediction_to_history(prediction, features):
    # Increment the unique counter
    st.session_state.unique_counter += 1
    # Generate a unique key using the counter
    new_id = st.session_state.unique_counter
    # Store prediction with its features and a unique ID
    st.session_state.prediction_history.append((new_id, result, features))
    # Keep only the last 10 entries
    st.session_state.prediction_history = st.session_state.prediction_history[-10:]

risk = ''
result = ''
prediction = predict_risk(model_choice, **inputs)
if submit_button:
    if prediction == 1:
        risk = "High Risk"
        result = "No"
    else:
        risk = "Low Risk"
        result = "Yes"
    add_prediction_to_history(prediction, inputs)

if risk:
    message = ''
    st.markdown(f"The corresponding Risk Performance is: **{risk}**")
    if prediction == 1:
        message = st.markdown("**Bad** Performance, one should **NOT** consider **APPROVE** the application")
    else:
        message = st.markdown("**Good** Performance, one should consider **APPROVE** the application")
else:
    st.markdown("**Prediction:**")
    st.write("Waiting for your input...")

tab1, tab2 = st.tabs(["Model Performance", "Feature Importances"])
with tab1:
    # Show model metrics on the main page
    if model_choice:
        st.subheader(f"Metrics for {model_choice}")
        metrics_to_show = df_pm.loc[[model_choice], ['Accuracy', 'Recall', 'Precision']]
        # Format metrics for display
        metrics_to_show = metrics_to_show.style.format({
            'Accuracy': "{:.2f}",
            'Recall': "{:.2f}",
            'Precision': "{:.2f}"
        })
        # Use st.write to display styled DataFrame
        st.write(metrics_to_show)

        accuracy = df_pm.loc[model_choice, 'Accuracy']
        recall = df_pm.loc[model_choice, 'Recall']
        precision = df_pm.loc[model_choice, 'Precision']

        # Detailed interpretation of metrics
        # Use HTML completely for formatting and styling
        st.markdown(f"That is to say, our predictions are correct "
                    f"<span style='font-size: 21px;'>{accuracy * 100:.0f}%</span> of the time, "
                    f"the model catches <span style='font-size: 21px;'>{recall * 100:.0f}%</span> of the defaulting customers, and "
                    f"the model is correct <span style='font-size: 21px;'>{precision * 100:.0f}%</span> of times it predicts \"Bad\".",
                    unsafe_allow_html=True)
with tab2:
    # Draw Model Feature Importance
    st.subheader(f"Feature Importances for {model_choice}")
    # Function to get feature importances
    def get_feature_importances(model):
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            # Return None or zero importances if the model does not support feature importance
            return np.zeros(len(feature_names))

    # Show model feature importances on the main page
    if model_choice:
        # st.markdown(f"<p style='text-align: center;'>Feature Importances for {model_choice}</p>", unsafe_allow_html=True)
        # Get feature importances
        importances = get_feature_importances(models[model_choice])

        if importances is not None and np.sum(importances) > 0:
            # Create a DataFrame for plotting
            df_importances = pd.DataFrame({'Features': feature_names, 'Importance': importances})
            df_importances = df_importances.sort_values('Importance', ascending=False)

            # Plotting feature importances
            fig, ax = plt.subplots()
            ax.barh(df_importances['Features'], df_importances['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importances')
            st.pyplot(fig)
        else:
            st.write("Selected model does not support feature importance or importance data is not available.")

def remove_prediction_from_history(pid):
    # Filter out the prediction with the given pid
    new_history = [item for item in st.session_state.prediction_history if item[0] != pid]
    st.session_state.prediction_history = new_history

# Sidebar for displaying and managing prediction history
with st.expander("See History Predictions"):
    st.caption("We will show only 10 most recent predictions")
    for pid, pred, _ in st.session_state.prediction_history:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Show ID {pid}: {pred}", key=f"show_{pid}"):
                # Show details when 'Show' is clicked
                features_used = next(item for item in st.session_state.prediction_history if item[0] == pid)[2]
                st.write(f"Details for ID {pid}:")
                st.json(features_used)
        with col2:
            # Add a button to delete this entry
            if st.button("Delete", key=f"delete_{pid}"):
                remove_prediction_from_history(pid)
                st.experimental_rerun()
