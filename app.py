from model import predictor
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
st.set_page_config(layout="wide")


st.title("Quick-Profile")
st.subheader("Get a quick profile of models that work best with your data.")

uploaded_file = st.file_uploader("Upload CSV file")

try:
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

            view_filter = st.selectbox("View Filtered Data", [
                                       "Yes", "No"], key="filter")
            if view_filter == "Yes":
                # Check if columns are eligible for regression
                non_continuous_columns = []
                for column in df.columns:
                    if df[column].dtype != 'float64' and df[column].dtype != 'int64':
                        non_continuous_columns.append(column)

                # Filter the dataframe by dropping non-continuous columns
                df = df.drop(columns=non_continuous_columns, axis=1)

                view_data = st.selectbox(
                    "View Data", ["Yes", "No"], key="data")
                if view_data == "Yes":
                    st.write(df)
            view_data = st.selectbox(
                "View Data", ["Yes", "No"], key="data_normal")
            if view_data == "Yes":
                st.write(df)

            columns = df.columns.tolist()

            target = st.selectbox("Select Target variable", [
                                  ""] + columns, index=0, key="target")
            labels = st.multiselect("Select Labels", columns, key="labels")
            if labels and target:
                class_choice = st.selectbox("Select Classification or Regression", [
                    "Classification", "Regression"], key="class_choice")

                # Ensure target and labels are lists of column names
                target = [target]
                labels = list(labels)

                # Select the specified columns as DataFrames
                X = df[labels]
                y = df[target]

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=42, test_size=0.2)

                models = predictor(X_train, X_test, y_train,
                                   y_test, class_choice=class_choice)
                st.text(models)

        else:
            st.error("Please upload a CSV file.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
