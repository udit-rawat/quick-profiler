# from model import predictor
# from sklearn.model_selection import train_test_split
# import streamlit as st
# import pandas as pd

# st.title("Quick-Profile")
# st.subheader("Get a quick profile of models that work best with your data.")

# uploaded_file = st.file_uploader("Upload CSV file")

# try:
#     if uploaded_file is not None:
#         if uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)

#             view_data = st.selectbox("View Data", ["Yes", "No"])
#             if view_data == "Yes":
#                 st.write(df)

#             columns = df.columns.tolist()

#             target = st.selectbox("Select Target variable", [
#                                   ""] + columns, index=0)
#             labels = st.multiselect("Select Labels", columns)
#             if labels and target:
#                 class_choice = st.selectbox("Select Classification or Regression", [
#                     "Classification", "Regression"])

#                 X_train, X_test, y_train, y_test = train_test_split(
#                     df[target], df[labels],  random_state=42, test_size=0.2)

#                 models = predictor(X_train, X_test, y_train,
#                                    y_test, class_choice=class_choice)

#         else:
#             st.error("Please upload a CSV file.")
# except Exception as e:
#     st.error(f"An error occurred: {str(e)}")
#     # st.warning("Please upload a valid CSV file to get started.")
from model import predictor
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd

st.title("Quick-Profile")
st.subheader("Get a quick profile of models that work best with your data.")

uploaded_file = st.file_uploader("Upload CSV file")

try:
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

            view_data = st.selectbox("View Data", ["Yes", "No"])
            if view_data == "Yes":
                st.write(df)

            columns = df.columns.tolist()

            target = st.selectbox("Select Target variable", [
                                  ""] + columns, index=0)
            labels = st.multiselect("Select Labels", columns)
            if labels and target:
                class_choice = st.selectbox("Select Classification or Regression", [
                    "Classification", "Regression"])

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
