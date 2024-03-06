from model import predictor
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd


def main():
    st.set_page_config(layout="centered")
    st.markdown("<h1 style='text-align: center;'>QUICK-PROFILER</h1>",
                unsafe_allow_html=True)
    st.subheader(
        "Get a quick profile of models that work best with your data.")

    uploaded_file = st.file_uploader("Upload CSV file")

    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                # Read the dataset
                df = pd.read_csv(uploaded_file)

                # Implement limiter to restrict dataset size
                if len(df) > 10000:
                    st.warning(
                        "The uploaded dataset contains more than 10,000 rows. Only the first 10,000 rows will be used.")
                    df = df.head(10000)

                st.markdown("---")
                view_data = st.selectbox(
                    "View Data", ["Select an option", "Yes", "No"], index=0)
                if view_data == "Yes":
                    st.markdown(
                        "<style>div[data-testid='stDataFrame'] div{margin: 0 auto;}</style>", unsafe_allow_html=True)
                    st.write(df)

                elif view_data == "No":
                    pass  # Do nothing if "No" is selected

                st.markdown("---")

                # Auto-detect the type of data based on dtype of target variable
                target, labels = select_variables(df.columns.tolist(), df)
                if target:
                    if df[target].dtype in ['object', 'int']:
                        class_choice = st.selectbox("Select Classification or Regression", [
                                                    "Classification", "Regression"])
                    else:
                        class_choice = "Regression"

                    # Add a button to initiate the predictor
                    if st.button("Run"):
                        # Show a progress bar while the model is being trained
                        progress_bar = st.progress(0)
                        for percent_complete in range(100):
                            progress_bar.progress(percent_complete + 1)

                            X, y = df[labels], df[target]
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, random_state=108, test_size=0.2)

                        # Perform prediction
                        models = predictor(
                            X_train, X_test, y_train, y_test, class_choice=class_choice)

                        # Show toast message upon completion
                        st.success("Prediction completed successfully!")

                        # Display the results
                        st.write_stream(models)
                else:
                    st.warning("Please select a target variable.")

            else:
                st.error("Please upload a CSV file.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Footer HTML
    footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f05941;
        color: white;
        text-align: right;
        padding: 10px;
        display: flex;
        justify-content: flex-end;
    }
    .footer img {
        border: none;
        margin-left: 10px;
    }
    </style>
    <div class="footer">
        <a href="https://www.linkedin.com/in/udit-rawat-65204a279/" target="_blank" style="padding: 5px;">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" width="20" height="20" style="vertical-align: middle;" alt="LinkedIn">
        </a>
        <a href="https://github.com/udit-rawat" target="_blank" style="padding: 5px;">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/github.svg" width="20" height="20" style="vertical-align: middle;" alt="GitHub">
        </a>
        <a href="mailto:uditcsrawat@gmail.com" target="_blank" style="padding: 5px;">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/gmail.svg" width="20" height="20" style="vertical-align: middle;" alt="Email">
        </a>
    </div>
    """

    # Display the footer
    st.markdown(footer_html, unsafe_allow_html=True)


def select_variables(columns, df):
    target = st.selectbox("Select Target variable", [
                          ""] + columns, index=0, key="target")
    if target:
        labels = st.multiselect(
            "Select Labels", [col for col in columns if col != target], key="labels")
    else:
        labels = []
    return target, labels


if __name__ == "__main__":
    main()