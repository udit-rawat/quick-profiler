from lazypredict.Supervised import LazyClassifier, LazyRegressor
import pandas as pd


def predictor(X_train, X_test, y_train, y_test, class_choice):
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.width', None)  # Remove column width restriction
    pd.set_option('display.float_format', lambda x: '%.2f' %
                  x)  # Set float format
    if class_choice == "Classification":
        clf = LazyClassifier(
            verbose=2, ignore_warnings=True, custom_metric=None)
    else:
        clf = LazyRegressor(
            verbose=2, ignore_warnings=True, custom_metric=None)
    models = clf.fit(X_train, X_test, y_train, y_test)
    return models
