from Supervised import LazyClassifier, LazyRegressor
import pandas as pd


def predictor(X_train, X_test, y_train, y_test, class_choice):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: '%.1f' %
                  x)
    if class_choice == "Classification":
        clf = LazyClassifier(
            verbose=0, ignore_warnings=True, custom_metric=None)
    else:
        clf = LazyRegressor(
            verbose=0, ignore_warnings=True, custom_metric=None)
    models = clf.fit(X_train, X_test, y_train, y_test)
    return models
