
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor


def predictor(X_train, X_test, y_train, y_test, class_choice):
    if class_choice == "Classification":
        clf = LazyClassifier(
            verbose=0, ignore_warnings=True, custom_metric=None)
    else:
        clf = LazyRegressor(
            verbose=0, ignore_warnings=True, custom_metric=None)
    models = clf.fit(X_train, X_test, y_train, y_test)
    return models
