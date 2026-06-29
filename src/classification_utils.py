import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def prepare_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def encode_train_test(X_train, X_test):
    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test)

    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0
    )

    return X_train_encoded, X_test_encoded

def train_decision_tree(X_train, y_train, random_state=44):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=44):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train, random_state=44):
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    return results