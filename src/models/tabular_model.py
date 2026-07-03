"""
XGBoost multi-class classifier for the tabular CBC dataset.
Predicts one of 4 classes: normal, leukemia, malaria, both.
"""

import xgboost as xgb


def build_tabular_model(num_classes=4):
    """
    Returns an XGBoost classifier configured for multi-class classification.
    """
    model = xgb.XGBClassifier(
        objective="multi:softprob",   # multi-class, returns class probabilities
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return model


if __name__ == "__main__":
    model = build_tabular_model()
    print(model)