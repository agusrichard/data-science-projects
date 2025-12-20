from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


models = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=21),
        {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
    ),
    "RandomForest": (
        RandomForestClassifier(
            n_estimators=300,
            random_state=21,
            n_jobs=-1
        ),
        {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False],
            'max_features': ['sqrt', 'log2']
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=21),
        {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 8],
            'subsample': [0.8, 1.0],
            'min_samples_split': [2, 5]
        }
    ),
    "XGBoost": (
        XGBClassifier(
            eval_metric="logloss",
            random_state=21,
            n_jobs=-1
        ),
        {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    )
}
