import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = (
    "data/processed/"
    "train_grouped_spatial.csv"
)

OUTPUT_MODEL_PATH = (
    "model/"
    "xgboost_grouped_spatial_tuned.joblib"
)

RESULTS_PATH = (
    "data/interim/"
    "xgboost_grouped_spatial_tuning_results.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "sale_year",
    "gisacres",
    "heatedarea",
    "finisharea",
    "bedrooms",
    "bathroom_equivalents",
    "fireplaces",
    "fingarage",
    "finattic",
    "property_age",
    "landusefulldescription",
    "neighborhood",
    "latitude",
    "longitude",
    "distance_to_uptown_miles",
]

TARGET = "saleprice"
GROUP_COLUMN = "parcelid"


numeric_features = [
    "sale_year",
    "gisacres",
    "heatedarea",
    "finisharea",
    "bedrooms",
    "bathroom_equivalents",
    "fireplaces",
    "fingarage",
    "finattic",
    "property_age",
    "latitude",
    "longitude",
    "distance_to_uptown_miles",
]

categorical_features = [
    "landusefulldescription",
    "neighborhood",
]


# ============================================================
# LOAD TRAINING DATA ONLY
# ============================================================

print("Loading grouped spatial training data...")

train_df = pd.read_csv(
    TRAIN_PATH,
    low_memory=False,
)

print(f"Training rows   : {len(train_df):,}")
print(
    f"Training parcels: "
    f"{train_df[GROUP_COLUMN].nunique():,}"
)


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train_df[FEATURES].copy()

y_train = train_df[TARGET].copy()

groups = train_df[GROUP_COLUMN].copy()


# ============================================================
# LOG TARGET
# ============================================================

y_train_log = np.log1p(
    y_train
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features,
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# ============================================================
# BASE XGBOOST MODEL
# ============================================================

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            xgb_model,
        ),
    ]
)


# ============================================================
# PARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    "model__n_estimators": [
        300,
        500,
        700,
        900,
    ],

    "model__learning_rate": [
        0.02,
        0.03,
        0.05,
        0.08,
    ],

    "model__max_depth": [
        4,
        5,
        6,
        7,
        8,
    ],

    "model__min_child_weight": [
        1,
        3,
        5,
        7,
    ],

    "model__subsample": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "model__colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "model__reg_alpha": [
        0,
        0.01,
        0.1,
        0.5,
        1.0,
    ],

    "model__reg_lambda": [
        1,
        2,
        5,
        10,
    ],
}


# ============================================================
# GROUP-AWARE CROSS-VALIDATION
# ============================================================

group_cv = GroupKFold(
    n_splits=3
)


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_distributions,

    n_iter=20,

    scoring="neg_root_mean_squared_error",

    cv=group_cv,

    verbose=2,

    random_state=42,

    n_jobs=1,

    return_train_score=True,
)


# ============================================================
# RUN SEARCH
# ============================================================

print("\n" + "=" * 60)
print("STARTING GROUP-AWARE XGBOOST TUNING")
print("=" * 60)

print("\nImportant:")
print(
    "Only the training dataset is being used."
)
print(
    "The final 25,319-row test set is not used here."
)

print(
    "\nSearch configuration:"
)

print("CV folds       : 3")
print("Random trials  : 20")
print("Total model fits: 60")


search.fit(
    X_train,
    y_train_log,
    groups=groups,
)


# ============================================================
# BEST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TUNING COMPLETED")
print("=" * 60)


print(
    "\nBest CV score "
    "(negative log-RMSE):"
)

print(
    search.best_score_
)


print("\nBest parameters:")

for parameter, value in (
    search.best_params_.items()
):

    print(
        f"{parameter}: {value}"
    )


# ============================================================
# SAVE CV RESULTS
# ============================================================

results_df = pd.DataFrame(
    search.cv_results_
)

results_df = results_df.sort_values(
    "rank_test_score"
)

os.makedirs(
    "data/interim",
    exist_ok=True,
)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)


print("\nTuning results saved:")

print(
    RESULTS_PATH
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True,
)

joblib.dump(
    search.best_estimator_,
    OUTPUT_MODEL_PATH,
)


print("\nBest tuned model saved:")

print(
    OUTPUT_MODEL_PATH
)


print(
    "\nDo not evaluate on the final "
    "test set yet."
)