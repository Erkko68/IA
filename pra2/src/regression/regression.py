# ================================
# 1. Import Necessary Libraries
# ================================
import warnings
import os

import geopandas as gpd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from utils import *

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Define working directory for data
DATA_DIR = "../../data"
PLOT_DIR = "../../plots"
os.makedirs(f"{PLOT_DIR}/RandomForestRegressor/",exist_ok=True)

consumption = pl.read_parquet(f"{DATA_DIR}/consumption_filtered.parquet")
weather = pl.read_parquet(f"{DATA_DIR}/weather.parquet")
cadaster = pl.read_parquet(f"{DATA_DIR}/cadaster_processed.parquet")
socioeconomic = pl.read_parquet(f"{DATA_DIR}/socioeconomic.parquet")
postalcodes = gpd.read_file(f"{DATA_DIR}/processed_postal_codes.gpkg")
postalcodes = postalcodes.to_crs(cadaster.crs)

all_data_hourly = (consumption.
    select(['postalcode', 'localtime', 'hour', 'contracts', 'consumption_filtered', 'consumption']).
    with_columns(
        pl.col("localtime").dt.year().cast(pl.Int64).alias("year")).
    join(
        socioeconomic, on = ["postalcode", "year"]).
    join(
        cadaster, on=["postalcode"]).
    join(
        weather.
        with_columns(
            pl.col("time").dt.convert_time_zone("Europe/Madrid").
                alias("localtime")),
        on=["postalcode", "localtime"]).
    with_columns(
        pl.col("localtime").dt.month().cast(pl.Utf8).alias("month"),
        pl.col("localtime").dt.weekday().cast(pl.Utf8).alias("weekday")).
    sort(
        ["localtime"]).
    to_pandas()
)

# Define features and target
X = (all_data_hourly.
    drop(
        ['localtime', 'time', 'consumption_filtered', 'consumption'],
        axis=1))
X = X[~pd.isna(all_data_hourly.consumption_filtered)]
y = all_data_hourly['consumption_filtered']
y = y[~pd.isna(all_data_hourly.consumption_filtered)]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Preprocessing for numerical data: scaling
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

models = {
    "Decision Tree": DecisionTreeRegressor()
}
hyperparams_ranges = {
    "Decision Tree": {
        'model__max_depth': [15, 25, 40, 60],
        'model__min_samples_leaf': [5, 10, 15, 20],
    }
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = (
    train_test_split(X, y,
                     test_size=(20*96*len(postalcodes))/len(X), shuffle=False))

# Evaluate each model
pipelines = {}
for model_name, model in models.items():
    pipelines[model_name] = evaluate_regression_model(
        model_name=model_name,
        preprocessor=preprocessor,
        model=model,
        hyperparams_ranges=hyperparams_ranges[model_name],
        cross_val=TimeSeriesSplit(n_splits=5, test_size=96*len(postalcodes)),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test)
    for postal_code in ["25001", "25193"]:
        plot_regression_results(
            pipeline=pipelines[model_name],
            df=all_data_hourly.drop(['time', 'consumption_filtered'], axis=1),
            filename=f"{PLOT_DIR}/RandomForestRegressor/results_{model_name}_postalcode_{postal_code}.pdf",
            postal_code=postal_code,
            model_name=model_name,
            hours=96,
            npred=10)
