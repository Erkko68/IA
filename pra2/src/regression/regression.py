import warnings
import os
import random
import geopandas as gpd
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import DateFormatter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE
import joblib  # Import joblib for saving/loading models

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ==================================
# 2. Define Global Constants
# ==================================
DATA_DIR = "../../data"
PLOT_DIR = "../../plots/Regression/"
MODEL_PATH = "../../models/"
os.makedirs(PLOT_DIR, exist_ok=True)

# ==================================
# 3. Load and Preprocess Data
# ==================================
# Load datasets
consumption_data = pl.read_parquet(f"{DATA_DIR}/consumption_filtered.parquet")
weather_data = pl.read_parquet(f"{DATA_DIR}/weather.parquet")
cadaster_data = pl.read_parquet(f"{DATA_DIR}/cadaster_processed.parquet")
socioeconomic_data = pl.read_parquet(f"{DATA_DIR}/socioeconomic.parquet")
postalcodes_data = gpd.read_file(f"{DATA_DIR}/processed_postal_codes.gpkg")

# Merge datasets
all_data_hourly = (
    consumption_data
    .select(['postalcode', 'localtime', 'hour', 'contracts', 'consumption_filtered', 'consumption'])
    .with_columns(
        pl.col("localtime").dt.year().cast(pl.Int64).alias("year")
    )
    .join(socioeconomic_data, on=["postalcode", "year"])
    .join(cadaster_data, on=["postalcode"])
    .join(
        weather_data.with_columns(
            pl.col("time").dt.convert_time_zone("Europe/Madrid").alias("localtime")
        ),
        on=["postalcode", "localtime"]
    )
    .with_columns(
        pl.col("localtime").dt.month().cast(pl.Utf8).alias("month"),
        pl.col("localtime").dt.weekday().cast(pl.Utf8).alias("weekday")
    )
    .sort(["localtime"])
    .to_pandas()
)

# ==================================
# 4. Prepare Features and Target
# ==================================
# Define features and target
X = all_data_hourly.drop(['localtime', 'time', 'consumption_filtered', 'consumption'], axis=1)
X = X[~pd.isna(all_data_hourly.consumption_filtered)]
y = all_data_hourly['consumption_filtered']
y = y[~pd.isna(all_data_hourly.consumption_filtered)]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

# ==================================
# 5. Define Preprocessing Steps
# ==================================
# Preprocessing for numerical data: scaling
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# ==================================
# 6. Define Models and Hyperparameters
# ==================================

models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Random Forest": RandomForestRegressor(),
}

hyperparameter_ranges = {
    "Decision Tree": {
        'model__max_depth': [10, 20, 30],
        'model__min_samples_leaf': [10, 15, 20],
    },
    "Gradient Boosting": {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 10],
    },
    "Random Forest": {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [10, 15, 20],
    },
}

# ==================================
# 7. Split Data into Train and Test Sets
# ==================================
test_size_ratio = (20 * 96 * len(postalcodes_data)) / len(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, shuffle=False)

# ==================================
# 8. Train and Evaluate Models with RFE
# ==================================
def evaluate_regression_model(
    model_name, preprocessor, model, hyperparameter_ranges, cross_val, X_train, y_train, X_test, y_test
):
    # Create the pipeline with preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=hyperparameter_ranges, cv=cross_val, n_jobs=-1,
        verbose=3, scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)

    # Get the best pipeline from the grid search
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    cvrmse = rmse / y_test.mean()

    # Print results
    print(f"{model_name} - Best Parameters: {grid_search.best_params_}")
    print(f"{model_name} - RMSE: {rmse:.2f}")
    print(f"{model_name} - CV(RMSE): {cvrmse * 100:.2f}%")

    # Save the trained model to a file
    os.makedirs(MODEL_PATH,exist_ok=True)
    joblib.dump(best_pipeline, f"{MODEL_PATH}{model_name}_best_model.pkl")


'''
for model_name, model in models.items():
    # Train and save the best model
    evaluate_regression_model(
        model_name=model_name,
        preprocessor=preprocessor,
        model=model,
        hyperparameter_ranges=hyperparameter_ranges[model_name],
        cross_val=TimeSeriesSplit(n_splits=5, test_size=96 * len(postalcodes_data)),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
'''

# ==================================
# 9. Plot Regression Results
# ==================================
def plot_regression_results(model_name, df, filename, postal_code, hours=96, npred=1, start=0):
    
    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model from the saved file
    loaded_model = joblib.load(f"{MODEL_PATH}{model_name}_best_model.pkl")

    postal_filter = df['postalcode'] == postal_code
    filtered_df = df[postal_filter].copy()

    X_test_filtered = filtered_df.drop(["localtime", "consumption"], axis=1)
    y_pred = loaded_model.predict(X_test_filtered)
    filtered_df["predicted"] = y_pred

    with PdfPages(filename) as pdf:
        for _ in range(npred):

            df_slice = filtered_df.iloc[start:start + hours]

            plt.figure(figsize=(12, 6))
            plt.plot(df_slice["localtime"], df_slice["predicted"], label='Predicted', marker='x', linestyle='--', markersize=3)
            plt.plot(df_slice["localtime"], df_slice["consumption"], label='Actual', marker='o', linestyle='-', markersize=3)

            plt.title(f"Actual vs Predicted Consumption\nPostal Code: {postal_code} ({model_name})")
            plt.xlabel("Time")
            plt.ylabel("Consumption")
            plt.legend()
            plt.grid(True)

            ax = plt.gca()
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            plt.tight_layout()

            pdf.savefig()
            plt.close()

# Set a fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

for postal_code in ["25001", "25193"]:
    # Calculate start
    postal_filter = all_data_hourly['postalcode'] == postal_code
    filtered_df = all_data_hourly[postal_filter].copy()
    max_start = len(filtered_df) - 96  # Assuming hours=96

    if max_start <= 0:
        print(f"Not enough data for postal code {postal_code}.")
        continue

    # Generate a consistent random start
    random_start = random.randint(0, max_start)

    # Iterate over models
    for model_name in models.keys():
        # Plot results
        plot_regression_results(
                model_name=model_name,  # Pass only the model name
                df=all_data_hourly.drop(['time', 'consumption_filtered'], axis=1),
                filename=f"{PLOT_DIR}/{model_name.replace(' ', '')}/postalcode_{postal_code}.pdf",
                postal_code=postal_code,
                hours=96,
                npred=10,
                start=random_start
        )
