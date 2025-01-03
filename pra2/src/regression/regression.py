import os
import random
import warnings

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import DateFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, validation_curve, TimeSeriesSplit
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

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
numerical_transformer = MinMaxScaler()

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
    #"DecisionTree": DecisionTreeRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    #"RandomForest": RandomForestRegressor(),
}

hyperparameter_ranges = {
    "DecisionTree": {
        'model__max_depth': [20, 30, 40, 60],
        'model__min_samples_leaf': [10, 15, 20],
    },
    # Best 100, 0.2 , 10
    # samples set to 100 by default
    "GradientBoosting": {
        'model__learning_rate': [0.1, 0.2],
        'model__max_depth': [10, 20],
    },
    # Best 300, 20 (Too resource intensive)
    "RandomForest": {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [15, 20, 30],
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

# Updated evaluation function
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
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(best_pipeline, f"{MODEL_PATH}{model_name}_best_model.pkl")

    # ============================
    # Visualizations
    # ============================

    # Create folder folder
    os.makedirs(f"{PLOT_DIR}{model_name}/",exist_ok=True)
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importances = best_pipeline.named_steps['model'].feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(X_train.columns, feature_importances, color='skyblue')
        plt.title(f"{model_name} - Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.savefig(f"{PLOT_DIR}{model_name}/feature_importances.png", format="png", dpi=300) 
    else:
        # Permutation importance (generic for all models)
        result = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=0, scoring='neg_mean_squared_error')
        sorted_idx = result.importances_mean.argsort()
        plt.figure(figsize=(10, 6))
        plt.barh(X_train.columns[sorted_idx], result.importances_mean[sorted_idx], color='lightgreen')
        plt.title(f"{model_name} - Permutation Importances")
        plt.xlabel("Mean Importance")
        plt.ylabel("Features")
        plt.savefig(f"{PLOT_DIR}{model_name}/permutation_importances.png", format="png", dpi=300) 
    
    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{model_name} - Residuals Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.savefig(f"{PLOT_DIR}{model_name}/residual_plot.png", format="png", dpi=300) 

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline, X_train, y_train, cv=cross_val, scoring='neg_mean_squared_error', n_jobs=-1
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.sqrt(train_scores_mean), label="Training RMSE", color="blue", marker='o')
    plt.plot(train_sizes, np.sqrt(test_scores_mean), label="Validation RMSE", color="green", marker='o')
    plt.title(f"{model_name} - Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{PLOT_DIR}{model_name}/learning_curve.png", format="png", dpi=300) 

    # Hyperparameter tuning visualization (validation curve)
    for param_name, param_values in hyperparameter_ranges.items():
        train_scores, test_scores = validation_curve(
            best_pipeline, X_train, y_train, param_name=param_name, param_range=param_values,
            cv=cross_val, scoring='neg_mean_squared_error', n_jobs=-1
        )
        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(param_values, np.sqrt(train_scores_mean), label="Training RMSE", color="blue", marker='o')
        plt.plot(param_values, np.sqrt(test_scores_mean), label="Validation RMSE", color="green", marker='o')
        plt.title(f"{model_name} - Validation Curve ({param_name})")
        plt.xlabel(param_name)
        plt.ylabel("RMSE")
        plt.legend()
        plt.savefig(f"{PLOT_DIR}{model_name}/validation_curve.png", format="png", dpi=300) 

    # Grid Search Heatmap (only for two hyperparameters)
    keys = list(hyperparameter_ranges.keys())
    if len(keys) >= 2:
        param1, param2 = keys[:2]
        results = pd.DataFrame(grid_search.cv_results_)
        heatmap_data = results.pivot_table(
            index=f"param_{param1}",
            columns=f"param_{param2}",
            values="mean_test_score"
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean Test Score'})
        plt.title(f"{model_name} - Grid Search Heatmap ({param1} vs {param2})")
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.savefig(f"{PLOT_DIR}{model_name}/grid_search_heatmap.png", format="png", dpi=300)

    # Hyperparameter Score Line Plot
    for param_name, param_values in hyperparameter_ranges.items():
        train_scores, test_scores = validation_curve(
            best_pipeline, X_train, y_train, param_name=param_name, param_range=param_values,
            cv=cross_val, scoring='neg_mean_squared_error', n_jobs=-1
        )
        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(param_values, np.sqrt(train_scores_mean), label="Training RMSE", color="blue", marker='o')
        plt.plot(param_values, np.sqrt(test_scores_mean), label="Validation RMSE", color="green", marker='o')
        plt.title(f"{model_name} - Hyperparameter Performance ({param_name})")
        plt.xlabel(param_name)
        plt.ylabel("RMSE")
        plt.legend()
        plt.savefig(f"{PLOT_DIR}{model_name}/{param_name}_performance.png", format="png", dpi=300)

    # Model Performance Across CV Folds
    results = pd.DataFrame(grid_search.cv_results_)
    mean_scores = results['mean_test_score']
    std_scores = results['std_test_score']

    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(mean_scores)), mean_scores, yerr=std_scores, fmt='o', ecolor='gray', capsize=4, color='blue')
    plt.title(f"{model_name} - CV Performance")
    plt.xlabel("Hyperparameter Set Index")
    plt.ylabel("Mean Test Score")
    plt.grid()
    plt.savefig(f"{PLOT_DIR}{model_name}/cv_performance.png", format="png", dpi=300)

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

# ==================================
# 9. Plot Regression Results
# ==================================
def plot_regression_results(model_name, df, filename, postal_code, hours=96, npred=1, start_points=None):
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
        for i in range(npred):
            # Use a different starting point for each iteration
            start = start_points[i]
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

N_PRED = 10

POSTALCODES_DIR = "../../plots/PostalCodesPredictions/"
os.makedirs(POSTALCODES_DIR,exist_ok=True)

for postal_code in range(25001, 25194):
    postal_code_str = str(postal_code).zfill(5)
    # Calculate the maximum possible start
    postal_filter = all_data_hourly['postalcode'] == postal_code_str
    filtered_df = all_data_hourly[postal_filter].copy()
    max_start = len(filtered_df) - 96  # Assuming hours=96

    if max_start <= 0:
        print(f"Not enough data for postal code {postal_code_str}.")
        continue

    # Generate a list of random start points for each iteration
    start_points = [random.randint(0, max_start) for _ in range(N_PRED)]  # Generate 10 random start points

    # Iterate over models
    for model_name in models.keys():
        # Plot results using the same start points for each model
        plot_regression_results(
                model_name=model_name,  # Pass only the model name
                df=all_data_hourly.drop(['time', 'consumption_filtered'], axis=1),
                filename=f"{POSTALCODES_DIR}{model_name.replace(' ', '')}/{postal_code_str}.pdf",
                postal_code=postal_code_str,
                hours=96,
                npred=N_PRED,  # Number of predictions
                start_points=start_points  # Pass the same start points array to each model
        )