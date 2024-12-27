# ================================
# 1. Import Necessary Libraries
# ================================
import os
import warnings
import pandas as pd
import polars as pl
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Define working directory for data
DATA_DIR = "../../data"

# ================================
# 2. Load Datasets
# ================================
consumption = pl.read_parquet(f"{DATA_DIR}/consumption_filtered.parquet")
consumption_wide = pd.read_parquet(f"{DATA_DIR}/consumption_wide.parquet")
weather = pl.read_parquet(f"{DATA_DIR}/weather.parquet")
socioeconomic = pl.read_parquet(f"{DATA_DIR}/socioeconomic.parquet")

# Load and combine cadaster data
cadaster_files = [
    f"{DATA_DIR}/cadaster_lleida.gml",
    f"{DATA_DIR}/cadaster_alcarras.gml",
    f"{DATA_DIR}/cadaster_alpicat.gml"
]
cadaster = pd.concat([gpd.read_file(file) for file in cadaster_files])

# Load postal code data
postalcodes = gpd.read_file(f"{DATA_DIR}/postal_codes_lleida.gpkg")

# ================================
# 3. Preprocess Cadaster and Postal Code Data
# ================================
# Rename columns for consistency
postalcodes.rename({"PROV": "provincecode", "CODPOS": "postalcode"}, axis=1, inplace=True)

# Calculate centroids for cadaster data
cadaster["centroid"] = cadaster.geometry.centroid

# Reproject postal codes to the same CRS as cadaster data
postalcodes = postalcodes.to_crs(cadaster.crs)

# Perform spatial join to match postal codes with cadaster
cadaster = gpd.sjoin(
    cadaster,  # GeoDataFrame with building centroids
    postalcodes[['postalcode', 'geometry']],  # Postal code polygons
    how='left',  # Include all rows from cadaster
    predicate='within'  # Match if the centroid is within a postal code polygon
)

# Filter cadaster data based on specific conditions
cadaster = cadaster[["postalcode", "currentUse", "conditionOfConstruction",
                     "numberOfDwellings", "reference", "value"]]
cadaster.rename({
    "numberOfDwellings": "households",
    "reference": "cadastralref",
    "value": "builtarea",
    "currentUse": "currentuse",
    "conditionOfConstruction": "condition"}, axis=1, inplace=True)

# Filter rows with missing postal codes or non-residential or non-functional buildings
cadaster = cadaster[~pd.isna(cadaster.postalcode) &
                    (cadaster.currentuse == "1_residential") &
                    (cadaster.condition == "functional")]

# Aggregate cadaster data by postal code
cadaster = pl.DataFrame(
    cadaster.groupby("postalcode")[["builtarea", "households"]].sum().reset_index())


# ================================
# 5. Apply Scaling
# ================================
# Apply Yeo-Johnson PowerTransformer for scaling
consumption_wide_znorm = pd.DataFrame(
    PowerTransformer(method="yeo-johnson").fit_transform(consumption_wide),
    columns=consumption_wide.columns,
    index=consumption_wide.index
)

# ================================
# 6. Perform Clustering
# ================================

# Initialize KMeans clustering algorithm

print("Performing clustering with KMeans")
clustering_algorithm = KMeans(n_clusters=5, init='k-means++', algorithm="lloyd", max_iter=300, n_init=25, random_state=42)

# Prepare data for clustering
clustering_X = consumption_wide_znorm.dropna(axis=0)
index_X = clustering_X.index.to_frame(index=False)
clustering_X = clustering_X.reset_index(drop=True)

# Perform clustering
cl_results = clustering_algorithm.fit_predict(clustering_X)

# Combine clustering results with postalcode and date information
clustering_results = (pl.concat([
        pl.DataFrame(index_X),
        pl.DataFrame({"cluster": cl_results})],
        how="horizontal").
    with_columns(pl.col("date").cast(pl.Date)))

# ================================
# 7. Aggregate Weather Data to Daily
# ================================
weather_daily = (weather
    .with_columns(
        pl.col("time").dt.convert_time_zone("Europe/Madrid").alias("localtime"))
    .with_columns(
        pl.col("localtime").dt.date().alias("date"))
    .group_by(["date", "postalcode"])
    .agg(
        (pl.col("airtemperature").drop_nans().mean()).round(2).alias("airtemperature"),
        (pl.col("relativehumidity").drop_nans().mean()).round(2).alias("relativehumidity"),
        (pl.col("totalprecipitation").drop_nans().mean() * 24).round(2).alias("totalprecipitation"),
        (pl.col("ghi").drop_nans().mean()).round(2).alias("ghi"),
        (pl.col("sunelevation").drop_nans().mean()).round(2).alias("sunelevation")
    )
)

# ================================
# 8. Prepare Daily Consumption Data
# ================================
# Aggregate daily consumption data and include clustering results
consumption_daily = (consumption
    .group_by(["date", "postalcode"])
    .agg(
        (pl.col("consumption_filtered").drop_nans().mean() * 24).round(2).alias("consumption"))
    .join(clustering_results, on=["postalcode", "date"]))

# ================================
# 9. Prepare Data for Classification Model
# ================================
# Merge consumption, socioeconomic, cadaster, and weather data
all_data_daily = (consumption_daily
    .with_columns(pl.col("date").dt.year().cast(pl.Int64).alias("year"))
    .join(socioeconomic, on=["postalcode", "year"])
    .join(cadaster, on=["postalcode"])
    .join(weather_daily, on=["postalcode", "date"])
    .with_columns(
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("weekday"))
    .sort(["postalcode", "date"])
    .to_pandas())

# ================================
# 10. Train Classification Model
# ================================
# Split the dataset into features (X) and target (y)
X_train, X_test, y_train, y_test = train_test_split(
    all_data_daily.drop(["date", "postalcode", "cluster", "consumption"], axis=1),
    all_data_daily.cluster,
    random_state=42,
    test_size=0.2
)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest
param_grid = {'n_estimators': [200, 400, 500, 600, 800], 'max_depth': [5, 10, 20, 30]}

# Perform GridSearchCV to find the best parameters

print("Performing GridSearch")

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1,  # Use all available CPU cores
    verbose=1
)
grid_search.fit(X_train, y_train)

# Output the best parameters from GridSearchCV
print(f"Best parameters: {grid_search.best_params_}")

# Initialize the Random Forest classifier with the best parameters
best_rf_model = grid_search.best_estimator_

# Fit the best model on the training set
best_rf_model.fit(X_train, y_train)

# Make predictions on the training set and calculate accuracy
y_train_pred = best_rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Model Training Accuracy: {round(train_accuracy * 100, 2)}%")

# Make predictions on the test set and calculate accuracy
y_pred = best_rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Model Testing Accuracy: {round(test_accuracy * 100, 2)}%")

import seaborn as sns
import matplotlib.pyplot as plt

# Extract results from grid search
results = pd.DataFrame(grid_search.cv_results_)

# Pivot the results for visualization
pivot_table = results.pivot_table(
    index='param_max_depth',
    columns='param_n_estimators',
    values='mean_test_score'
)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    pivot_table,
    annot=True,
    cmap="YlGnBu",
    cbar_kws={'label': 'Mean Test Accuracy'},
    fmt=".3f"
)
plt.title("Grid Search Results (Mean Test Accuracy)")
plt.xlabel("Number of Estimators")
plt.ylabel("Max Depth")
plt.show()