# ================================
# 1. Import Necessary Libraries
# ================================
import warnings
import pandas as pd
import polars as pl
import geopandas as gpd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

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
# 4. Adjust weather data
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
# 5. Apply Scaling
# ================================
# Define scaling methods
scalers = {
    "ZNormScaling": pd.DataFrame(
        StandardScaler().fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    ),
    "RobustScaling": pd.DataFrame(
        RobustScaler().fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    )
}

# Select the scaling type
selected_scaling = "ZNormScaling"

# ================================
# 6. Perform Clustering
# ================================
# Initialize KMeans clustering algorithm
print(f"Performing clustering with KMeans using {selected_scaling}")
clustering_algorithm = KMeans(
    n_clusters=3, init='k-means++', algorithm="lloyd", max_iter=300, n_init=25, random_state=42
)

# Prepare data for clustering
clustering_X = scalers[selected_scaling].dropna(axis=0)
index_X = clustering_X.index.to_frame(index=False)
clustering_X = clustering_X.reset_index(drop=True)

# Perform clustering
cl_results = clustering_algorithm.fit_predict(clustering_X)

# Combine clustering results with postal code and date information
clustering_results = (pl.concat([
        pl.DataFrame(index_X),
        pl.DataFrame({"cluster": cl_results})],
        how="horizontal").
    with_columns(pl.col("date").cast(pl.Date)))

# ================================
# 7. Prepare Daily Consumption Data
# ================================
# Aggregate daily consumption data and include clustering results
consumption_daily = (consumption
    .group_by(["date", "postalcode"])
    .agg(
        (pl.col("consumption_filtered").drop_nans().mean() * 24).round(2).alias("consumption"))
    .join(clustering_results, on=["postalcode", "date"]))

# ================================
# 8. Prepare Data for Classification Model
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
# 9. Train Classification Model
# ================================
# Split the dataset into features (X) and target (y)
X_train, X_test, y_train, y_test = train_test_split(
    all_data_daily.drop(["postalcode", "date", "cluster", "consumption"], axis=1),
    all_data_daily.cluster,
    random_state=42,
    test_size=0.2
)

# Define the parameter grid for Random Forest
param_grid = {'n_estimators': [200, 400, 500, 600, 800], 'max_depth': [5, 10, 20, 30, 40]}

# Perform GridSearchCV to find the best parameters
print("Performing GridSearch")

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,       # Cross validation
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

# ================================
# 10. Visualize Grid Search Results
# ================================
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
plt.title(f"Grid Search Results (Mean Test Accuracy) - {selected_scaling}")
plt.xlabel("Number of Estimators")
plt.ylabel("Max Depth")
plt.savefig(f"grid_search_results_{selected_scaling}.png", dpi=300, bbox_inches="tight")

# =====================
# 11. Confusion Matrix
# =====================

# Here we want to visaulize the relation between number of samples with the training, 
# and see if the accuracy is directly proportional to the number of data points, while
# also checking wich clusters fail more to predict.

# Get cluster counts from actual and predicted
actual_cluster_counts = y_test.value_counts().sort_index()
predicted_cluster_counts = pd.Series(y_pred).value_counts().sort_index()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=actual_cluster_counts.index, yticklabels=actual_cluster_counts.index)
plt.title(f"{selected_scaling} - Confusion Matrix Heatmap")
plt.xlabel("Predicted Cluster")
plt.ylabel("Actual Cluster")
plt.tight_layout()

plt.savefig(f"confusion_matrix_{selected_scaling}.png", dpi=300, bbox_inches="tight")

# =====================
# 12. Feature Importances
# =====================

# Get feature importances from the trained model
feature_importances = best_rf_model.feature_importances_

# Create a DataFrame for easier manipulation and visualization
feature_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('Feature Importances - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()

# Save the plot
plt.savefig(f"feature_importances_{selected_scaling}.png", dpi=300, bbox_inches="tight")


'''
# ================================
# 12. Plot an Individual Tree
# ================================
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Get the best tree (you can adjust the index or apply custom criteria)
best_tree = best_rf_model.estimators_[0]

# Plot the tree using matplotlib
plt.figure(figsize=(20, 10))
plot_tree(best_tree, 
          filled=True, 
          feature_names=X_train.columns,  # Use feature names
          class_names=list(map(str, best_rf_model.classes_)),  # Class names
          rounded=True, 
          proportion=True)

# Save the plot as an HTML file (using static plotting)
plt.savefig("best_tree.png", format="svg")
'''