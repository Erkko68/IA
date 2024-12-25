import os
import warnings
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from utils import consumption_plotter, plot_dendrogram, plot_daily_load_curves_with_centroids_to_pdf
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ================================
# 1. Load and Preprocess Data
# ================================
# Read electricity consumption dataset
consumption = pl.read_parquet("./data/electricity_consumption.parquet")

# Add local time, hour, and date columns
consumption = consumption.with_columns(
    pl.col("time").dt.convert_time_zone("Europe/Madrid").alias("localtime")
)

consumption = consumption.with_columns(
    pl.col("localtime").dt.hour().alias("hour"),
    pl.col("localtime").dt.date().alias("date")
)

# Sort data by postal code and time for proper grouping
consumption = consumption.sort(["postalcode", "localtime"])

# ================================
# 2. Calculate Rolling Statistics
# ================================
# Compute rolling mean, standard deviation, and 10th quantile
rolling_metrics = consumption.group_by("postalcode", maintain_order=True).agg([
    pl.col("consumption").rolling_mean(48, min_periods=4, center=True).alias("rolling_mean"),
    pl.col("consumption").rolling_std(48, min_periods=4, center=True).alias("rolling_std"),
    pl.col("consumption").rolling_quantile(
        quantile=0.1, window_size=168, min_periods=24, center=True, interpolation="nearest"
    ).alias("rolling_q10")
])

# Merge rolling statistics back into the main dataframe
consumption = pl.concat([
    consumption,
    rolling_metrics.explode(["rolling_mean", "rolling_std", "rolling_q10"]).select(pl.all().exclude("postalcode"))
], how="horizontal").with_columns(
    ((pl.col("consumption") - pl.col("rolling_mean")) / pl.col("rolling_std")).alias("z_norm")
)

# Plot Z-normalized values for a sample postal code
consumption_plotter(
    consumption.filter(pl.col("postalcode") == "25001"),
    html_file="plots/example_z_norm.html",
    y_columns=["z_norm"],
    y_title="Z-Normalized Consumption"
)

# ================================
# 3. Filter Outliers
# ================================
# Define Z-normalization thresholds
MAX_Z_THRESHOLD = 4
MIN_Z_THRESHOLD = -2

# Filter consumption data based on Z-normalization and rolling quantile thresholds
consumption = consumption.with_columns(
    pl.when(
        (pl.col("z_norm") < MAX_Z_THRESHOLD) &
        (pl.col("z_norm") > MIN_Z_THRESHOLD) &
        (pl.col("consumption") > (pl.col("rolling_q10") * 0.7))
    ).then(pl.col("consumption")).otherwise(np.nan).alias("consumption_filtered")
)

'''
# Plot original and filtered consumption for each postal code
for postal_code in consumption["postalcode"].unique():
    consumption_plotter(
        consumption.filter(pl.col("postalcode") == postal_code),
        html_file=f"plots/input_consumption_{postal_code}.html",
        y_columns=["consumption", "consumption_filtered"],
        y_title="kWh"
    )
    plot_daily_load_curves_with_centroids_to_pdf(
        df=consumption.filter(pl.col("postalcode") == postal_code),
        pdf_path=f"plots/daily_load_curves_all_{postal_code}.pdf"
    )
'''

# ================================
# 4. Prepare Data for Clustering
# ================================
# Define aggregation window (hours)
N_HOURS = 1

# Compute intraday consumption percentages
consumption_long = consumption.join(
    consumption.group_by(["postalcode", "date"]).agg(
        (pl.col("consumption").mean() * 24).alias("daily_consumption")
    ),
    on=["postalcode", "date"]
).with_columns([
    (((pl.col("localtime").dt.hour() / N_HOURS).floor()) * N_HOURS).alias("hour")
]).group_by(["postalcode", "hour", "date", "daily_consumption"]).agg(
    (pl.col("consumption").mean() * N_HOURS).alias("consumption")
).with_columns(
    (pl.col("consumption") * 100 / pl.col("daily_consumption")).alias("consumption_percentage")
)

# Transform the dataset to wide format
consumption_wide = consumption_long.sort(["postalcode", "hour", "date"]).select([
    "postalcode", "date", "hour", "consumption_percentage"
]).pivot(index=["postalcode", "date"], columns="hour", values="consumption_percentage").to_pandas()
consumption_wide.set_index(["postalcode", "date"], inplace=True)

# ================================
# 5. Apply Scaling
# ================================
# Scale the data using different methods
scalers = {
    "NoScaling": consumption_wide,
    "MinMaxScaling": pd.DataFrame(
        MinMaxScaler().fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    ),
    "ZNormScaling": pd.DataFrame(
        StandardScaler().fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    )
}

# ================================
# 6. Perform Clustering
# ================================
# Range of cluster numbers to evaluate
CLUSTER_RANGE = range(2, 10)

for scaling_type, scaled_data in scalers.items():
    silhouette_scores = []
    clustering_data = scaled_data.dropna()
    clustering_index = clustering_data.index.to_frame(index=False)
    clustering_data.reset_index(drop=True, inplace=True)

    for n_clusters in CLUSTER_RANGE:
        print(f"Clustering ({scaling_type}) with {n_clusters} clusters")
        
        # Perform hierarchical clustering
        clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", compute_distances=True
        )
        cluster_labels = clustering_model.fit_predict(clustering_data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(clustering_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        # Plot dendrogram
        plt.figure(figsize=(15, 8))
        plt.title(f"Dendrogram ({scaling_type}, {n_clusters} clusters)")
        plot_dendrogram(clustering_model, truncate_mode="lastp", p=n_clusters)
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        plt.savefig(f"plots/dendrogram_{scaling_type}_{n_clusters}.pdf", format="pdf")

        # Plot daily load curves with centroids
        plot_daily_load_curves_with_centroids_to_pdf(
            df=(consumption.select(pl.all().exclude("cluster"))
                .join(
                    consumption.group_by(["postalcode", "date"]).agg(
                        (pl.col("consumption_filtered").mean() * 24).alias("daily_consumption")
                    ),
                    on=["postalcode", "date"]
                ).with_columns(
                    (pl.col("consumption_filtered") * 100 / pl.col("daily_consumption")).alias("consumption_filtered")
                ).join(
                    pl.DataFrame(
                        pd.concat([
                            clustering_index.reset_index(drop=True),
                            pd.DataFrame(cluster_labels, columns=["cluster"])
                        ], axis=1)
                    ).with_columns(pl.col("date").cast(pl.Date)),
                    on=["postalcode", "date"]
                )
            ),
            pdf_path=f"plots/load_curves_{scaling_type}_{n_clusters}.pdf",
            add_in_title=scaling_type
        )

    # Plot silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(CLUSTER_RANGE, silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Analysis ({scaling_type})")
    plt.savefig(f"plots/silhouette_{scaling_type}.pdf", format="pdf")