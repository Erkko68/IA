import os
import warnings

import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer
from sklearn.decomposition import PCA

from utils import *

warnings.filterwarnings("ignore")

PLOT_DIR = "../../plots"
os.makedirs(PLOT_DIR,exist_ok=True)

# ================================
# 1. Load and Preprocess Data
# ================================
# Read electricity consumption dataset
consumption = pl.read_parquet("../../data/electricity_consumption.parquet")

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

# ================================
# 3. Filter Outliers
# ================================
# Define Z-normalization thresholds
MAX_Z_THRESHOLD = 4
MIN_Z_THRESHOLD = -2

# Filter rows with 0 consumption (preventing one extra cluster just for 0 consumption patterns)
consumption = consumption.filter(pl.col("consumption") > 30.0)

# Filter consumption data based on Z-normalization and rolling quantile thresholds
consumption = consumption.with_columns(
    pl.when(
        (pl.col("z_norm") < MAX_Z_THRESHOLD) &
        (pl.col("z_norm") > MIN_Z_THRESHOLD) &
        (pl.col("consumption") > (pl.col("rolling_q10") * 0.8))
    ).then(pl.col("consumption")).otherwise(np.nan).alias("consumption_filtered")
)

consumption.write_parquet("../../data/consumption_filtered.parquet")

# ================================
# 4. Prepare Data for Clustering
# ================================
# Define aggregation window (hours)
N_HOURS = 3

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

# Create a copy of the DataFrame before saving
consumption_wide_copy = consumption_wide.copy()
# Save the copy as a Parquet file
consumption_wide_copy.to_parquet("../../data/consumption_wide.parquet")

# ================================
# 5. Apply Scaling
# ================================
# Scale the data using different methods
scalers = {
    "MinMaxScaling": pd.DataFrame(
        MinMaxScaler().fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    ),
    "ZNormScaling": pd.DataFrame(
        StandardScaler().fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    ),
    # Using robust scaling gives no major differences (this means the data doesn't have many outliners)
    "RobustScaling": pd.DataFrame(
        RobustScaler().fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    ),
    #
    # The same result as MinMaxScaling
    "MaxAbsScaling": pd.DataFrame(
        MaxAbsScaler().fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    ),
    "PowerTransformerScaling": pd.DataFrame(
        PowerTransformer(method="yeo-johnson").fit_transform(consumption_wide),
        columns=consumption_wide.columns,
        index=consumption_wide.index
    )
}


# ================================
# 6. Perform K-Means Clustering with PCA and t-SNE Validation
# ================================
# Range of cluster numbers to evaluate
'''
CLUSTER_RANGE = range(2, 8)

for scaling_type, scaled_data in scalers.items():
    os.makedirs(f"{PLOT_DIR}/kmeans/{scaling_type}", exist_ok=True)
    silhouette_scores = []
    clustering_data = scaled_data.dropna()
    clustering_index = clustering_data.index.to_frame(index=False)
    clustering_data.reset_index(drop=True, inplace=True)
    
    # PCA Loop: Define PCA components to explore
    PCA_COMPONENTS = [2,3,4,5]  # Adjust based on your analysis needs
    pca_silhouette_scores = []

    for n_components in PCA_COMPONENTS:
        print(f"Performing PCA ({scaling_type}) with {n_components} components")
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca_data = pca.fit_transform(clustering_data)
        pca_dir = f"{PLOT_DIR}/kmeans/{scaling_type}/pca_{n_components}"
        os.makedirs(pca_dir, exist_ok=True)
        
        silhouette_scores_pca = []
        
        for n_clusters in CLUSTER_RANGE:
            print(f"K-Means Clustering ({scaling_type}, PCA={n_components}) with {n_clusters} clusters")
            
            # Perform K-Means clustering
            clustering_model = KMeans(
                n_clusters=n_clusters, init='k-means++', algorithm="lloyd",
                max_iter=100, n_init=15, random_state=42
            )
            cluster_labels = clustering_model.fit_predict(pca_data)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(pca_data, cluster_labels)
            silhouette_scores_pca.append(silhouette_avg)
            
            # Plot daily load curves with centroids
            plot_daily_load_curves_with_centroids_to_png(
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
                png_path=f"{pca_dir}/load_curves_{n_clusters}.png",
                add_in_title=f"K-Means {scaling_type}, PCA={n_components}"
            )
            
            # Perform t-SNE for validation
            print(f"Performing t-SNE Visualization ({scaling_type}, PCA={n_components}) for {n_clusters} clusters")
            tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=2000)
            tsne_results = tsne.fit_transform(pca_data)
            
            # Create a DataFrame for t-SNE results
            tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE 1', 't-SNE 2'])
            tsne_df['Cluster'] = cluster_labels
            
            # Plot t-SNE results
            plt.figure(figsize=(8, 8))
            sns.scatterplot(
                x='t-SNE 1', y='t-SNE 2', hue='Cluster', data=tsne_df,
                palette='tab10', legend='full', alpha=0.8
            )
            plt.title(f't-SNE Visualization ({scaling_type}, PCA={n_components}, {n_clusters} Clusters)')
            plt.savefig(f"{pca_dir}/tsne_{n_clusters}.png", dpi=300)
            plt.close()
        
        # Average silhouette score across cluster ranges for the current PCA
        avg_silhouette_score = sum(silhouette_scores_pca) / len(silhouette_scores_pca)
        pca_silhouette_scores.append(avg_silhouette_score)

        # Plot silhouette scores for PCA
        plt.figure(figsize=(10, 5))
        plt.plot(CLUSTER_RANGE, silhouette_scores_pca, marker="o")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette Analysis ({scaling_type}, PCA={n_components})")
        plt.savefig(f"{pca_dir}/silhouette.png", dpi=300)
'''

# ================================
# 7. Perform Hierarchical Clustering with PCA and Dendrogram Validation
# ================================
# Range of cluster numbers to evaluate
CLUSTER_RANGE = range(2, 8)

for scaling_type, scaled_data in scalers.items():
    os.makedirs(f"{PLOT_DIR}/hierarchical/{scaling_type}", exist_ok=True)
    silhouette_scores = []
    clustering_data = scaled_data.dropna()
    clustering_index = clustering_data.index.to_frame(index=False)
    clustering_data.reset_index(drop=True, inplace=True)
    
    # PCA Loop: Define PCA components to explore
    PCA_COMPONENTS = [2, 3, 4, 5]  # Adjust based on your analysis needs
    pca_silhouette_scores = []

    for n_components in PCA_COMPONENTS:
        print(f"Performing PCA ({scaling_type}) with {n_components} components")
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca_data = pca.fit_transform(clustering_data)
        pca_dir = f"{PLOT_DIR}/hierarchical/{scaling_type}/pca_{n_components}"
        os.makedirs(pca_dir, exist_ok=True)
        
        silhouette_scores_pca = []
        
        for n_clusters in CLUSTER_RANGE:
            print(f"Hierarchical Clustering ({scaling_type}, PCA={n_components}) with {n_clusters} clusters")
            
            # Perform Agglomerative Hierarchical clustering
            clustering_model = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward", compute_distances=True
            )
            cluster_labels = clustering_model.fit_predict(pca_data)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(pca_data, cluster_labels)
            silhouette_scores_pca.append(silhouette_avg)
            
            # Plot dendrogram
            plt.figure(figsize=(15, 8))
            plt.title(f"Hierarchical Clustering Dendrogram - {scaling_type}, PCA={n_components}")
            plot_dendrogram(clustering_model, truncate_mode='lastp', p=n_clusters)
            plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            plt.ylabel("Distance based on linkage")
            plt.savefig(f"{pca_dir}/dendrogram_{n_clusters}.png", dpi=300)
            plt.close()

            # Plot daily load curves with cluster labels
            plot_daily_load_curves_with_centroids_to_png(
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
                png_path=f"{pca_dir}/load_curves_{n_clusters}.png",
                add_in_title=f"Hierarchical {scaling_type}, PCA={n_components}"
            )
        
        # Average silhouette score across cluster ranges for the current PCA
        avg_silhouette_score = sum(silhouette_scores_pca) / len(silhouette_scores_pca)
        pca_silhouette_scores.append(avg_silhouette_score)

        # Plot silhouette scores for PCA
        plt.figure(figsize=(10, 5))
        plt.plot(CLUSTER_RANGE, silhouette_scores_pca, marker="o")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette Analysis ({scaling_type}, PCA={n_components})")
        plt.savefig(f"{pca_dir}/silhouette.png", dpi=300)

# Dendrogram plot function
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
