import plotly.express as px
import plotly.io as pio
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colormaps
import numpy as np
from scipy.cluster.hierarchy import dendrogram

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


def plot_daily_load_curves_with_centroids_to_png(
    df: pl.DataFrame, 
    png_path: str, 
    add_in_title: str = ""
):
    """
    Plots daily load curves with centroids for each cluster from a Polars DataFrame and saves the plot to a PNG.

    Parameters:
        df (pl.DataFrame): A Polars DataFrame containing 'date', 'hour', 'cluster', and 'consumption' columns.
        png_path (str): The file path where the PNG should be saved.
        add_in_title (str, optional): A string to append to the title of the plot. Defaults to "".
    """
    # Ensure the 'cluster' column exists; if not, create it filled with 1's
    if "cluster" not in df.columns:
        df = df.with_columns(pl.lit(1).alias("cluster"))

    # Ensure the DataFrame is sorted by date and hour
    df = df.sort(["date", "postalcode", "hour"])

    # Group by 'date' and 'cluster', and collect consumption data for each day
    daily_curves = df.group_by(["date", "postalcode", "cluster"]).agg(pl.col("consumption_filtered"))

    # Create a centroid DataFrame by averaging consumption for each hour and cluster
    centroids = (
        df.group_by(["hour", "cluster"])
        .agg(pl.col("consumption_filtered").drop_nans().mean().alias("consumption_filtered"))
        .sort(["hour", "cluster"])
        .pivot(index=["cluster"], on="hour")
    )

    # Convert to pandas for easy plotting with matplotlib
    daily_curves_pandas = daily_curves.to_pandas()
    centroids_pandas = centroids.to_pandas()

    # Plot setup
    plt.figure(figsize=(10, 6))
    unique_clusters = df["cluster"].unique().to_list()
    colors = colormaps["tab10"]  # Use the updated way to access colormaps

    # Plot individual daily load curves with low opacity
    for _, row in daily_curves_pandas.iterrows():
        date = row["date"]
        cluster = row["cluster"]
        consumption = row["consumption_filtered"]
        plt.plot(
            range(len(consumption)), 
            consumption, 
            alpha=0.1, 
            lw=0.005, 
            color=colors(cluster / len(unique_clusters)),
            label=None
        )

    # Plot centroids with bold lines
    for cluster in unique_clusters:
        centroid = centroids_pandas[
            centroids_pandas["cluster"] == cluster
        ].drop("cluster", axis=1).iloc[0].to_numpy()
        plt.plot(
            range(len(centroid)), 
            centroid, 
            linewidth=2.5, 
            label=f"Cluster {cluster}", 
            color=colors(cluster / len(unique_clusters))
        )

    # Add labels and legend
    plt.xlabel("Hour of Day")
    plt.ylabel("Consumption")
    plt.title(f"Daily Load Curves {add_in_title} Clustering")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a PNG file
    plt.savefig(png_path, dpi=300)
    plt.close()



def weather_plotter(weather, html_file):
    # Assuming 'weather' is a Polars DataFrame and needs to be sorted by time
    weather = weather.sort('time')

    # Convert the sorted Polars DataFrame to Pandas for compatibility with Plotly
    weather_df = weather.to_pandas()

    # Create a separate interactive line plot for each postal code with Plotly
    figs = []
    postal_codes = weather_df['postalcode'].unique()

    for postal_code in postal_codes:
        subset = weather_df[weather_df['postalcode'] == postal_code]
        fig = px.line(
            subset,
            x='time',
            y=['airtemperature'],
            title=f'Postal Code: {postal_code}',
            labels={'airtemperature': 'Air Temperature'},
        )
        fig.update_layout(xaxis_title='Time', yaxis_title='Air Temperature')
        figs.append(fig)

    # Save each plot to an HTML file and show it
    with open(html_file, "w") as f:
        for fig in figs:
            f.write(pio.to_html(fig, full_html=False))