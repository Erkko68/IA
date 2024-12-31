import plotly.express as px
import plotly.io as pio
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colormaps
from matplotlib.dates import DateFormatter
import numpy as np
import random
from scipy.cluster.hierarchy import dendrogram
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd

# Function to evaluate regression models
def evaluate_regression_model(model_name, preprocessor, model, hyperparams_ranges, cross_val,
                              X_train, y_train, X_test, y_test):
    # Create and evaluate the pipeline with the given model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=hyperparams_ranges,
                               cv=cross_val, n_jobs=-1, verbose=1,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_  # Get the best estimator from the grid search
    y_pred = best_pipeline.predict(X_test)

    best_params = grid_search.best_params_

    # Calculate Mean Squared Error (MSE) and Root MSE (RMSE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    ## Coefficient of Variation of the RMSE
    cvrmse = rmse / y_test.mean()

    print(f"Best parameters: {best_params}")
    print(f"{model_name} - Root Mean Squared Error: {round(rmse, 2)}")
    print(f"{model_name} - Coefficient of Variation of the Root Mean Squared Error: {round(cvrmse * 100, 2)} %")

    return best_pipeline


def plot_regression_results(pipeline, df, filename, postal_code, model_name, hours=96, npred=1):
    """
    Plot and save actual vs predicted consumption for a specific postal code into a single PDF.

    Parameters:
        pipeline: Trained model pipeline with a `.predict` method.
        df (DataFrame): Data containing postal codes, 'localtime', and 'consumption'.
        filename (str): The file name to store the plots.
        postal_code (str/int): The postal code to filter data.
        model_name (str): The name of the model used for predictions.
        hours (int): Number of hours to plot.
        npred (int): Number of random plots to generate.
    """
    # Filter data by postal code
    postal_filter = df['postalcode'] == postal_code
    filtered_df = df[postal_filter].copy()

    # Drop irrelevant columns for prediction
    X_test_filtered = filtered_df.drop(["localtime", "consumption"], axis=1)

    # Predict consumption and add it to the dataframe
    y_pred = pipeline.predict(X_test_filtered)
    filtered_df["predicted"] = y_pred

    # Create a single PDF file to store all plots
    with PdfPages(filename) as pdf:
        for i in range(npred):
            # Randomly select a continuous range of data
            max_start = len(filtered_df) - hours
            if max_start <= 0:
                print("Not enough data to plot the specified number of hours.")
                return

            rand_start = random.randint(0, max_start)
            df_slice = filtered_df.iloc[rand_start:rand_start + hours]

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(df_slice["localtime"], df_slice["predicted"], label='Predicted', marker='x', linestyle='--',
                     markersize=3)
            plt.plot(df_slice["localtime"], df_slice["consumption"], label='Actual', marker='o', linestyle='-',
                     markersize=3)

            plt.title(f"Actual vs Predicted Consumption\nPostal Code: {postal_code} ({model_name})")
            plt.xlabel("Time")
            plt.ylabel("Consumption")
            plt.legend()
            plt.grid(True)

            # Format x-axis for date display
            ax = plt.gca()  # Get current axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Add the current figure to the PDF
            pdf.savefig()
            plt.close()

        # Add metadata to the PDF
        d = pdf.infodict()
        d['Title'] = f"Prediction Results for Postal Code {postal_code} ({model_name})"
        d['Author'] = 'Your Name'
        d['Subject'] = 'Actual vs Predicted Consumption Comparison'
        d['Keywords'] = 'Consumption, Prediction, Regression, Model'
        d['CreationDate'] = pd.Timestamp.now()