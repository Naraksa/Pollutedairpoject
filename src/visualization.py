import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Determine the static images directory relative to this file
THIS_DIR = Path(__file__).parent        # e.g., webapp/src
STATIC_IMAGES = THIS_DIR.parent / 'static' / 'images'
STATIC_IMAGES.mkdir(parents=True, exist_ok=True)

# Save a plot to the static images directory
def _save_fig(fig, name):
    path = STATIC_IMAGES / name
    fig.savefig(path, bbox_inches='tight')  
    plt.close(fig)

# Plot PM2.5 and PM10 median values over time and save to file
def plot_time_series(pm25, pm10):
    """Plot timse series for PM2.5 and PM10 median values."""
    if 'date' not in pm25.columns or 'median' not in pm25.columns:
        raise ValueError("PM2.5 DataFrame must contain 'date' and 'median' columns.")
    if 'date' not in pm10.columns or 'median' not in pm10.columns:
        raise ValueError("PM10 DataFrame must contain 'date' and 'median' columns.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pm25['date'], pm25['median'], label='PM2.5', color='green')
    ax.plot(pm10['date'], pm10['median'], label='PM10', color='orange')
    ax.set(title='PM2.5 and PM10 Over Time', xlabel='Date', ylabel='Median Concentration')
    ax.legend()
    ax.grid(True)
    _save_fig(fig, 'pm25_pm10_time_series.png')

# Plot the distribution of PM2.5 and PM10 median values using boxplot
def plot_boxplot(df):
    """Plot boxplot for PM2.5 and PM10 median values."""
    if 'pollutant' not in df.columns or 'median' not in df.columns:
        raise ValueError("DataFrame must contain 'pollutant' and 'median' columns.")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='pollutant', y='median', data=df, ax=ax)
    ax.set(title='Distribution of PM2.5 vs PM10', xlabel='Pollutant', ylabel='Median Concentration')
    ax.grid(True)
    _save_fig(fig, 'pm_boxplot.png')

# Plot scatter plot to show correlation between PM2.5 and PM10 on same dates
def plot_correlation(pm25, pm10):
    """Plot correlation between PM2.5 and PM10 median values."""
    if 'date' not in pm25.columns or 'median' not in pm25.columns:
        raise ValueError("PM2.5 DataFrame must contain 'date' and 'median' columns.")
    if 'date' not in pm10.columns or 'median' not in pm10.columns:
        raise ValueError("PM10 DataFrame must contain 'date' and 'median' columns.")

    merged = pd.merge(
        pm25[['date', 'median']], pm10[['date', 'median']],
        on='date', suffixes=('_pm25', '_pm10')
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x='median_pm25', y='median_pm10', data=merged, ax=ax)
    ax.set(title='Correlation Between PM2.5 and PM10', xlabel='PM2.5 Median', ylabel='PM10 Median')
    ax.grid(True)
    _save_fig(fig, 'pm_correlation.png')

# Plot actual vs predicted values for Linear Regression
def plot_predictions(y_test_lr, y_pred_lr):
    """Plot actual vs predicted values for Linear Regression."""
    if len(y_test_lr) != len(y_pred_lr):
        raise ValueError("y_test_lr and y_pred_lr must have the same length.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y_test_lr.values, label='Actual', color='black')
    ax.plot(y_pred_lr, label='Predicted (Linear)', color='blue')
    ax.set(title="Linear Regression Predictions", xlabel="Time Index", ylabel="PM Value")
    ax.legend()
    ax.grid(True)
    _save_fig(fig, 'model_comparison.png')

# Plot the future predictions for both Linear Regression and Random Forest
def plot_future_prediction(df, future_df_lr):
    """Plot future predictions for Linear Regression."""
    if 'date' not in df.columns or 'median' not in df.columns:
        raise ValueError("Historical DataFrame must contain 'date' and 'median' columns.")
    if 'date' not in future_df_lr.columns or 'median' not in future_df_lr.columns:
        raise ValueError("Future DataFrame must contain 'date' and 'median' columns.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['date'], df['median'], label='Historical', color='gray')
    ax.plot(future_df_lr['date'], future_df_lr['median'], label='Linear Forecast', color='blue')
    ax.set(xlabel='Date', ylabel='Predicted PM', title='Future PM Predictions (7 days)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    _save_fig(fig, 'future_forecast.png')