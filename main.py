import pandas as pd
from src.data_preprocessing import (
    load_data,
    label_pollutants,
    remove_outliers,
    split_by_pollutant,
    save_data
)
from src.visualization import (
    plot_time_series,
    plot_boxplot,
    plot_correlation,
    plot_predictions,
    plot_future_prediction
)
from src.model_training import (
    create_features,
    train_model,
    save_model,
    predict_future,
    classify_and_alert,
    evaluate_model_cv
)
from src.send_alert import send_air_quality_alerts

print("Loading and preprocessing the data...")
df = load_data("src/data.csv")
df = label_pollutants(df)
df = remove_outliers(df)
save_data(df, "src/cleaned_data.csv")

# Reload cleaned data and split pollutants
df = load_data("src/cleaned_data.csv")
df = label_pollutants(df)
df = remove_outliers(df)
pm25, pm10 = split_by_pollutant(df)

# Visualizations
plot_time_series(pm25, pm10)
plot_boxplot(df)
plot_correlation(pm25, pm10)

# Feature engineering
df_feat = create_features(pm25, lags=3)
print("Feature DataFrame shape:", df_feat.shape)
print("Columns:", df_feat.columns.tolist())

# Train models
print("Training Linear Regression model...")
lr_model, lr_rmse, lr_mae, lr_mse, lr_r2, rmse_percentage_lr, y_test_lr, y_pred_lr= train_model(df_feat, model_type='linear')

print("Evaluating Linear Regression model...")
evaluate_model_cv(lr_model, df_feat)
# Plot actual vs. predicted
plot_predictions(y_test_lr, y_pred_lr)  # Ensure y_test_rf and y_pred_rf are defined

# Forecast and plot future
future_df_lr = predict_future(pm25[['date', 'median']].copy(), lr_model, days=7)
plot_future_prediction(pm25, future_df_lr)
print("Future predictions plotted successfully.")

# Show text inference using the tuned alerts
print("\n--- Air Quality Forecast (PM2.5) with Classification & Recommendations ---")
for idx, row in future_df_lr.iterrows():
    status, recommendation = classify_and_alert(row['median'])
    date_str = row['date'].strftime('%Y-%m-%d')
    print(f"{date_str}: PM2.5 = {row['median']:.2f} → {status} ({recommendation})")

# Prepare full forecast tables with status & recommendation
def annotate(df_preds):
    df = df_preds.copy()
    df[['status', 'recommendation']] = df['median']\
        .apply(lambda x: pd.Series(classify_and_alert(x)))
    return df

future_lr = annotate(future_df_lr)

#Send alerts for the air levels
#send_air_quality_alerts(future_df_lr, model_name="Linear Regression")
