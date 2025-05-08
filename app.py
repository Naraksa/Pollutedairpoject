import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
from src.send_alert import send_air_quality_alerts

from src.data_preprocessing import (
    load_data,
    label_pollutants,
    remove_outliers,
    split_by_pollutant
)
from src.model_training import (
    create_features,
    train_model,
    predict_future,
    classify_and_alert
)
from src.visualization import (
        plot_time_series,
        plot_boxplot,
        plot_correlation,
        plot_predictions,
        plot_future_prediction
    )

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predictions = None

    if request.method == 'POST':
        # Load data & model
        df = load_data(os.path.join(BASE_DIR, 'src', 'cleaned_data.csv'))
        df = label_pollutants(df)
        df = remove_outliers(df)
        pm25, _ = split_by_pollutant(df)

        model = joblib.load(os.path.join(BASE_DIR, 'src', 'best_model.pkl'))

        # Prepare features and forecast
        feat_df = create_features(pm25, lags=3)
        future_df = predict_future(feat_df, model, days=7)

        # Annotate with combined status + recommendation
        future_df[['status', 'recommendation']] = future_df['median']\
            .apply(lambda x: pd.Series(classify_and_alert(x)))

        predictions = future_df.to_dict(orient='records')
        send_air_quality_alerts(future_df, model_name="PA Prediction")

    return render_template('predict.html', predictions=predictions)


@app.route('/visualize', methods=['GET'])
def visualize():
    # Load and split data
    df = load_data(os.path.join(BASE_DIR, 'src', 'cleaned_data.csv'))
    df = label_pollutants(df)
    df = remove_outliers(df)
    pm25, pm10 = split_by_pollutant(df)

    # Generate static charts
    plot_time_series(pm25, pm10)
    plot_boxplot(df)
    plot_correlation(pm25, pm10)

    # Train on PM2.5 to get actual vs. predicted
    feat_df = create_features(pm25, lags=3)
    lr_model, lr_rmse, lr_mse, lr_r2, lr_mae, rmse_percentage, y_test_lr, y_pred_lr = train_model(feat_df, model_type='linear')

    # Plot model comparisons
    plot_predictions(y_test_lr, y_pred_lr)

    # Plot 7-day forecasts
    future_lr = predict_future(feat_df, lr_model, days=7)
    plot_future_prediction(pm25, future_lr)

    return render_template('visualize.html')


if __name__ == '__main__':
    app.run(debug=True)
