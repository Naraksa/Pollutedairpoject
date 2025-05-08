import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def create_features(df, lags=3):
    df = df.copy()
    df = df.sort_values('date')

    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['median'].shift(i)

    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    return df.dropna().reset_index(drop=True)

def train_model(df, model_type='linear'):
    features = [col for col in df.columns if col.startswith('lag_')] + ['dayofweek', 'month']
    X = df[features]
    y = df['median']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    if model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError("model_type must be 'linear'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse_percentage = rmse / 500 * 100  

    print(f"{model_type.upper()} Model Performance:")
    print(f"RMSE: {rmse:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}")
    print(f"RMSE Percentage: {rmse_percentage:.2f}%")

    return model, rmse, mse, r2, mae, rmse_percentage, y_test, y_pred

def save_model(model, scaler=None, model_filename='model.pkl', scaler_filename='scaler.pkl'):
    joblib.dump(model, model_filename)
    if scaler:
        joblib.dump(scaler, scaler_filename)


def predict_future(df, model, scaler=None, days=1, lags=3):
    future_preds = []
    df = df.copy().sort_values('date')

    for _ in range(days):
        latest = df.iloc[-lags:].copy()

        row = {f'lag_{i}': latest.iloc[-i]['median'] for i in range(1, lags + 1)}
        last_date = df['date'].max()
        row['dayofweek'] = (last_date + pd.Timedelta(days=1)).dayofweek
        row['month'] = (last_date + pd.Timedelta(days=1)).month

        features = pd.DataFrame([row])

        prediction = model.predict(features)[0]

        if scaler:
            prediction = scaler.inverse_transform(np.array([[prediction]])).ravel()[0]

        new_row = {
            'date': last_date + pd.Timedelta(days=1),
            'median': prediction
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        future_preds.append(new_row)

    return pd.DataFrame(future_preds)

def evaluate_model_cv(model, df):
    features = [col for col in df.columns if col.startswith('lag_')] + ['dayofweek', 'month']
    X = df[features]
    y = df['median']

    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = (-scores) ** 0.5
    avg_rmse = rmse_scores.mean()
    print(f"Average Cross-Validation RMSE: {avg_rmse:.2f}")
    return avg_rmse

# Combined classification and alert function
def classify_and_alert(pm_value):
    if pm_value <= 12:
        return "Good", "🟢 Good – Air quality is satisfactory."
    elif pm_value <= 35.4:
        return "Moderate", "🟡 Moderate – Acceptable, but sensitive individuals should be cautious."
    elif pm_value <= 55.4:
        return "Unhealthy for Sensitive Groups", "🟠 Limit outdoor exertion."
    elif pm_value <= 150.4:
        return "Unhealthy", "🔴 Everyone may begin to experience health effects."
    elif pm_value <= 250.4:
        return "Very Unhealthy", "🟣 Serious effects for everyone."
    else:
        return "Hazardous", "⚫️ Avoid all outdoor activity."