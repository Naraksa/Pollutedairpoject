import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def label_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    last_date = None
    toggle = True

    for _, row in df.iterrows():
        current_date = row['date']
        if current_date != last_date:
            toggle = True
            last_date = current_date

        labels.append("PM2.5" if toggle else "PM10")
        toggle = not toggle

    df['pollutant'] = labels
    return df

def remove_outliers(df: pd.DataFrame, column='median', threshold=150) -> pd.DataFrame:
    """Remove extreme pollution values that skew the model."""
    df = df[df[column] <= threshold].reset_index(drop=True)
    return df

def filter_low_counts(df: pd.DataFrame, min_count=100) -> pd.DataFrame:
    """Filter out rows where the number of measurements is too low (too noisy)."""
    df = df[df['count'] >= min_count].reset_index(drop=True)
    return df

def split_by_pollutant(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm25 = df[df['pollutant'] == 'PM2.5'].reset_index(drop=True)
    pm10 = df[df['pollutant'] == 'PM10'].reset_index(drop=True)
    return pm25, pm10

def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
