from src.telegram_noti import send_telegram_message
from src.model_training import classify_and_alert
from dotenv import load_dotenv
import os
# Your Telegram Bot credentials (replace with your real ones)
load_dotenv()
TELEGRAM_TOKEN = os.getenv("MY_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("MY_TELEGRAM_ID")

# Annotate and send alerts for a given forecast DataFrame
def send_air_quality_alerts(forecast_df, model_name="Model"):
    for _, row in forecast_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        pm_value = row['median']
        status, recommendation = classify_and_alert(pm_value)

        message = f"*{model_name} Forecast {date_str}*\n"
        message += f"PM2.5: *{pm_value:.2f}* µg/m³\n"
        message += f"Status: *{status}*\n{recommendation}"

        print(message)  # Print locally

        # Only send alerts for concerning statuses
        if status not in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
            send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
        else:
            send_telegram_message(TELEGRAM_TOKEN,TELEGRAM_CHAT_ID, message)