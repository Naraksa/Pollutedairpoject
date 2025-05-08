import requests

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print("❌ Failed to send message:", response.json())
        return response.ok
    except Exception as e:
        print("❌ Error sending Telegram message:", e)
        return False