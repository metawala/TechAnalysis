# # Automated NSE Stock Technical Analysis with Chart Image Download
# # ===============================================================
# # Requirements:
# #   - Live chart image download (hourly & 15 min, optional daily/weekly)
# #   - Add EMA indicators, volume, RSI
# #   - Technical analysis via both image and AI/TA-Lib
# #   - Output a detailed human-readable summary
# #   - Flask web interface

# # Updated to use Falcon-7B-Instruct for free AI analysis

# import os
# import datetime
# import requests
# from PIL import Image
# import pandas as pd
# import numpy as np
# import talib
# from flask import Flask, request
# from transformers import pipeline

# app = Flask(__name__)


# # ---- FUNCTION: Download Live Chart Images ----
# def fetch_chart_image(symbol, interval):
#     payload = {
#         "symbol": symbol,
#         "interval": interval,
#         "studies": [
#             { "name": "Volume", "overrides": {} },
#             { "name": "Moving Average Exponential", "inputs": { "length": 20 } },
#             { "name": "Relative Strength Index", "inputs": { "length": 14 } }    
#             ],
#             "theme": "dark",
#             "width": 600,
#             "height": 600
#     }

#     headers = {
#         "x-api-key": API_KEY,
#         "Content-Type": "application/json"
#     }

#     response = requests.post("https://api.chart-img.com/v2/tradingview/advanced-chart", json=payload, headers=headers)

#     if response.status_code == 200:
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"static/{symbol.replace(':', '_')}_{interval}_{timestamp}.png"
#         os.makedirs("static", exist_ok=True)
#         with open(filename, 'wb') as f:
#             f.write(response.content)
#         return filename
#     else:
#         raise Exception(f"Chart API failed: {response.status_code} - {response.text}")


# # ---- FUNCTION: Technical Analysis using TA-Lib ----
# def perform_technical_analysis(df):
#     df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
#     df['EMA50'] = talib.EMA(df['close'], timeperiod=50)
#     df['EMA100'] = talib.EMA(df['close'], timeperiod=100)
#     df['EMA200'] = talib.EMA(df['close'], timeperiod=200)
#     df['RSI'] = talib.RSI(df['close'], timeperiod=14)

#     latest = df.iloc[-1]

#     sentiment = "Bullish" if latest['EMA20'] > latest['EMA50'] else "Bearish"
#     rsi_signal = "Overbought" if latest['RSI'] > 70 else ("Oversold" if latest['RSI'] < 30 else "Neutral")

#     summary = {
#         "sentiment": sentiment,
#         "rsi": float(round(latest['RSI'], 2)),
#         "rsi_signal": rsi_signal,
#         "current_price": float(round(latest['close'], 2)),
#         "ema20": float(round(latest['EMA20'], 2)),
#         "ema50": float(round(latest['EMA50'], 2)),
#         "ema100": float(round(latest['EMA100'], 2)),
#         "ema200": float(round(latest['EMA200'], 2)),
#         "entry": float(round(latest['close'], 2)) if sentiment == "Bullish" else None,
#         "stop_loss": float(round(latest['EMA200'], 2)),
#         "near_term_target": float(round(latest['EMA20'] + 20, 2)),
#         "mid_term_target": float(round(latest['EMA50'] + 50, 2)),
#         "long_term_target": float(round(latest['EMA100'] + 100, 2))
#     }
#     return summary

# # ---- FUNCTION: Real AI-Based Analysis with Falcon ----
# def ai_based_technical_insight(df):
#     last_price = df['close'].iloc[-1]
#     trend = np.polyfit(range(len(df['close'][-20:])), df['close'][-20:], 1)[0]
#     ma = df['close'].rolling(window=20).mean().dropna().tail(10).tolist()

#     # Format prompt
#     prompt = (
#         f"You are a financial analyst. Based on the 20-period moving average values: {ma},\n"
#         f"and the last price {round(last_price, 2)}, provide:\n"
#         f"- Sentiment (bullish or bearish)\n"
#         f"- Good entry and exit points\n"
#         f"- Near, mid, and long-term price targets\n"
#         f"- Brief explanation why.\n"
#     )

#     generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")
#     response = generator(prompt, max_new_tokens=300)[0]['generated_text']
#     return {
#         "sentiment": "Bullish" if trend > 0 else "Bearish",
#         "commentary": response
#     }

# # ---- FUNCTION: Summary and Comparison ----
# def compare_results(ta_summary, ai_summary):
#     return {
#         "technical_analysis": ta_summary,
#         "ai_analysis": ai_summary,
#         "comparison": "Agree" if ta_summary['sentiment'] == ai_summary['sentiment'] else "Disagree"
#     }

# # ---- FLASK ROUTES ----
# @app.route('/')
# def index():
#     return '''<form method="get" action="/analyze">
#                 NSE/BSE Symbol: <input type="text" name="symbol" value="GANESHHOUC">
#                 <input type="submit" value="Analyze">
#               </form>'''

# @app.route('/analyze')
# def analyze():
#     symbol = request.args.get("symbol", default="GANESHHOUC").upper()
#     full_symbol = f"NSE:{symbol}"

#     chart_files = []
#     for interval in INTERVALS:
#         try:
#             chart_files.append(fetch_chart_image(full_symbol, interval))
#         except Exception as e:
#             return f"Error fetching chart: {e}"

#     # Simulated data (replace with live NSE source)
#     dummy_data = pd.DataFrame({
#         "close": np.random.normal(1000, 20, 100),
#         "volume": np.random.randint(1000, 5000, 100)
#     })

#     ta_result = perform_technical_analysis(dummy_data)
#     ai_result = ai_based_technical_insight(dummy_data)
#     summary = compare_results(ta_result, ai_result)

#     html = f"""
#     <h2>Chart Snapshots</h2>
#     {''.join([f'<img src="/{img}" width="600"><br>' for img in chart_files])}
#     <h3>Technical Analysis</h3>
#     <pre>{ta_result}</pre>
#     <h3>AI Analysis</h3>
#     <pre>{ai_result['commentary']}</pre>
#     <h3>Summary</h3>
#     <pre>{summary['comparison']} between technical and AI analysis</pre>
#     """
#     return html

# if __name__ == "__main__":
#     app.run(debug=True)

# # ---- OPTIONAL DAILY / WEEKLY ----
# # To enable, just append to INTERVALS:
# # INTERVALS += ['1d', '1W']


# Automated NSE Stock Technical Analysis with Chart Image Download
# ===============================================================
# Requirements:
#   - Live chart image download (hourly & 15 min, optional daily/weekly)
#   - Add EMA indicators, volume, RSI
#   - Technical analysis via both image and AI/TA-Lib
#   - Output a detailed human-readable summary
#   - Flask web interface

# Automated NSE Stock Technical Analysis with Chart Image Download and AI Insight
# ====================================================================================
# Requirements:
#   - Live chart image download (hourly & 15 min, optional daily/weekly)
#   - Add EMA indicators, volume, RSI
#   - Technical analysis via both image and AI (Falcon-7B-Instruct)
#   - Output a detailed human-readable summary
#   - Flask web interface


# Automated NSE Stock Technical Analysis with Chart Image Download + AI Insight
# =======================================================================
# Requirements:
#   - Live chart image download (hourly & 15 min, optional daily/weekly)
#   - Add EMA indicators, volume, RSI
#   - Technical analysis via both image and AI
#   - Output a detailed human-readable summary
#   - Flask web interface
#   - Real-time NSE price data

import os
import datetime
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import talib
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

app = Flask(__name__)


# Load lightweight AI model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", use_auth_token=True)
model.eval()

# ---- FUNCTION: Download Chart Images in Batches ----
def fetch_chart_images(symbol, interval):
    image_paths = []

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    # Batch 1: EMAs (20, 50, 100)
    payload1 = {
        "symbol": symbol,
        "interval": interval,
        "studies": [
            {"name": "Moving Average Exponential", "inputs": {"length": 20}},
            {"name": "Moving Average Exponential", "inputs": {"length": 50}},
            {"name": "Moving Average Exponential", "inputs": {"length": 100}}
        ],
        "theme": "dark",
        "width": 800,
        "height": 600
    }

    response1 = requests.post("https://api.chart-img.com/v2/tradingview/advanced-chart", json=payload1, headers=headers)
    if response1.status_code == 200:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"static/{symbol.replace(':', '_')}_{interval}_set1_{ts}.png"
        os.makedirs("static", exist_ok=True)
        with open(fname, 'wb') as f:
            f.write(response1.content)
        image_paths.append(fname)
    else:
        raise Exception(f"Chart API Set1 failed: {response1.status_code} - {response1.text}")

    # Batch 2: EMA 200, RSI, Volume
    payload2 = {
        "symbol": symbol,
        "interval": interval,
        "studies": [
            {"name": "Moving Average Exponential", "inputs": {"length": 200}},
            {"name": "Relative Strength Index", "inputs": {"length": 14}},
            {"name": "Volume", "overrides": {}}
        ],
        "theme": "dark",
        "width": 800,
        "height": 600
    }

    response2 = requests.post("https://api.chart-img.com/v2/tradingview/advanced-chart", json=payload2, headers=headers)
    if response2.status_code == 200:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"static/{symbol.replace(':', '_')}_{interval}_set2_{ts}.png"
        with open(fname, 'wb') as f:
            f.write(response2.content)
        image_paths.append(fname)
    else:
        raise Exception(f"Chart API Set2 failed: {response2.status_code} - {response2.text}")

    return image_paths

# # ---- FUNCTION: Fetch Real-Time OHLC Data ----
# def fetch_realtime_data(symbol):
#     url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS?interval=15m&range=3m"
#     r = requests.get(url)
#     try:
#         json_data = r.json()
#     except Exception as e:
#         print("Raw response:", r.text)  # Helpful for debugging
#         raise Exception(f"Failed to parse JSON: {e}")
#     json_data = r.json()
#     timestamps = json_data['chart']['result'][0]['timestamp']
#     indicators = json_data['chart']['result'][0]['indicators']['quote'][0]
#     df = pd.DataFrame(indicators)
#     df['close'] = df['close'].fillna(method='ffill')
#     df = df[['close', 'volume']].dropna()
#     return df

import requests

def fetch_realtime_data(symbol):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-chart"
    querystring = {"symbol": f"{symbol}.NS", "interval": "15m", "range": "3m"}

    headers = {
        "X-RapidAPI-Key": "82979df23cmsha6204642f240bf1p158160jsn478e2c117588",
        "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    try:
        json_data = response.json() 
    except Exception as e:
        print("Raw response:", response.text)  # Helpful for debugging
        raise Exception(f"Failed to parse JSON: {e}")

    timestamps = json_data['chart']['result'][0]['timestamp']
    indicators = json_data['chart']['result'][0]['indicators']['quote'][0]

    df = pd.DataFrame(indicators)
    df['close'] = df['close'].fillna(method='ffill')
    df = df[['close', 'volume']].dropna()
    return df

# ---- FUNCTION: Technical Analysis using CSV ----
def perform_technical_analysis(df):
    df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
    df['EMA50'] = talib.EMA(df['close'], timeperiod=50)
    df['EMA100'] = talib.EMA(df['close'], timeperiod=100)
    df['EMA200'] = talib.EMA(df['close'], timeperiod=200)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)

    latest = df.iloc[-1]

    sentiment = "Bullish" if latest['EMA20'] > latest['EMA50'] else "Bearish"
    rsi_signal = "Overbought" if latest['RSI'] > 70 else ("Oversold" if latest['RSI'] < 30 else "Neutral")

    summary = {
        "sentiment": sentiment,
        "rsi": latest['RSI'],
        "rsi_signal": rsi_signal,
        "current_price": latest['close'],
        "ema20": latest['EMA20'],
        "ema50": latest['EMA50'],
        "ema100": latest['EMA100'],
        "ema200": latest['EMA200'],
        "entry": latest['close'] if sentiment == "Bullish" else None,
        "stop_loss": latest['EMA200'],
        "near_term_target": latest['EMA20'] + 20,
        "mid_term_target": latest['EMA50'] + 50,
        "long_term_target": latest['EMA100'] + 100
    }
    return summary

# ---- FUNCTION: AI-Based Technical Analysis using LLM ----
def ai_based_technical_insight(df):
    prices = df['close'].tail(60).tolist()
    prompt = f"""
    Analyze this stock based on the last 60 price points:
    {prices}

    Provide sentiment (Bullish/Bearish), near/mid/long-term targets, entry and exit points, and a confidence summary. Also provide insight into what candlestick, and trendline patterns are forming. What graph patterns are forming?
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "sentiment": "See AI Output",
        "commentary": response.strip()
    }

# ---- FUNCTION: Summary and Comparison ----
def compare_results(ta_summary, ai_summary):
    return {
        "technical_analysis": ta_summary,
        "ai_analysis": ai_summary,
        "comparison": "Agree" if ta_summary['sentiment'] in ai_summary['commentary'] else "Mixed"
    }

# ---- FLASK ROUTES ----
@app.route('/')
def index():
    return '''<form method="get" action="/analyze">
                NSE/BSE Symbol: <input type="text" name="symbol" value="GANESHHOUC">
                <input type="submit" value="Analyze">
              </form>'''

@app.route('/analyze')
def analyze():
    symbol = request.args.get("symbol", default="GANESHHOUC").upper()
    full_symbol = f"NSE:{symbol}"

    chart_files = []
    for interval in INTERVALS:
        try:
            chart_files += fetch_chart_images(full_symbol, interval)
        except Exception as e:
            return jsonify({"error": f"Error fetching chart: {str(e)}"})

    try:
        df = fetch_realtime_data(symbol)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch real-time data: {str(e)}"})

    ta_result = perform_technical_analysis(df)
    ai_result = ai_based_technical_insight(df)
    summary = compare_results(ta_result, ai_result)

    html = f"""
    <h2>Chart Snapshots</h2>
    {''.join([f'<img src="/{img}" width="800"><br>' for img in chart_files])}
    <h3>Technical Analysis</h3>
    <pre>{ta_result}</pre>
    <h3>AI Analysis</h3>
    <pre>{ai_result['commentary']}</pre>
    <h3>Summary</h3>
    <pre>{summary['comparison']} between technical and AI analysis</pre>
    """
    return html

if __name__ == "__main__":
    app.run(debug=True)
