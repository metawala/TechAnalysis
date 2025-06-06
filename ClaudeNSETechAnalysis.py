import os
import datetime
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from functools import wraps
from dotenv import load_dotenv
import base64
import json
import re

app = Flask(__name__)
load_dotenv()

# Configuration
CHART_API_KEY = os.getenv("CHART_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head><title>NSE Stock Analyzer</title></head>
    <body style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px;">
        <h2>NSE Stock Technical Analysis</h2>
        <form method="get" action="/analyze">
            <label>Enter NSE Stock Symbol:</label><br>
            <input type="text" name="symbol" value="RELIANCE" style="padding: 8px; width: 200px; margin: 10px 0;">
            <br>
            <input type="submit" value="Analyze" style="padding: 8px 16px;">
        </form>
    </body>
    </html>
    ''')

def fetch_charts(symbol, interval='1h'):
    """Download 2 charts within free tier limits"""
    headers = {
        "x-api-key": CHART_API_KEY,
        "Content-Type": "application/json"
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_paths = []
    
    # Chart 1: Price + EMAs + Volume
    chart1_payload = {
        "symbol": symbol,
        "interval": interval,
        "studies": [
            {"name": "Moving Average Exponential", "inputs": {"length": 20}},
            {"name": "Moving Average Exponential", "inputs": {"length": 200}},
            {"name": "Volume"}
        ],
        "theme": "dark",
        "width": 800,
        "height": 600,
        "bars": 100
    }
    
    # Chart 2: RSI + MACD + Volume
    chart2_payload = {
        "symbol": symbol,
        "interval": interval,
        "studies": [
            {"name": "Relative Strength Index", "inputs": {"length": 14}},
            {"name": "MACD", "inputs": {"fast_length": 12, "slow_length": 26, "signal_length": 9}},
            {"name": "Volume"}
        ],
        "theme": "dark",
        "width": 800,
        "height": 600,
        "bars": 100
    }
    
    chart_configs = [
        (chart1_payload, "price_emas"),
        (chart2_payload, "rsi_macd")
    ]
    
    try:
        print(f"Downloading charts for {symbol}...")
        os.makedirs("static", exist_ok=True)
        
        for i, (payload, chart_type) in enumerate(chart_configs, 1):
            try:
                response = requests.post(
                    "https://api.chart-img.com/v2/tradingview/advanced-chart", 
                    json=payload, 
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    fname = f"static/{symbol.replace(':', '_')}_{chart_type}_{interval}_{timestamp}.png"
                    with open(fname, 'wb') as f:
                        f.write(response.content)
                    chart_paths.append(fname)
                    print(f"Chart {i} saved: {fname}")
                    
                    if i < len(chart_configs):
                        time.sleep(6)  # Rate limiting
                else:
                    print(f"Chart {i} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"Error downloading chart {i}: {e}")
                continue
        
        return chart_paths if chart_paths else None
            
    except Exception as e:
        print(f"Chart download error: {e}")
        return None

def encode_image_to_base64(image_path):
    """Convert image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Image encoding error: {e}")
        return None

def analyze_chart_with_groq(image_path, chart_type):
    """Analyze chart using Groq API"""
    try:
        if not GROQ_API_KEY:
            return None
        
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if 'price_emas' in chart_type:
            prompt = """Analyze this price chart and extract:
1. Current stock price (exact number)
2. EMA 20 value
3. EMA 200 value  
4. Support level
5. Resistance level
6. Trend (Bullish/Bearish/Neutral)
7. Candlestick patterns visible
Provide specific numerical values."""
        else:
            prompt = """Analyze this technical indicators chart:
1. Current RSI value (0-100)
2. MACD line value
3. MACD signal line value
4. Volume trend (High/Low/Average)
5. Any divergences
6. Momentum (Increasing/Decreasing)
Provide specific numbers and analysis."""
        
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.1
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                               headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            return parse_analysis(content, chart_type)
        else:
            print(f"Groq API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Groq analysis error: {e}")
        return None

def parse_analysis(text, chart_type):
    """Parse analysis text and extract data"""
    result = {'raw_analysis': text}
    text_lower = text.lower()
    
    # Extract numbers using regex
    numbers = re.findall(r'\d+\.?\d*', text)
    
    if 'price_emas' in chart_type:
        # Extract price data
        price_match = re.search(r'price.*?(\d+\.?\d*)', text_lower)
        if price_match:
            result['current_price'] = float(price_match.group(1))
        
        ema20_match = re.search(r'ema.*?20.*?(\d+\.?\d*)', text_lower)
        if ema20_match:
            result['ema_20'] = float(ema20_match.group(1))
        
        ema50_match = re.search(r'ema.*?50.*?(\d+\.?\d*)', text_lower)
        if ema50_match:
            result['ema_50'] = float(ema50_match.group(1))
        
        support_match = re.search(r'support.*?(\d+\.?\d*)', text_lower)
        if support_match:
            result['support'] = float(support_match.group(1))
        
        resistance_match = re.search(r'resistance.*?(\d+\.?\d*)', text_lower)
        if resistance_match:
            result['resistance'] = float(resistance_match.group(1))
        
        if 'bullish' in text_lower:
            result['trend'] = 'Bullish'
        elif 'bearish' in text_lower:
            result['trend'] = 'Bearish'
        else:
            result['trend'] = 'Neutral'
    
    else:
        # Extract technical indicators
        rsi_patterns = [
            r'rsi.*?value.*?(\d+\.?\d*)',
            r'rsi.*?(\d+\.\d+)',
            r'rsi.*?:\s*(\d+\.?\d*)',
            r'current\s+rsi.*?(\d+\.?\d*)',
            r'rsi.*?(\d+\.?\d*)'
            ]
        
        for pattern in rsi_patterns:
            rsi_match = re.search(pattern, text_lower)
            if rsi_match:
                try:
                    rsi_val = float(rsi_match.group(1))
                    if 0 <= rsi_val <= 100 and rsi_val > 10:  # Avoid catching period numbers like "14"
                        result['rsi'] = rsi_val
                        break
                except ValueError:
                    continue
        
        macd_match = re.search(r'macd.*?line.*?(\d+\.?\d*)', text_lower)
        if macd_match:
            result['macd_line'] = float(macd_match.group(1))
    
    return result

def generate_trading_analysis(price_data, tech_data, symbol):
    """Generate trading recommendations"""
    
    # Get values with fallbacks
    current_price = price_data.get('current_price', 1450.0)
    rsi = tech_data.get('rsi', 50.0)
    trend = price_data.get('trend', 'Neutral')
    support = price_data.get('support', current_price * 0.95)
    resistance = price_data.get('resistance', current_price * 1.05)
    
    # Generate signal
    if rsi < 30 and trend == 'Bullish':
        signal = 'STRONG BUY'
    elif rsi < 40 and trend in ['Bullish', 'Neutral']:
        signal = 'BUY'
    elif rsi > 70 and trend == 'Bearish':
        signal = 'STRONG SELL'
    elif rsi > 60 and trend in ['Bearish', 'Neutral']:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    # Calculate levels
    if 'BUY' in signal:
        entry_levels = {
            'conservative': support * 1.01,
            'aggressive': current_price * 1.002
        }
        targets = [current_price * 1.03, current_price * 1.06, resistance * 0.98]
        stop_loss = support * 0.98
    elif 'SELL' in signal:
        entry_levels = {
            'conservative': resistance * 0.99,
            'aggressive': current_price * 0.998
        }
        targets = [current_price * 0.97, current_price * 0.94, support * 1.02]
        stop_loss = resistance * 1.02
    else:
        entry_levels = {'current': current_price}
        targets = [current_price * 1.03, current_price * 1.06, current_price * 1.09]
        stop_loss = current_price * 0.95
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'signal': signal,
        'trend': trend,
        'rsi': rsi,
        'support': support,
        'resistance': resistance,
        'entry_levels': entry_levels,
        'targets': targets,
        'stop_loss': stop_loss,
        'ema_20': price_data.get('ema_20', 'N/A'),
        'ema_50': price_data.get('ema_50', 'N/A'),
        'macd_line': tech_data.get('macd_line', 'N/A'),
        'price_analysis': price_data.get('raw_analysis', 'Not available'),
        'tech_analysis': tech_data.get('raw_analysis', 'Not available')
    }

def format_analysis_text(text):
    """Format analysis text with visual enhancements"""
    if not text:
        return "Analysis not available"
    
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        # Remove numbered lists and bullet points
        line = re.sub(r'^\d+\.\s*', '', line)
        line = re.sub(r'^[-*•]\s*', '', line)
        
        # Add visual icons based on content
        if any(word in line.lower() for word in ['price', 'current', 'trading']):
            icon = '💰'
        elif any(word in line.lower() for word in ['ema', 'moving average', 'ma']):
            icon = '📈'
        elif any(word in line.lower() for word in ['support', 'resistance']):
            icon = '🎯'
        elif any(word in line.lower() for word in ['trend', 'bullish', 'bearish']):
            icon = '📊'
        elif any(word in line.lower() for word in ['rsi', 'relative strength']):
            icon = '⚡'
        elif any(word in line.lower() for word in ['macd', 'momentum']):
            icon = '🔄'
        elif any(word in line.lower() for word in ['volume', 'vol']):
            icon = '📦'
        elif any(word in line.lower() for word in ['pattern', 'candlestick']):
            icon = '🕯️'
        else:
            icon = '📌'
        
        lines.append(f'<div style="margin: 8px 0; padding: 8px 12px; background: rgba(255,255,255,0.7); border-radius: 6px; border-left: 3px solid #ddd;"><span style="margin-right: 8px;">{icon}</span>{line}</div>')
    
    return ''.join(lines)

@app.route('/analyze')
def analyze():
    symbol = request.args.get('symbol', 'RELIANCE').upper()
    
    try:
        # Download charts
        chart_paths = fetch_charts(f"NSE:{symbol}", interval='1h')
        
        if not chart_paths:
            return f"<h3>Error: Could not download charts for {symbol}</h3><a href='/'>Go Back</a>"
        
        # Analyze charts
        price_data = {}
        tech_data = {}
        
        for chart_path in chart_paths:
            chart_type = 'price_emas' if 'price_emas' in chart_path else 'rsi_macd'
            analysis = analyze_chart_with_groq(chart_path, chart_type)
            
            if analysis:
                if chart_type == 'price_emas':
                    price_data = analysis
                else:
                    tech_data = analysis
        
        # Generate trading analysis
        result = generate_trading_analysis(price_data, tech_data, symbol)
        result['price_analysis'] = format_analysis_text(result.get('price_analysis', ''))
        result['tech_analysis'] = format_analysis_text(result.get('tech_analysis', ''))
        
        # Display results
        return render_template_string(RESULTS_TEMPLATE, 
                                    chart_paths=chart_paths, 
                                    **result)
        
    except Exception as e:
        return f"<h3>Analysis failed: {e}</h3><a href='/'>Go Back</a>"

RESULTS_TEMPLATE = '''
<html>
<head><title>{{ symbol }} Analysis</title></head>
<body style="font-family: Arial; max-width: 1000px; margin: 20px auto; padding: 20px;">
    
    <h2>{{ symbol }} Technical Analysis</h2>
    <a href="/">← Back to Home</a>
    
    <h3>Charts</h3>
    <div style="display: flex; gap: 10px; margin: 10px 0;">
        {% for chart in chart_paths %}
        <img src="/{{ chart }}" style="width: 48%; border: 1px solid #ccc;">
        {% endfor %}
    </div>
    
    <h3>Current Data</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><td><b>Current Price</b></td><td>₹{{ "%.2f"|format(current_price) }}</td></tr>
        <tr><td><b>Trend</b></td><td>{{ trend }}</td></tr>
        <tr><td><b>RSI</b></td><td>{{ "%.1f"|format(rsi) }}</td></tr>
        <tr><td><b>EMA 20</b></td><td>{{ ema_20 }}</td></tr>
        <tr><td><b>EMA 200</b></td><td>{{ ema_50 }}</td></tr>
        <tr><td><b>MACD Line</b></td><td>{{ macd_line }}</td></tr>
        <tr><td><b>Support</b></td><td>₹{{ "%.2f"|format(support) }}</td></tr>
        <tr><td><b>Resistance</b></td><td>₹{{ "%.2f"|format(resistance) }}</td></tr>
    </table>
    
    <h3>Trading Recommendation</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><td><b>Signal</b></td><td><b>{{ signal }}</b></td></tr>
        <tr><td><b>Entry Levels</b></td><td>
            {% for level, price in entry_levels.items() %}
            {{ level.title() }}: ₹{{ "%.2f"|format(price) }}<br>
            {% endfor %}
        </td></tr>
        <tr><td><b>Targets</b></td><td>
            {% for target in targets %}
            Target {{ loop.index }}: ₹{{ "%.2f"|format(target) }}<br>
            {% endfor %}
        </td></tr>
        <tr><td><b>Stop Loss</b></td><td>₹{{ "%.2f"|format(stop_loss) }}</td></tr>
    </table>
    
    <h3>Detailed Analysis</h3>
    <div style="display: flex; gap: 20px; margin: 20px 0;">
        <div style="flex: 1;">
            <h4>Price & EMAs Analysis:</h4>
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{{ price_analysis|safe }}</div>
        </div>
        <div style="flex: 1;">
            <h4>Technical Indicators Analysis:</h4>
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f0f8ff 100%); padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{{ tech_analysis|safe }}</div>
        </div>
    </div>
    
    <p><a href="/">← Analyze Another Stock</a></p>
    
</body>
</html>
'''

if __name__ == '__main__':
    print("NSE Stock Analyzer starting...")
    print("Access: http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)