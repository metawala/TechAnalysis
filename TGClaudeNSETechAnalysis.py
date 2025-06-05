import os
import datetime
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import base64
import json
import re
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import io
from bs4 import BeautifulSoup
import yfinance as yf

load_dotenv()

# Configuration
CHART_API_KEY = os.getenv("CHART_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4000  # Leave some buffer under 4096 limit

def fetch_charts(symbol, interval='1h'):
    """Download 2 charts with 3-month range using range parameter"""
    headers = {
        "x-api-key": CHART_API_KEY,
        "Content-Type": "application/json"
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Chart 1: Price + EMAs + Volume (3 months range)
    chart1_payload = {
        "symbol": symbol,
        "interval": interval,
        "range": "3M",  # 3 months of data
        "studies": [
            {"name": "Moving Average Exponential", "input": {"length": 20}},
            {"name": "Moving Average Exponential", "input": {"length": 50}},
            {"name": "Moving Average Exponential", "input": {"length": 200}},
        ],
        "theme": "dark",
        "width": 800,
        "height": 600
    }
    
    # Chart 2: RSI + MACD + Volume (3 months range)
    chart2_payload = {
        "symbol": symbol,
        "interval": interval,
        "range": "3M",  # 3 months of data
        "studies": [
            {"name": "Relative Strength Index", "input": {"length": 14}},
            {"name": "MACD", "input": {"fast_length": 12, "slow_length": 26, "signal_length": 9}},
            {"name": "Volume"}
        ],
        "theme": "dark",
        "width": 800,
        "height": 600
    }
    
    chart_configs = [
        (chart1_payload, "price_emas"),
        (chart2_payload, "rsi_macd")
    ]
    
    chart_data = []  # Store raw image data instead of files
    
    try:
        print(f"Downloading charts for {symbol}...")
        
        for i, (payload, chart_type) in enumerate(chart_configs, 1):
            try:
                response = requests.post(
                    "https://api.chart-img.com/v2/tradingview/advanced-chart", 
                    json=payload, 
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    chart_data.append({
                        'type': chart_type,
                        'data': response.content,
                        'filename': f"{symbol.replace(':', '_')}_{chart_type}_{interval}_{timestamp}.png"
                    })
                    print(f"Chart {i} downloaded: {chart_type}")
                    
                    if i < len(chart_configs):
                        time.sleep(6)  # Rate limiting
                else:
                    print(f"Chart {i} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"Error downloading chart {i}: {e}")
                continue
        
        return chart_data if chart_data else None
            
    except Exception as e:
        print(f"Chart download error: {e}")
        return None

def analyze_chart_with_groq_telegram(chart_data, chart_type):
    """Analyze chart using Groq API with enhanced prompts for patterns and trading style"""
    try:
        if not GROQ_API_KEY:
            return None
        
        # Convert raw image data to base64
        base64_image = base64.b64encode(chart_data).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if 'price_emas' in chart_type:
            prompt = """Analyze this 3-month hourly price chart and provide concise analysis:

PRICE DATA:
1. Current stock price (exact number)
2. EMA 20 value
3. EMA 50 value
4. EMA 200 value

SUPPORT & RESISTANCE LEVELS (Provide 3 each):
4. Support Level 1 (strongest support)
5. Support Level 2 (secondary support)
6. Support Level 3 (tertiary support)
7. Resistance Level 1 (strongest resistance)
8. Resistance Level 2 (secondary resistance)
9. Resistance Level 3 (tertiary resistance)

TREND & PATTERNS:
10. Overall Trend (Bullish/Bearish/Neutral)
11. Chart Pattern (Head & Shoulders, Triangle, Channel, Flag, Wedge, etc.)
12. Candlestick Pattern (Doji, Hammer, Engulfing, etc.)
13. Trendline Pattern (Ascending, Descending, Sideways)

TRADING RECOMMENDATION:
14. Trading Style (Day Trading/Swing Trading/Position Trading)
15. Time Horizon (specific duration like "2-4 hours", "3-7 days", "2-8 weeks")

Provide specific numerical values and clear pattern names. Keep analysis concise."""
        else:
            prompt = """Analyze this 3-month hourly technical indicators chart concisely:

TECHNICAL INDICATORS:
1. Current RSI value (0-100)
2. RSI Trend (Oversold <27, Neutral 27-80, Overbought >80) - NOTE: Use 27-80 range, not conventional 30-70
3. MACD line value
4. MACD signal line value
5. MACD Histogram trend
6. Volume trend (High/Low/Average compared to recent average)

MOMENTUM & DIVERGENCES:
7. Momentum (Increasing/Decreasing/Neutral)
8. Any RSI divergences with price
9. Any MACD divergences with price
10. Volume divergences

TRADING SIGNALS:
11. RSI trading signal (Buy if <27, Sell if >80, Hold if 27-80)
12. MACD trading signal (Buy/Sell/Hold)
13. Overall momentum signal

TRADING RECOMMENDATION:
14. Recommended Trading Style (Day Trading/Swing Trading/Position Trading)
15. Time Horizon (specific duration like "2-4 hours", "3-7 days", "2-8 weeks")

Provide specific numbers and clear analysis. Keep response concise."""
        
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
            "max_tokens": 800,  # Reduced to keep responses shorter
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
    
    
def create_summary_analysis(result):
    """Create a comprehensive summary of both price and technical analysis"""
    summary = f"""📋 COMPREHENSIVE ANALYSIS SUMMARY FOR {result['symbol']}

🔍 MARKET OVERVIEW:
Current market sentiment shows {result['trend'].lower()} bias with RSI at {result['rsi']:.1f} (using 27-80 range). Price is trading at ₹{result['current_price']:.2f} with the stock showing {result['signal'].lower()} characteristics.

📈 PRICE ACTION ANALYSIS:
The stock is positioned relative to key EMAs with EMA20 at {result['ema_20']}, EMA50 at {result['ema_50']}, and EMA200 at {result['ema_200']}. 

Key support zones are identified at ₹{result['support_levels'][0]:.2f} (primary), ₹{result['support_levels'][1]:.2f} (secondary), and ₹{result['support_levels'][2]:.2f} (tertiary). 

Resistance barriers are located at ₹{result['resistance_levels'][0]:.2f} (immediate), ₹{result['resistance_levels'][1]:.2f} (intermediate), and ₹{result['resistance_levels'][2]:.2f} (strong).

🔧 TECHNICAL INDICATORS:
RSI reading of {result['rsi']:.1f} suggests {"oversold conditions" if result['rsi'] < 27 else "overbought conditions" if result['rsi'] > 80 else "neutral momentum"}. MACD signals show {result['macd_line']} line against {result['macd_signal']} signal line.

🎯 PATTERN RECOGNITION:
Chart structure reveals {result['chart_pattern'].lower()} formation with {result['candlestick_pattern'].lower()} candlestick patterns. The overall trendline pattern appears {result['trendline_pattern'].lower()}.

💡 TRADING STRATEGY:
Recommended approach is {result['trading_style'].lower()} with {result['time_horizon']} time horizon. Current signal strength: {result['signal']}.

Risk management suggests stop loss at ₹{result['stop_loss']:.2f} with profit targets at ₹{result['targets'][0]:.2f}, ₹{result['targets'][1]:.2f}, and ₹{result['targets'][2]:.2f}.

⚠️ DISCLAIMER: This analysis is for educational purposes. Always conduct your own research and risk assessment before trading."""
    
    return summary



def parse_analysis(text, chart_type):
    """Parse analysis text and extract data with multiple support/resistance levels and patterns"""
    result = {'raw_analysis': text}
    text_lower = text.lower()
    
    if 'price_emas' in chart_type:
        # Extract price data
        price_match = re.search(r'(?:current\s+)?price.*?(\d+\.?\d*)', text_lower)
        if price_match:
            result['current_price'] = float(price_match.group(1))
        
        # Extract EMAs
        ema20_match = re.search(r'ema.*?20.*?(\d+\.?\d*)', text_lower)
        if ema20_match:
            result['ema_20'] = float(ema20_match.group(1))
        
        ema50_match = re.search(r'ema.*?50.*?(\d+\.?\d*)', text_lower)
        if ema50_match:
            result['ema_50'] = float(ema50_match.group(1))
        
        ema200_match = re.search(r'ema.*?200.*?(\d+\.?\d*)', text_lower)
        if ema200_match:
            result['ema_200'] = float(ema200_match.group(1))
        
        # Extract multiple support levels
        support_levels = []
        support_patterns = [
            r'support\s+level\s+1.*?(\d+\.?\d*)',
            r'support\s+1.*?(\d+\.?\d*)',
            r'strongest\s+support.*?(\d+\.?\d*)',
            r'support\s+level\s+2.*?(\d+\.?\d*)',
            r'support\s+2.*?(\d+\.?\d*)',
            r'secondary\s+support.*?(\d+\.?\d*)',
            r'support\s+level\s+3.*?(\d+\.?\d*)',
            r'support\s+3.*?(\d+\.?\d*)',
            r'tertiary\s+support.*?(\d+\.?\d*)'
        ]
        
        for pattern in support_patterns:
            match = re.search(pattern, text_lower)
            if match:
                support_levels.append(float(match.group(1)))
        
        # If we didn't get 3 specific levels, try general support mentions
        if len(support_levels) < 3:
            general_supports = re.findall(r'support.*?(\d+\.?\d*)', text_lower)
            for support in general_supports:
                if float(support) not in support_levels:
                    support_levels.append(float(support))
                if len(support_levels) >= 3:
                    break
        
        result['support_levels'] = support_levels[:3] if support_levels else []
        
        # Extract multiple resistance levels
        resistance_levels = []
        resistance_patterns = [
            r'resistance\s+level\s+1.*?(\d+\.?\d*)',
            r'resistance\s+1.*?(\d+\.?\d*)',
            r'strongest\s+resistance.*?(\d+\.?\d*)',
            r'resistance\s+level\s+2.*?(\d+\.?\d*)',
            r'resistance\s+2.*?(\d+\.?\d*)',
            r'secondary\s+resistance.*?(\d+\.?\d*)',
            r'resistance\s+level\s+3.*?(\d+\.?\d*)',
            r'resistance\s+3.*?(\d+\.?\d*)',
            r'tertiary\s+resistance.*?(\d+\.?\d*)'
        ]
        
        for pattern in resistance_patterns:
            match = re.search(pattern, text_lower)
            if match:
                resistance_levels.append(float(match.group(1)))
        
        # If we didn't get 3 specific levels, try general resistance mentions
        if len(resistance_levels) < 3:
            general_resistances = re.findall(r'resistance.*?(\d+\.?\d*)', text_lower)
            for resistance in general_resistances:
                if float(resistance) not in resistance_levels:
                    resistance_levels.append(float(resistance))
                if len(resistance_levels) >= 3:
                    break
        
        result['resistance_levels'] = resistance_levels[:3] if resistance_levels else []
        
        # Extract trend
        if 'bullish' in text_lower:
            result['trend'] = 'Bullish'
        elif 'bearish' in text_lower:
            result['trend'] = 'Bearish'
        else:
            result['trend'] = 'Neutral'
        
        # Extract chart patterns
        chart_patterns = ['head and shoulders', 'triangle', 'channel', 'flag', 'wedge', 'pennant', 'cup and handle', 'double top', 'double bottom']
        for pattern in chart_patterns:
            if pattern in text_lower:
                result['chart_pattern'] = pattern.title()
                break
        
        # Extract candlestick patterns
        candle_patterns = ['doji', 'hammer', 'engulfing', 'shooting star', 'hanging man', 'spinning top', 'marubozu']
        for pattern in candle_patterns:
            if pattern in text_lower:
                result['candlestick_pattern'] = pattern.title()
                break
        
        # Extract trendline pattern
        if 'ascending' in text_lower:
            result['trendline_pattern'] = 'Ascending'
        elif 'descending' in text_lower:
            result['trendline_pattern'] = 'Descending'
        elif 'sideways' in text_lower:
            result['trendline_pattern'] = 'Sideways'
        
        # Extract trading style and time horizon
        if 'day trading' in text_lower:
            result['trading_style'] = 'Day Trading'
        elif 'swing trading' in text_lower:
            result['trading_style'] = 'Swing Trading'
        elif 'position trading' in text_lower:
            result['trading_style'] = 'Position Trading'
        
        # Extract time horizon
        time_patterns = [
            r'(\d+-\d+\s+hours?)',
            r'(\d+-\d+\s+days?)',
            r'(\d+-\d+\s+weeks?)',
            r'(\d+\s+hours?)',
            r'(\d+\s+days?)',
            r'(\d+\s+weeks?)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result['time_horizon'] = match.group(1)
                break
    
    else:
        # Extract RSI data with multiple patterns
        rsi_patterns = [
            r'rsi.*?value.*?(\d+\.?\d*)',
            r'current\s+rsi.*?(\d+\.?\d*)',
            r'rsi.*?(\d+\.\d+)',
            r'rsi.*?:\s*(\d+\.?\d*)'
        ]

        for pattern in rsi_patterns:
            rsi_match = re.search(pattern, text_lower)
            if rsi_match:
                try:
                    rsi_val = float(rsi_match.group(1))
                    if 0 <= rsi_val <= 100:
                        result['rsi'] = rsi_val
                        break
                except ValueError:
                    continue
        
        # Extract MACD data
        macd_line_match = re.search(r'macd\s+line.*?(\d+\.?\d*)', text_lower)
        if macd_line_match:
            result['macd_line'] = float(macd_line_match.group(1))
        
        macd_signal_match = re.search(r'macd\s+signal.*?(\d+\.?\d*)', text_lower)
        if macd_signal_match:
            result['macd_signal'] = float(macd_signal_match.group(1))
        
        # Extract trading style and time horizon for technical analysis too
        if 'day trading' in text_lower:
            result['trading_style'] = 'Day Trading'
        elif 'swing trading' in text_lower:
            result['trading_style'] = 'Swing Trading'
        elif 'position trading' in text_lower:
            result['trading_style'] = 'Position Trading'
        
        # Extract time horizon
        time_patterns = [
            r'(\d+-\d+\s+hours?)',
            r'(\d+-\d+\s+days?)',
            r'(\d+-\d+\s+weeks?)',
            r'(\d+\s+hours?)',
            r'(\d+\s+days?)',
            r'(\d+\s+weeks?)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result['time_horizon'] = match.group(1)
                break
    
    return result

def generate_trading_analysis(price_data, tech_data, symbol):
    """Generate enhanced trading recommendations with custom RSI thresholds (27-80)"""
    current_price = price_data.get('current_price', 1450.0)
    rsi = tech_data.get('rsi', 50.0)
    trend = price_data.get('trend', 'Neutral')
    
    # Get multiple support and resistance levels
    support_levels = price_data.get('support_levels', [current_price * 0.97, current_price * 0.94, current_price * 0.91])
    resistance_levels = price_data.get('resistance_levels', [current_price * 1.03, current_price * 1.06, current_price * 1.09])
    
    # Ensure we have 3 levels each
    while len(support_levels) < 3:
        support_levels.append(current_price * (0.97 - 0.03 * len(support_levels)))
    
    while len(resistance_levels) < 3:
        resistance_levels.append(current_price * (1.03 + 0.03 * len(resistance_levels)))
    
    # Generate signal based on CUSTOM RSI thresholds (27-80 instead of 30-70)
    if rsi < 27 and trend == 'Bullish':
        signal = 'STRONG BUY'
    elif rsi < 35 and trend in ['Bullish', 'Neutral']:
        signal = 'BUY'
    elif rsi > 80 and trend == 'Bearish':
        signal = 'STRONG SELL'
    elif rsi > 75 and trend in ['Bearish', 'Neutral']:
        signal = 'SELL'
    else:
        signal = 'HOLD'
        
    print()
    print("------------------- RSI: " + str(rsi))
    print("------------------- SIGNAL: " + signal)
    print()
    
    # Calculate entry levels based on multiple supports/resistances
    if 'BUY' in signal:
        entry_levels = {
            'aggressive': max(support_levels) * 1.005,
            'moderate': (support_levels[0] + support_levels[1]) / 2,
            'conservative': support_levels[0] * 1.01
        }
        targets = resistance_levels
        stop_loss = min(support_levels) * 0.98
    elif 'SELL' in signal:
        entry_levels = {
            'aggressive': min(resistance_levels) * 0.995,
            'moderate': (resistance_levels[0] + resistance_levels[1]) / 2,
            'conservative': resistance_levels[0] * 0.99
        }
        targets = [s for s in reversed(support_levels)]
        stop_loss = max(resistance_levels) * 1.02
    else:
        entry_levels = {'current': current_price}
        targets = resistance_levels
        stop_loss = min(support_levels) * 0.98
    
    # Get trading style and time horizon from analysis
    trading_style = price_data.get('trading_style') or tech_data.get('trading_style', 'Swing Trading')
    time_horizon = price_data.get('time_horizon') or tech_data.get('time_horizon', '3-7 days')
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'signal': signal,
        'trend': trend,
        'rsi': rsi,
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'entry_levels': entry_levels,
        'targets': targets,
        'stop_loss': stop_loss,
        'trading_style': trading_style,
        'time_horizon': time_horizon,
        'chart_pattern': price_data.get('chart_pattern', 'Not identified'),
        'candlestick_pattern': price_data.get('candlestick_pattern', 'Not identified'),
        'trendline_pattern': price_data.get('trendline_pattern', 'Not identified'),
        'ema_20': price_data.get('ema_20', 'N/A'),
        'ema_50': price_data.get('ema_50', 'N/A'),
        'ema_200': price_data.get('ema_200', 'N/A'),
        'macd_line': tech_data.get('macd_line', 'N/A'),
        'macd_signal': tech_data.get('macd_signal', 'N/A'),
        'price_analysis': price_data.get('raw_analysis', 'Not available'),
        'tech_analysis': tech_data.get('raw_analysis', 'Not available')
    }

def sanitize_text_for_telegram(text):
    """Sanitize text to prevent Telegram Markdown parsing errors - IMPROVED VERSION"""
    if not text:
        return "Not available"
    
    # Convert to string and only escape the most problematic characters
    text = str(text)
    
    # Only escape characters that commonly break Telegram messages
    # Removed most escaping to improve readability
    text = text.replace('`', "'")  # Replace backticks with single quotes
    text = text.replace('*', '•')  # Replace asterisks with bullets
    text = text.replace('_', '-')  # Replace underscores with hyphens
    
    # Keep other characters as they rarely cause issues in plain text mode
    return text

def format_analysis_text_telegram(text):
    """Format analysis text for Telegram (no HTML) - Keep it concise and safe"""
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
        
        # Sanitize the line for Telegram
        line = sanitize_text_for_telegram(line)
        
        # Keep only the most important lines (limit to 5 lines max)
        if any(word in line.lower() for word in ['price', 'support', 'resistance', 'trend', 'signal', 'rsi', 'macd']):
            # Add emojis based on content
            if any(word in line.lower() for word in ['price', 'current', 'trading']):
                icon = '💰'
            elif any(word in line.lower() for word in ['support', 'resistance']):
                icon = '🎯'
            elif any(word in line.lower() for word in ['trend', 'bullish', 'bearish']):
                icon = '📊'
            elif any(word in line.lower() for word in ['rsi', 'relative strength']):
                icon = '⚡'
            elif any(word in line.lower() for word in ['macd', 'momentum']):
                icon = '🔄'
            else:
                icon = '📌'
            
            lines.append(f'{icon} {line}')
            
            if len(lines) >= 5:  # Limit to 5 lines
                break
    
    return '\n'.join(lines)

async def split_and_send_message(update, text, parse_mode=None):
    """Split long messages into chunks and send them safely without Markdown"""
    # Don't use Markdown to avoid parsing errors - send as plain text
    if len(text) <= MAX_MESSAGE_LENGTH:
        await update.message.reply_text(text)
        return
    
    # Split the message into chunks
    chunks = []
    current_chunk = ""
    
    for line in text.split('\n'):
        if len(current_chunk + line + '\n') > MAX_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                # Line itself is too long, split it further
                chunks.append(line[:MAX_MESSAGE_LENGTH])
        else:
            current_chunk += line + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Send each chunk
    for i, chunk in enumerate(chunks):
        if i == 0:
            await update.message.reply_text(chunk)
        else:
            await update.message.reply_text(f"Analysis continued...\n\n{chunk}")
        
        # Small delay between messages
        await asyncio.sleep(0.5)

###############################
## ALL CODE FOR RELATIVE RSI ##
###############################

def get_stock_market_cap_category(symbol):
    """
    Dynamically determine stock category (Large/Mid/Small cap) using yfinance
    """
    try:
        # Try NSE symbol first, then BSE if needed
        nse_symbol = f"{symbol}.NS"
        stock = yf.Ticker(nse_symbol)
        info = stock.info
        
        # Get market cap in INR (yfinance returns in USD, so we convert)
        market_cap_usd = info.get('marketCap', 0)
        
        if market_cap_usd == 0:
            # Try BSE symbol as fallback
            bse_symbol = f"{symbol}.BO"
            stock = yf.Ticker(bse_symbol)
            info = stock.info
            market_cap_usd = info.get('marketCap', 0)
        
        if market_cap_usd == 0:
            print(f"❌ Could not fetch market cap for {symbol}")
            return None, None
            
        # Convert USD to INR (approximate rate: 1 USD = 83 INR)
        market_cap_inr = market_cap_usd * 83
        
        # Indian market cap categories (in crores)
        market_cap_crores = market_cap_inr / 10000000  # Convert to crores
        
        print(f"📊 {symbol} Market Cap: ₹{market_cap_crores:.0f} crores")
        
        # Categorize based on Indian market standards
        if market_cap_crores >= 20000:  # 20,000+ crores = Large Cap
            return "Large Cap", "^NSEI"  # Nifty 50
        elif market_cap_crores >= 5000:  # 5,000-20,000 crores = Mid Cap
            return "Mid Cap", "^NSEMDCP50"  # Nifty Midcap 50
        else:  # < 5,000 crores = Small Cap
            return "Small Cap", "^CNXSC"  # Nifty Smallcap 100
            
    except Exception as e:
        print(f"❌ Error fetching market cap for {symbol}: {e}")
        return None, None

def calculate_relative_strength(stock_symbol, benchmark_symbol, period='3mo'):
    """
    Calculate relative strength of stock vs benchmark index
    """
    try:
        # Fetch stock data
        stock_data = yf.download(f"{stock_symbol}.NS", period=period, interval='1d')
        if stock_data.empty:
            stock_data = yf.download(f"{stock_symbol}.BO", period=period, interval='1d')
        
        # Fetch benchmark data
        benchmark_data = yf.download(benchmark_symbol, period=period, interval='1d')
        
        if stock_data.empty or benchmark_data.empty:
            return None
        
        # Calculate returns
        stock_start = stock_data['Close'].iloc[0]
        stock_end = stock_data['Close'].iloc[-1]
        stock_return = ((stock_end - stock_start) / stock_start) * 100
        
        benchmark_start = benchmark_data['Close'].iloc[0]
        benchmark_end = benchmark_data['Close'].iloc[-1]
        benchmark_return = ((benchmark_end - benchmark_start) / benchmark_start) * 100
        
        # Calculate relative strength
        relative_strength = stock_return - benchmark_return
        
        # Calculate correlation (optional but useful)
        aligned_data = pd.concat([
            stock_data['Close'].pct_change().dropna(),
            benchmark_data['Close'].pct_change().dropna()
        ], axis=1, keys=['Stock', 'Benchmark']).dropna()
        
        correlation = aligned_data.corr().iloc[0, 1] if len(aligned_data) > 1 else 0
        
        return {
            'stock_return': stock_return,
            'benchmark_return': benchmark_return,
            'relative_strength': relative_strength,
            'correlation': correlation,
            'outperforming': relative_strength > 0
        }
        
    except Exception as e:
        print(f"❌ Error calculating relative strength: {e}")
        return None

def get_benchmark_name(benchmark_symbol):
    """Get friendly name for benchmark index"""
    benchmark_names = {
        "^NSEI": "Nifty 50",
        "^NSEMDCP50": "Nifty Midcap 50", 
        "^CNXSC": "Nifty Smallcap 100",
        "^NSMIDCP": "Nifty Midcap 150"
    }
    return benchmark_names.get(benchmark_symbol, benchmark_symbol)

# Modify the generate_trading_analysis function to include relative strength
def generate_trading_analysis_with_relative_strength(price_data, tech_data, symbol):
    """Generate enhanced trading recommendations with relative strength analysis"""
    
    # Get existing analysis
    result = generate_trading_analysis(price_data, tech_data, symbol)
    
    # Add relative strength analysis
    print(f"🔍 Determining market cap category for {symbol}...")
    category, benchmark_symbol = get_stock_market_cap_category(symbol)
    
    if category and benchmark_symbol:
        print(f"📊 {symbol} categorized as: {category}")
        print(f"🎯 Benchmark: {get_benchmark_name(benchmark_symbol)}")
        
        # Calculate relative strength
        rel_strength = calculate_relative_strength(symbol, benchmark_symbol)
        
        if rel_strength:
            result.update({
                'market_cap_category': category,
                'benchmark_index': get_benchmark_name(benchmark_symbol),
                'benchmark_symbol': benchmark_symbol,
                'stock_3m_return': rel_strength['stock_return'],
                'benchmark_3m_return': rel_strength['benchmark_return'],
                'relative_strength': rel_strength['relative_strength'],
                'correlation': rel_strength['correlation'],
                'outperforming_benchmark': rel_strength['outperforming']
            })
        else:
            result.update({
                'market_cap_category': category,
                'benchmark_index': get_benchmark_name(benchmark_symbol),
                'relative_strength_error': 'Could not calculate relative strength'
            })
    else:
        result.update({
            'market_cap_category': 'Unknown',
            'relative_strength_error': 'Could not determine market cap category'
        })
    
    return result

# Update the create_summary_analysis function to include relative strength
def create_summary_analysis_with_relative_strength(result):
    """Create a comprehensive summary including relative strength analysis"""
    
    # Base summary (keep existing content)
    summary = f"""📋 COMPREHENSIVE ANALYSIS SUMMARY FOR {result['symbol']}

🏢 MARKET CLASSIFICATION:
Stock Category: {result.get('market_cap_category', 'Unknown')}
Benchmark Index: {result.get('benchmark_index', 'N/A')}

🔍 RELATIVE PERFORMANCE (3-Month):"""
    
    if result.get('relative_strength') is not None:
        rs = result['relative_strength']
        stock_ret = result['stock_3m_return']
        bench_ret = result['benchmark_3m_return']
        correlation = result['correlation']
        
        summary += f"""
Stock Return: {stock_ret:+.2f}%
Benchmark Return: {bench_ret:+.2f}%
Relative Strength: {rs:+.2f}%
Correlation: {correlation:.2f}
Performance: {'✅ OUTPERFORMING' if rs > 0 else '❌ UNDERPERFORMING'} benchmark"""
        
        # Add interpretation
        if rs > 5:
            summary += f"\n🚀 Strong outperformance vs {result.get('benchmark_index')}"
        elif rs > 0:
            summary += f"\n📈 Modest outperformance vs {result.get('benchmark_index')}"
        elif rs > -5:
            summary += f"\n📉 Slight underperformance vs {result.get('benchmark_index')}"
        else:
            summary += f"\n⚠️ Significant underperformance vs {result.get('benchmark_index')}"
    else:
        summary += f"""
❌ Relative strength data unavailable
{result.get('relative_strength_error', 'Unknown error')}"""
    
    summary += f"""

🔍 MARKET OVERVIEW:
Current market sentiment shows {result['trend'].lower()} bias with RSI at {result['rsi']:.1f} (using 27-80 range). Price is trading at ₹{result['current_price']:.2f} with the stock showing {result['signal'].lower()} characteristics.

📈 PRICE ACTION ANALYSIS:
The stock is positioned relative to key EMAs with EMA20 at {result['ema_20']}, EMA50 at {result['ema_50']}, and EMA200 at {result['ema_200']}. 

Key support zones are identified at ₹{result['support_levels'][0]:.2f} (primary), ₹{result['support_levels'][1]:.2f} (secondary), and ₹{result['support_levels'][2]:.2f} (tertiary). 

Resistance barriers are located at ₹{result['resistance_levels'][0]:.2f} (immediate), ₹{result['resistance_levels'][1]:.2f} (intermediate), and ₹{result['resistance_levels'][2]:.2f} (strong).

🔧 TECHNICAL INDICATORS:
RSI reading of {result['rsi']:.1f} suggests {"oversold conditions" if result['rsi'] < 27 else "overbought conditions" if result['rsi'] > 80 else "neutral momentum"}. MACD signals show {result['macd_line']} line against {result['macd_signal']} signal line.

🎯 PATTERN RECOGNITION:
Chart structure reveals {result['chart_pattern'].lower()} formation with {result['candlestick_pattern'].lower()} candlestick patterns. The overall trendline pattern appears {result['trendline_pattern'].lower()}.

💡 TRADING STRATEGY:
Recommended approach is {result['trading_style'].lower()} with {result['time_horizon']} time horizon. Current signal strength: {result['signal']}.

Risk management suggests stop loss at ₹{result['stop_loss']:.2f} with profit targets at ₹{result['targets'][0]:.2f}, ₹{result['targets'][1]:.2f}, and ₹{result['targets'][2]:.2f}.

⚠️ DISCLAIMER: This analysis is for educational purposes. Always conduct your own research and risk assessment before trading."""
    
    return summary

# Also update the main analysis message format in analyze_stock function to include relative strength
# Add this section after the existing main_analysis content:
def get_updated_main_analysis_text(result):
    """Updated main analysis text with relative strength"""
    
    main_analysis = f"""📊 {result['symbol']} Technical Analysis (3-Month Hourly)

🏢 Market Category: {result.get('market_cap_category', 'Unknown')}
📈 Benchmark: {result.get('benchmark_index', 'N/A')}"""
    
    if result.get('relative_strength') is not None:
        rs_emoji = "🚀" if result['relative_strength'] > 0 else "📉"
        stock_ret = result.get('stock_3m_return', 0)
        bench_ret = result.get('benchmark_3m_return', 0)
        rs_value = result['relative_strength']
        
        main_analysis += f"""
{rs_emoji} Relative Performance (3M):
  Stock: {stock_ret:+.2f}% | Benchmark: {bench_ret:+.2f}% | RS: {rs_value:+.2f}%"""
    
    main_analysis += f"""

💰 Market Data:
• Price: ₹{result['current_price']:.2f} | Trend: {result['trend']} | RSI: {result['rsi']:.1f}
• EMA20/50/200: {result['ema_20']}/{result['ema_50']}/{result['ema_200']}

🎯 Support Levels:
• S1: ₹{result['support_levels'][0]:.2f} | S2: ₹{result['support_levels'][1]:.2f} | S3: ₹{result['support_levels'][2]:.2f}

🚀 Resistance Levels:
• R1: ₹{result['resistance_levels'][0]:.2f} | R2: ₹{result['resistance_levels'][1]:.2f} | R3: ₹{result['resistance_levels'][2]:.2f}

🔥 Signal: {result.get('enhanced_signal', result['signal'])}
⏰ Strategy: {result['trading_style']} | Time: {result['time_horizon']}

🕯️ Patterns:
• Chart: {result['chart_pattern']} | Candle: {result['candlestick_pattern']}
• Trendline: {result['trendline_pattern']}"""
    
    return main_analysis

def generate_enhanced_signal_with_relative_strength(result):
    """Generate enhanced signal considering relative strength"""
    base_signal = result['signal']
    rs = result.get('relative_strength', 0)
    
    # Enhance signal based on relative strength
    if rs > 10:  # Strong outperformance
        if base_signal in ['BUY', 'STRONG BUY']:
            enhanced_signal = 'STRONG BUY (RS+)'
        elif base_signal == 'HOLD':
            enhanced_signal = 'BUY (RS+)'
        else:
            enhanced_signal = base_signal
    elif rs < -10:  # Strong underperformance
        if base_signal in ['SELL', 'STRONG SELL']:
            enhanced_signal = 'STRONG SELL (RS-)'
        elif base_signal == 'HOLD':
            enhanced_signal = 'SELL (RS-)'
        else:
            enhanced_signal = base_signal
    else:
        enhanced_signal = base_signal
    
    return enhanced_signal

# TELEGRAM BOT HANDLERS

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = """
🚀 *NSE Stock Technical Analysis Bot*

📈 *Enhanced Features:*
• 3-month hourly chart analysis
• Multiple support & resistance levels
• Chart pattern recognition
• Trading style recommendations
• Time horizon specifications

💡 *How to use:*
• Send me any NSE stock symbol (e.g., RELIANCE, TCS, INFY)
• Get detailed technical analysis with trading recommendations
• Receive entry/exit levels with time-based strategies

🔥 *Examples:*
• RELIANCE
• TCS
• INFY
• HDFCBANK

📊 Just send the ticker symbol and I'll provide comprehensive analysis!
    """
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

# Updated analyze_stock function with safer message formatting
async def analyze_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle stock analysis requests with enhanced analysis - SAFE VERSION"""
    symbol = update.message.text.strip().upper()
    
    # Basic validation
    if len(symbol) > 20 or not symbol.isalpha():
        await update.message.reply_text("❌ Please send a valid NSE stock symbol (e.g., RELIANCE, TCS)")
        return
    
    # Send initial message
    status_message = await update.message.reply_text(f"🔍 Analyzing {symbol}...\n⏳ Downloading 3-month charts...")
    
    try:
        # Download charts with 3-month range
        chart_data = fetch_charts(f"NSE:{symbol}", interval='1h')
        
        if not chart_data:
            await status_message.edit_text(f"❌ Could not download charts for {symbol}. Please check the symbol and try again.")
            return
        
        await status_message.edit_text(f"🔍 Analyzing {symbol}...\n🤖 AI processing 3-month charts...")
        
        # Get relative strength data first for AI context
        category, benchmark_symbol = get_stock_market_cap_category(symbol)
        relative_strength_context = None

        if category and benchmark_symbol:
            rel_strength = calculate_relative_strength(symbol, benchmark_symbol)
            if rel_strength:
                relative_strength_context = {
                    'category': category,
                    'benchmark_name': get_benchmark_name(benchmark_symbol),
                    'stock_return': rel_strength['stock_return'],
                    'benchmark_return': rel_strength['benchmark_return'],
                    'relative_strength': rel_strength['relative_strength'],
                    'outperforming': rel_strength['outperforming']
                }

        # Analyze charts with enhanced AI prompts including relative strength
        price_data = {}
        tech_data = {}

        for chart in chart_data:
            chart_type = chart['type']
            analysis = analyze_chart_with_groq_telegram(chart['data'], chart_type, relative_strength_context)
            
            if analysis:
                if chart_type == 'price_emas':
                    price_data = analysis
                else:
                    tech_data = analysis
        
        await status_message.edit_text(f"🔍 Analyzing {symbol}...\n🎯 Generating trading strategy...")
        
        # Generate enhanced trading analysis
        result = generate_trading_analysis_with_relative_strength(price_data, tech_data, symbol)
        result['price_analysis'] = format_analysis_text_telegram(result.get('price_analysis', ''))
        result['tech_analysis'] = format_analysis_text_telegram(result.get('tech_analysis', ''))
        result['enhanced_signal'] = generate_enhanced_signal_with_relative_strength(result)
        
        # Send charts first
        await status_message.edit_text(f"📈 Analysis complete for {symbol}!\n📤 Sending charts...")
        
        for chart in chart_data:
            caption = "📈 3-Month Price & EMAs Chart" if chart['type'] == 'price_emas' else "📊 3-Month RSI & MACD Chart"
            await update.message.reply_photo(
                photo=io.BytesIO(chart['data']),
                caption=caption,
                filename=chart['filename']
            )
        
        # Create main analysis message - PLAIN TEXT VERSION (no Markdown)
        main_analysis = get_updated_main_analysis_text(result)
        
        # Entry and targets message
        entry_analysis = f"""📈 Entry Strategies:
"""
        for level, price in result['entry_levels'].items():
            entry_analysis += f"• {level.title()}: ₹{price:.2f}\n"
        
        entry_analysis += f"""
🎯 Targets: ₹{result['targets'][0]:.2f} | ₹{result['targets'][1]:.2f} | ₹{result['targets'][2]:.2f}
⛔ Stop Loss: ₹{result['stop_loss']:.2f}"""
        
        # AI Analysis sections
        ai_analysis = f"""📈 AI Price Analysis:
{result['price_analysis']}

📊 AI Technical Analysis:
{result['tech_analysis']}

💡 Note: Analysis based on 3-month hourly data"""
        
        # Delete status message and send analysis in parts
        await status_message.delete()
        
        # Send main analysis (plain text, no Markdown)
        await update.message.reply_text(main_analysis)
        
        # Send entry analysis (plain text, no Markdown)
        await update.message.reply_text(entry_analysis)
        
        # Send AI analysis (this might be longer, so use the split function)
        await split_and_send_message(update, ai_analysis)
        
        # Send comprehensive summary analysis
        summary_text = create_summary_analysis_with_relative_strength(result)
        await asyncio.sleep(1)  # Small delay
        await update.message.reply_text("📋 COMPREHENSIVE SUMMARY")
        await split_and_send_message(update, summary_text)
        
    except Exception as e:
        error_msg = f"❌ Analysis failed for {symbol}: {str(e)}"
        print(f"Error in analyze_stock: {e}")  # Add logging to terminal
        try:
            await status_message.edit_text(error_msg)
        except:
            # If editing fails, send a new message
            await update.message.reply_text(error_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message."""
    help_text = """
🚀 *NSE Stock Technical Analysis Bot Help*

📊 *Features:*
• 3-month hourly chart analysis
• Multiple support & resistance levels
• AI-powered pattern recognition
• Trading recommendations with time horizons

💡 *Commands:*
• /start - Start the bot
• /help - Show this help message
• Just send any NSE stock symbol for analysis

🔥 *Valid Symbols:*
• RELIANCE, TCS, INFY, HDFCBANK
• WIPRO, ICICIBANK, SBIN, LT
• BAJFINANCE, ASIANPAINT, etc.

📈 *Analysis Includes:*
• Current price & trend direction
• 3 support and 3 resistance levels
• RSI, MACD, EMA indicators
• Chart patterns & candlestick patterns
• Entry strategies (aggressive/moderate/conservative)
• Target levels and stop loss

⚠️ *Disclaimer:* This is for educational purposes only. Please do your own research before making investment decisions.
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all text messages as stock symbol requests."""
    await analyze_stock(update, context)

def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    print("🚀 Starting NSE Technical Analysis Bot...")
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    print("✅ Bot is running. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()