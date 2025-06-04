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

load_dotenv()

# Configuration
CHART_API_KEY = os.getenv("CHART_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Add this to your .env file

# Keep all your existing functions exactly as they are
def fetch_charts(symbol, interval='1h'):
    """Download 2 charts within free tier limits"""
    headers = {
        "x-api-key": CHART_API_KEY,
        "Content-Type": "application/json"
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Chart 1: Price + EMAs + Volume
    chart1_payload = {
        "symbol": symbol,
        "interval": interval,
        "studies": [
            {"name": "Moving Average Exponential", "inputs": {"length": 20}},
            {"name": "Moving Average Exponential", "inputs": {"length": 50}},
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
    """Analyze chart using Groq API - modified for telegram (takes raw data)"""
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
            prompt = """Analyze this price chart and extract:
1. Current stock price (exact number)
2. EMA 20 value
3. EMA 50 value  
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
        # Try multiple RSI patterns
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
                    if 0 <= rsi_val <= 100 and rsi_val > 10:
                        result['rsi'] = rsi_val
                        break
                except ValueError:
                    continue
        
        macd_match = re.search(r'macd.*?line.*?(\d+\.?\d*)', text_lower)
        if macd_match:
            result['macd_line'] = float(macd_match.group(1))
    
    return result

def generate_trading_analysis(price_data, tech_data, symbol):
    """Generate trading recommendations - same as your existing function"""
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

def format_analysis_text_telegram(text):
    """Format analysis text for Telegram (no HTML)"""
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
        
        # Add emojis based on content
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
        
        lines.append(f'{icon} {line}')
    
    return '\n'.join(lines)

# TELEGRAM BOT HANDLERS

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = """
🚀 *NSE Stock Technical Analysis Bot*

📈 *How to use:*
• Send me any NSE stock symbol (e.g., RELIANCE, TCS, INFY)
• I'll analyze the charts and provide detailed technical analysis
• Get trading recommendations with entry/exit levels

💡 *Examples:*
• RELIANCE
• TCS
• INFY
• HDFCBANK

🔥 Just send the ticker symbol and I'll do the rest!
    """
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

async def analyze_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle stock analysis requests"""
    symbol = update.message.text.strip().upper()
    
    # Basic validation
    if len(symbol) > 20 or not symbol.isalpha():
        await update.message.reply_text("❌ Please send a valid NSE stock symbol (e.g., RELIANCE, TCS)")
        return
    
    # Send initial message
    status_message = await update.message.reply_text(f"🔍 Analyzing {symbol}...\n⏳ Downloading charts...")
    
    try:
        # Download charts
        chart_data = fetch_charts(f"NSE:{symbol}", interval='1h')
        
        if not chart_data:
            await status_message.edit_text(f"❌ Could not download charts for {symbol}. Please check the symbol and try again.")
            return
        
        await status_message.edit_text(f"🔍 Analyzing {symbol}...\n📊 Processing charts with AI...")
        
        # Analyze charts
        price_data = {}
        tech_data = {}
        
        for chart in chart_data:
            chart_type = chart['type']
            analysis = analyze_chart_with_groq_telegram(chart['data'], chart_type)
            
            if analysis:
                if chart_type == 'price_emas':
                    price_data = analysis
                else:
                    tech_data = analysis
        
        await status_message.edit_text(f"🔍 Analyzing {symbol}...\n🎯 Generating recommendations...")
        
        # Generate trading analysis
        result = generate_trading_analysis(price_data, tech_data, symbol)
        result['price_analysis'] = format_analysis_text_telegram(result.get('price_analysis', ''))
        result['tech_analysis'] = format_analysis_text_telegram(result.get('tech_analysis', ''))
        
        # Send charts first
        await status_message.edit_text(f"📈 Analysis complete for {symbol}!\n📤 Sending charts...")
        
        for chart in chart_data:
            caption = "📈 Price & EMAs Chart" if chart['type'] == 'price_emas' else "📊 RSI & MACD Chart"
            await update.message.reply_photo(
                photo=io.BytesIO(chart['data']),
                caption=caption,
                filename=chart['filename']
            )
        
        # Format and send analysis
        analysis_text = f"""
📊 *{result['symbol']} Technical Analysis*

💰 *Current Data:*
• Price: ₹{result['current_price']:.2f}
• Trend: {result['trend']}
• RSI: {result['rsi']:.1f}
• EMA 20: {result['ema_20']}
• EMA 50: {result['ema_50']}
• MACD: {result['macd_line']}
• Support: ₹{result['support']:.2f}
• Resistance: ₹{result['resistance']:.2f}

🎯 *Trading Signal: {result['signal']}*

📈 *Entry Levels:*
"""
        
        for level, price in result['entry_levels'].items():
            analysis_text += f"• {level.title()}: ₹{price:.2f}\n"
        
        analysis_text += f"""
🎯 *Targets:*
• Target 1: ₹{result['targets'][0]:.2f}
• Target 2: ₹{result['targets'][1]:.2f}
• Target 3: ₹{result['targets'][2]:.2f}

⛔ *Stop Loss:* ₹{result['stop_loss']:.2f}

📈 *Price & EMAs Analysis:*
{result['price_analysis']}

📊 *Technical Indicators Analysis:*
{result['tech_analysis']}
        """
        
        await status_message.delete()
        await update.message.reply_text(analysis_text, parse_mode='Markdown')
        
    except Exception as e:
        await status_message.edit_text(f"❌ Analysis failed for {symbol}: {str(e)}")
        print(f"Analysis error: {e}")

def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment variables!")
        return
    
    print("🚀 Starting NSE Stock Analysis Telegram Bot...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_stock))
    
    # Run the bot
    print("✅ Bot is running! Send /start to begin.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()