import os
import datetime
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import base64
import json
import re
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import io
import yfinance as yf
import warnings


def format_value(value, format_type="currency", decimal_places=2, fallback="N/A"):
    """Universal value formatter with type-specific formatting"""
    if not isinstance(value, (int, float)):
        return fallback
    
    if format_type == "currency" and value > 0:
        return f"₹{value:.{decimal_places}f}"
    elif format_type == "percent":
        return f"{value:+.{decimal_places}f}%"
    elif format_type == "number":
        return f"{value:.{decimal_places}f}"
    elif format_type == "currency" and value <= 0:
        return fallback
    
    return fallback

def format_error(operation, identifier, error):
    """Standardized error message formatting"""
    return f"❌ {operation} failed for {identifier}: {error}"

def format_api_error(service, status_code):
    """Standardized API error formatting"""
    return f"❌ {service} API error: HTTP {status_code}"

load_dotenv()

def validate_api_keys():
    """Validate all required API keys are present"""
    missing_keys = []
    
    if not CHART_API_KEY:
        missing_keys.append("CHART_API_KEY")
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        missing_keys.append("TELEGRAM_BOT_TOKEN")
    
    if missing_keys:
        print(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
        return False
    return True

# Configuration
CHART_API_KEY = os.getenv("CHART_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4000  # Leave some buffer under 4096 limit
EXCHANGE_RATE_USD_TO_INR = 83
LARGE_CAP_THRESHOLD = 20000  # crores
MID_CAP_THRESHOLD = 5000     # crores

## REMOVE THIS IF ALL CONCLUSION DO NOT EXIST.
def extract_all_sections(text, section_keywords):
    """Extract ALL sections matching keywords - returns list of all matches"""
    found_sections = []
    
    for keyword in section_keywords:
        # Pattern 1: **KEYWORD**: or **KEYWORD** followed by content
        pattern1 = rf'\*\*{re.escape(keyword)}\*\*:?\s*(.*?)(?=\n\*\*[A-Z]|\n##|\n\n\*\*|\Z)'
        matches = re.finditer(pattern1, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            content = match.group(1).strip()
            content = re.sub(r'\n+', '\n', content)
            if content and len(content) > 10:
                found_sections.append({
                    'keyword': keyword,
                    'content': content,
                    'pattern': 'bold_keyword'
                })
        
        # Pattern 2: ## KEYWORD followed by content
        pattern2 = rf'##\s*{re.escape(keyword)}\s*(.*?)(?=\n##|\n\*\*|\Z)'
        matches = re.finditer(pattern2, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            content = match.group(1).strip()
            content = re.sub(r'\n+', '\n', content)
            if content and len(content) > 10:
                found_sections.append({
                    'keyword': keyword,
                    'content': content,
                    'pattern': 'header_keyword'
                })
        
        # Pattern 3: KEYWORD: at start of line
        pattern3 = rf'^{re.escape(keyword)}:\s*(.*?)(?=\n[A-Z][A-Z\s]*:|\n\n|\Z)'
        matches = re.finditer(pattern3, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for match in matches:
            content = match.group(1).strip()
            content = re.sub(r'\n+', '\n', content)
            if content and len(content) > 10:
                found_sections.append({
                    'keyword': keyword,
                    'content': content,
                    'pattern': 'line_start'
                })
        
        # Pattern 4: Look for content after keyword anywhere in text
        pattern4 = rf'{re.escape(keyword)}[:\s]+(.*?)(?=\n\n|\Z)'
        matches = re.finditer(pattern4, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            content = match.group(1).strip()
            content = re.sub(r'\n+', '\n', content)
            if content and len(content) > 10:
                # Check if this content is already found by other patterns
                is_duplicate = any(
                    abs(len(content) - len(section['content'])) < 20 and 
                    content[:50] == section['content'][:50] 
                    for section in found_sections
                )
                if not is_duplicate:
                    found_sections.append({
                        'keyword': keyword,
                        'content': content,
                        'pattern': 'anywhere'
                    })
    
    return found_sections



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
                    print(format_api_error("Chart", response.status_code))
                    
            except Exception as e:
                print(f"Error downloading chart {i}: {e}")
                continue
        
        return chart_data if chart_data else None
            
    except Exception as e:
        print(f"Chart download error: {e}")
        return None

def analyze_chart_with_groq_telegram(chart_data, chart_type):
    """Analyze chart using Groq API with enhanced prompts for comprehensive analysis"""
    try:        
        # Convert raw image data to base64
        base64_image = base64.b64encode(chart_data).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Updated prompt to match the desired output format
        if 'price_emas' in chart_type:
            prompt = """Analyze this 3-month hourly candlestick chart and provide detailed technical analysis:

CURRENT CONTEXT (Extract exact values):
1. Current Price: [exact price from chart]
2. Today's High: [if visible]
3. Today's Low: [if visible]
4. EMA 20 value: [exact number]
5. EMA 50 value: [exact number]
6. EMA 200 value: [exact number]

PATTERN IDENTIFICATION:
7. Chart Pattern: [Flag, Triangle, Channel, Head & Shoulders, Wedge, etc.]
8. Candlestick Pattern: [Doji, Hammer, Engulfing, Spinning Top, etc.]
9. Pattern Description: [Brief description of what the pattern suggests]

KEY LEVELS (Provide 3 each with exact prices):
10. Resistance Level 1: [strongest/nearest resistance]
11. Resistance Level 2: [intermediate resistance]
12. Resistance Level 3: [major resistance]
13. Support Level 1: [strongest/nearest support]
14. Support Level 2: [intermediate support]
15. Support Level 3: [major support]

VOLUME ANALYSIS:
16. Volume Trend: [High/Low/Average compared to recent periods]
17. Volume Pattern: [Description of volume behavior]

Provide specific numerical values and clear pattern identification."""
        else:
            prompt = """Analyze this 3-month hourly technical indicators chart:

TECHNICAL INDICATORS (Extract exact values):
1. RSI Value: [0-100, exact number]
2. RSI Status: [Oversold <30, Neutral 30-70, Overbought >70]
3. MACD Line: [exact value if visible]
4. MACD Signal Line: [exact value if visible]
5. MACD Histogram: [Positive/Negative/Neutral trend]

MOMENTUM ANALYSIS:
6. Overall Momentum: [Increasing/Decreasing/Neutral]
7. RSI Trend: [Rising/Falling/Sideways]
8. MACD Trend: [Bullish/Bearish/Neutral]

DIVERGENCES:
9. RSI Divergence: [Any divergence with price action]
10. MACD Divergence: [Any divergence with price action]

VOLUME:
11. Volume Trend: [High/Low/Average volume]
12. Volume Pattern: [Description]

Provide specific numbers and clear momentum assessment."""
        
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
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                               headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            return parse_analysis(content, chart_type)
        else:
            print(format_api_error("Groq", response.status_code))
            return None
            
    except Exception as e:
        print(f"Groq analysis error: {e}")
        return None


def parse_analysis(text, chart_type):
    """Parse analysis text and extract data - ENHANCED VERSION to extract ALL sections"""
    result = {'raw_analysis': text}
    text_lower = text.lower()
    
    if 'price_emas' in chart_type:
        # Extract price data - FIXED patterns
        price_patterns = [
            r'current\s+price[:\s*]+[₹$]?(\d+\.?\d*)',
            r'price[:\s*]+[₹$]?(\d+\.?\d*)',
            r'[₹$](\d+\.?\d*)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result['current_price'] = float(match.group(1))
                break
        
        # Extract EMAs - MORE FLEXIBLE patterns
        ema_search_patterns = {
            'ema_20': [
                r'ema\s*20[^0-9]*?(\d+\.?\d*)',
                r'20\s*ema[^0-9]*?(\d+\.?\d*)',
                r'ema20[^0-9]*?(\d+\.?\d*)',
                r'20-period\s*ema[^0-9]*?(\d+\.?\d*)',
                r'20\s*period[^0-9]*?(\d+\.?\d*)'
            ],
            'ema_50': [
                r'ema\s*50[^0-9]*?(\d+\.?\d*)',
                r'50\s*ema[^0-9]*?(\d+\.?\d*)',
                r'ema50[^0-9]*?(\d+\.?\d*)',
                r'50-period\s*ema[^0-9]*?(\d+\.?\d*)',
                r'50\s*period[^0-9]*?(\d+\.?\d*)'
            ],
            'ema_200': [
                r'ema\s*200[^0-9]*?(\d+\.?\d*)',
                r'200\s*ema[^0-9]*?(\d+\.?\d*)',
                r'ema200[^0-9]*?(\d+\.?\d*)',
                r'200-period\s*ema[^0-9]*?(\d+\.?\d*)',
                r'200\s*period[^0-9]*?(\d+\.?\d*)'
            ]
        }
        
        for ema_key, patterns in ema_search_patterns.items():
            found = False
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                valid_matches = [float(m) for m in matches if 1 <= float(m) <= 50000]
                if valid_matches:
                    result[ema_key] = valid_matches[0]
                    found = True
                    break
            
            if not found:
                print(f"❌ No valid {ema_key} found in text")
        
        # Extract support and resistance levels - MORE FLEXIBLE patterns
        resistance_levels = []
        support_levels = []
        
        resistance_patterns = [
            r'resistance[^0-9]*?(\d+\.?\d*)',
            r'resistance\s*level[^0-9]*?(\d+\.?\d*)',
            r'resistance[^0-9]*?[₹$](\d+\.?\d*)',
            r'target[^0-9]*?(\d+\.?\d*)',
            r'upside[^0-9]*?(\d+\.?\d*)'
        ]
        
        support_patterns = [
            r'support[^0-9]*?(\d+\.?\d*)',
            r'support\s*level[^0-9]*?(\d+\.?\d*)',
            r'support[^0-9]*?[₹$](\d+\.?\d*)',
            r'downside[^0-9]*?(\d+\.?\d*)',
            r'floor[^0-9]*?(\d+\.?\d*)'
        ]
        
        # Extract resistance levels
        for pattern in resistance_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    level = float(match)
                    if 1 <= level <= 50000 and level not in resistance_levels:
                        resistance_levels.append(level)
                except ValueError as e:
                    print(f"⚠️ Failed to parse number: {e}")
                    continue
        
        # Extract support levels
        for pattern in support_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    level = float(match)
                    if 1 <= level <= 50000 and level not in support_levels:
                        support_levels.append(level)
                except ValueError as e:
                    print(f"⚠️ Failed to parse number: {e}")
                    continue
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        support_levels = sorted(list(set(support_levels)), reverse=True)
        
        result['resistance_levels'] = resistance_levels
        result['support_levels'] = support_levels
        
        # Extract patterns
        chart_patterns = ['ascending channel', 'flag', 'triangle', 'channel', 'head and shoulders', 'wedge', 'pennant', 'cup and handle', 'double top', 'double bottom']
        for pattern in chart_patterns:
            if pattern in text_lower:
                result['chart_pattern'] = pattern.title()
                break
        
        candle_patterns = ['bullish engulfing', 'doji', 'hammer', 'engulfing', 'spinning top', 'shooting star', 'hanging man', 'marubozu']
        for pattern in candle_patterns:
            if pattern in text_lower:
                result['candlestick_pattern'] = pattern.title()
                break
        
        # Extract ALL conclusion-like sections for price analysis
        conclusion_keywords = [
            'conclusion', 'summary', 'recommendation', 'recommendations', 
            'additional observations', 'key takeaways', 'final thoughts',
            'outlook', 'overall assessment', 'bottom line', 'analysis summary'
        ]
        
        all_conclusions = extract_all_sections(text, conclusion_keywords)
        if all_conclusions:
            # Store all conclusions
            result['all_conclusions'] = all_conclusions
            # Keep the first one as main conclusion for backward compatibility
            result['conclusion'] = all_conclusions[0]['content']
        
    else:  # Technical indicators chart
        # Extract RSI - MORE FLEXIBLE patterns
        rsi_patterns = [
            r'rsi[^0-9]*?(\d+\.?\d*)',
            r'relative\s*strength[^0-9]*?(\d+\.?\d*)',
            r'rsi\s*value[^0-9]*?(\d+\.?\d*)',
            r'rsi[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in rsi_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    rsi_val = float(match)
                    if 0 <= rsi_val <= 100:
                        result['rsi'] = rsi_val
                        break
                except ValueError as e:
                    print(f"⚠️ Failed to parse number: {e}")
                    continue
            if 'rsi' in result:
                break
        
        # Extract MACD - MORE FLEXIBLE patterns
        macd_line_patterns = [
            r'macd\s*line[^0-9]*?(\d+\.?\d*)',
            r'macd[^0-9]*?(\d+\.?\d*)',
            r'macd\s*value[^0-9]*?(\d+\.?\d*)'
        ]
        
        for pattern in macd_line_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    result['macd_line'] = float(matches[0])
                    break
                except ValueError as e:
                    print(f"⚠️ Failed to parse number: {e}")
                    continue
        
        macd_signal_patterns = [
            r'macd\s*signal[^0-9]*?(\d+\.?\d*)',
            r'signal\s*line[^0-9]*?(\d+\.?\d*)',
            r'signal[^0-9]*?(\d+\.?\d*)'
        ]
        
        for pattern in macd_signal_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    result['macd_signal'] = float(matches[0])
                    break
                except ValueError as e:
                    print(f"⚠️ Failed to parse number: {e}")
                    continue
        
        # Extract ALL summary-like sections for technical analysis
        summary_keywords = [
            'conclusion', 'summary', 'recommendation', 'recommendations',
            'additional observations', 'key takeaways', 'final thoughts',
            'outlook', 'overall assessment', 'bottom line', 'analysis summary',
            'technical summary', 'momentum analysis', 'indicator summary'
        ]
        
        all_summaries = extract_all_sections(text, summary_keywords)
        if all_summaries:
            # Store all summaries
            result['all_summaries'] = all_summaries
            # Keep the first one as main summary for backward compatibility
            result['summary'] = all_summaries[0]['content']
    
    return result


def get_stock_market_cap_category(symbol):
    """Dynamically determine stock category using alternative free API"""
    try:        
        # Use Yahoo Finance single request without retries
        nse_symbol = f"{symbol}.NS"
        stock = yf.Ticker(nse_symbol)
        
        # Single attempt only with error suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = stock.info
        
        market_cap_usd = info.get('marketCap', 0)
        
        if market_cap_usd and market_cap_usd > 0:
            market_cap_inr = market_cap_usd * 83
            market_cap_crores = market_cap_inr / 10000000
            
            if market_cap_crores >= 20000:
                return "Large Cap", "^NSEI"
            elif market_cap_crores >= 5000:
                return "Mid Cap", "^NSEMDCP50"
            else:
                return "Small Cap", "^CNXSC"
        
        # Default fallback
        return "Large Cap", "^NSEI"
        
    except Exception as e:
        print(f"⚠️ Market cap category failed for {symbol}: {e}")
        return "Large Cap", "^NSEI"


def calculate_relative_strength(stock_symbol, benchmark_symbol, period='3mo'):
    """Calculate relative strength using single API call"""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        # Single attempt, no retries with error suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stock_data = yf.download(f"{stock_symbol}.NS", period=period, interval='1d', progress=False)
            if stock_data.empty:
                stock_data = yf.download(f"{stock_symbol}.BO", period=period, interval='1d', progress=False)
            
            benchmark_data = yf.download(benchmark_symbol, period=period, interval='1d', progress=False)
        
        if stock_data.empty or benchmark_data.empty:
            return None
        
        # Calculate returns
        stock_return = ((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100
        benchmark_return = ((benchmark_data['Close'].iloc[-1] - benchmark_data['Close'].iloc[0]) / benchmark_data['Close'].iloc[0]) * 100
        
        return {
            'stock_return': stock_return,
            'benchmark_return': benchmark_return,
            'relative_strength': stock_return - benchmark_return,
            'outperforming': stock_return > benchmark_return
        }
        
    except Exception as e:
        print(f"⚠️ Relative strength calculation failed for {stock_symbol}: {e}")
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


def create_new_analysis_report(price_data, tech_data, symbol, relative_strength_data=None):
    """Create the new analysis report matching the desired format - ENHANCED VERSION"""
    
    # Extract data with defaults and validation
    current_price = price_data.get('current_price', 0)
    ema_20 = price_data.get('ema_20', 0)
    ema_50 = price_data.get('ema_50', 0) 
    ema_200 = price_data.get('ema_200', 0)
    rsi = tech_data.get('rsi', 50)
    
    support_levels = price_data.get('support_levels', [])
    resistance_levels = price_data.get('resistance_levels', [])
    
    # Ensure we have 3 levels each (generate reasonable estimates if needed)
    if len(support_levels) < 3:
        base_supports = []
        if isinstance(current_price, (int, float)) and current_price > 0:
            base_supports = [
                current_price * 0.95,  # 5% below current
                current_price * 0.90,  # 10% below current  
                current_price * 0.85   # 15% below current
            ]
        else:
            base_supports = [0, 0, 0]
        
        # Fill missing support levels
        while len(support_levels) < 3:
            idx = len(support_levels)
            if idx < len(base_supports):
                support_levels.append(base_supports[idx])
            else:
                support_levels.append(0)
    
    if len(resistance_levels) < 3:
        base_resistances = []
        if isinstance(current_price, (int, float)) and current_price > 0:
            base_resistances = [
                current_price * 1.05,  # 5% above current
                current_price * 1.10,  # 10% above current
                current_price * 1.15   # 15% above current
            ]
        else:
            base_resistances = [0, 0, 0]
        
        # Fill missing resistance levels
        while len(resistance_levels) < 3:
            idx = len(resistance_levels)
            if idx < len(base_resistances):
                resistance_levels.append(base_resistances[idx])
            else:
                resistance_levels.append(0)
    
    # Get pattern descriptions with context
    chart_pattern = price_data.get('chart_pattern', 'Consolidation/Sideways - price moving in a range without clear direction')
    candle_pattern = price_data.get('candlestick_pattern', 'Mixed signals - combination of bullish and bearish candles showing indecision')
    
    # Get volume analysis
    volume_analysis = price_data.get('volume_analysis') or tech_data.get('volume_analysis')
    if not volume_analysis:
        volume_analysis = "Recent volume patterns show normal trading activity with occasional spikes during price movements"
    
    # Determine RSI status and bias
    if isinstance(rsi, (int, float)):
        if rsi < 30:
            rsi_status = "oversold"
            bias = "BUY"
            timeframe_rec = "Swing Trade"
            duration = "1-3 weeks"
        elif rsi > 70:
            rsi_status = "overbought" 
            bias = "SELL"
            timeframe_rec = "Swing Trade"
            duration = "1-2 weeks"
        else:
            rsi_status = "neutral"
            bias = "Range Trade"
            timeframe_rec = "Intraday"
            duration = "Today"
    else:
        rsi_status = "neutral"
        bias = "HOLD"
        timeframe_rec = "Wait & Watch"
        duration = "TBD"

    # Build the report with safe formatting
    report = f"""**Current Context**
   • **Price:** {format_value(current_price, "currency")}
      • **EMAs:**
         • 20 EMA ≃ {format_value(ema_20, "currency")} | 50 EMA ≃ {format_value(ema_50, "currency")} | 200 EMA ≃ {format_value(ema_200, "currency")}
      • **RSI (14):** ~{format_value(rsi, "number", 1)} ({rsi_status})"""

    # Add relative strength section if available
    if relative_strength_data:
        rs = relative_strength_data.get('relative_strength', 0)
        stock_ret = relative_strength_data.get('stock_return', 0)
        bench_ret = relative_strength_data.get('benchmark_return', 0)
        benchmark_name = relative_strength_data.get('benchmark_name', 'Benchmark')
        
        report += f"""
      • **Relative Strength (3M):** {format_value(rs, "percent", 1)} vs {benchmark_name}
         • Stock: {format_value(stock_ret, "percent", 1)} | Benchmark: {format_value(bench_ret, "percent", 1)}"""

    report += f"""
      • **Volume:** {volume_analysis}

   • **Pattern**
      • **Chart Pattern:** {chart_pattern}
      • **Candles:** {candle_pattern}

   • **Key Levels**
      • **Resistance:**
        1. {format_value(resistance_levels[0], "currency")}
        2. {format_value(resistance_levels[1], "currency")}
        3. {format_value(resistance_levels[2], "currency")}
      • **Support:**
        1. {format_value(support_levels[0], "currency")}
        2. {format_value(support_levels[1], "currency")}
        3. {format_value(support_levels[2], "currency")}

   • **Trade Triggers & Targets**"""

    # Generate trade recommendations based on bias
    if bias == "BUY":
        # Calculate entry points safely
        entry_aggressive = current_price * 0.995 if isinstance(current_price, (int, float)) and current_price > 0 else None
        entry_conservative = support_levels[0] if isinstance(support_levels[0], (int, float)) and support_levels[0] > 0 else None
        sl1 = support_levels[0] * 0.995 if isinstance(support_levels[0], (int, float)) and support_levels[0] > 0 else None
        sl2 = support_levels[1] * 0.995 if isinstance(support_levels[1], (int, float)) and support_levels[1] > 0 else None
        
        report += f"""
      • **Bullish ({timeframe_rec}, {duration}):**
         • **Entry:**
            • Aggressive: {format_value(entry_aggressive, "currency")} (current levels)
            • Conservative: {format_value(entry_conservative, "currency")} (on dip to support)
         • **Stop:**
            • If buying current: SL < {format_value(support_levels[0], "currency")} (e.g. {format_value(sl1, "currency")})
            • If buying dip: SL < {format_value(support_levels[1], "currency")} (e.g. {format_value(sl2, "currency")})
         • **Targets:**
            1. {format_value(resistance_levels[0], "currency")}
            2. {format_value(resistance_levels[1], "currency")}
            3. {format_value(resistance_levels[2], "currency")}"""
    
    elif bias == "SELL":
        # Calculate entry points safely
        entry_aggressive = current_price * 1.005 if isinstance(current_price, (int, float)) and current_price > 0 else None
        entry_conservative = resistance_levels[0] if isinstance(resistance_levels[0], (int, float)) and resistance_levels[0] > 0 else None
        sl1 = resistance_levels[0] * 1.005 if isinstance(resistance_levels[0], (int, float)) and resistance_levels[0] > 0 else None
        sl2 = resistance_levels[1] * 1.005 if isinstance(resistance_levels[1], (int, float)) and resistance_levels[1] > 0 else None
        
        report += f"""
      • **Bearish ({timeframe_rec}, {duration}):**
         • **Entry:**
            • Aggressive: {format_value(entry_aggressive, "currency")} (current levels)
            • Conservative: {format_value(entry_conservative, "currency")} (on bounce to resistance)
         • **Stop:**
            • If shorting current: SL > {format_value(resistance_levels[0], "currency")} (e.g. {format_value(sl1, "currency")})
            • If shorting bounce: SL > {format_value(resistance_levels[1], "currency")} (e.g. {format_value(sl2, "currency")})
         • **Targets:**
            1. {format_value(support_levels[0], "currency")}
            2. {format_value(support_levels[1], "currency")}
            3. {format_value(support_levels[2], "currency")}"""
    
    else:  # Range Trade or HOLD
        # Calculate range trade levels safely
        long_entry = support_levels[0] if isinstance(support_levels[0], (int, float)) and support_levels[0] > 0 else None
        long_sl = support_levels[0] * 0.995 if isinstance(support_levels[0], (int, float)) and support_levels[0] > 0 else None
        long_target = resistance_levels[0] if isinstance(resistance_levels[0], (int, float)) and resistance_levels[0] > 0 else None
        
        short_entry = resistance_levels[0] if isinstance(resistance_levels[0], (int, float)) and resistance_levels[0] > 0 else None
        short_sl = resistance_levels[0] * 1.005 if isinstance(resistance_levels[0], (int, float)) and resistance_levels[0] > 0 else None
        short_target = support_levels[0] if isinstance(support_levels[0], (int, float)) and support_levels[0] > 0 else None
        
        report += f"""
      • **Intraday (Today's Range):**
         • **Long near {format_value(long_entry, "currency")}** (stop {format_value(long_sl, "currency")}) → target {format_value(long_target, "currency")}
         • **Short near {format_value(short_entry, "currency")}** (stop {format_value(short_sl, "currency")}) → target {format_value(short_target, "currency")}"""

    # Calculate bottom line values safely
    resistance_1_str = format_value(resistance_levels[0], "currency") if isinstance(resistance_levels[0], (int, float)) and resistance_levels[0] > 0 else "key resistance"
    support_1_str = format_value(support_levels[0], "currency") if isinstance(support_levels[0], (int, float)) and support_levels[0] > 0 else "key support"
    
    report += f"""

   • **Overall Recommendation**
      • **Primary:** {bias} - {timeframe_rec} ({duration})
      • **Risk Management:** Use appropriate position sizing and stop losses
      • **Monitor:** Key level breaks for trend continuation/reversal

   • **Bottom Line**
      • RSI at {format_value(rsi, "number", 1)} suggests {'oversold bounce potential' if rsi_status == 'oversold' else 'overbought correction risk' if rsi_status == 'overbought' else 'neutral momentum - watch for directional break'}
      • Watch for hourly close > {resistance_1_str} (continue up) or < {support_1_str} (deeper pullback)
      • Pattern: {chart_pattern.split(' - ')[0].lower()} - {'bullish continuation expected' if bias == 'BUY' else 'bearish continuation expected' if bias == 'SELL' else 'range-bound action likely'}"""

    # ENHANCED: Add ALL conclusion and summary sections
    all_price_conclusions = price_data.get('all_conclusions', [])
    all_tech_summaries = tech_data.get('all_summaries', [])
    
    print(all_price_conclusions)
    print(all_tech_summaries)
    
    # Add all price analysis conclusions
    if all_price_conclusions:
        report += f"""

   • **Detailed Analysis (Price & EMAs)**"""
        for i, conclusion_data in enumerate(all_price_conclusions, 1):
            keyword = conclusion_data['keyword'].title()
            content = conclusion_data['content']
            if len(all_price_conclusions) > 1:
                report += f"""
      • **{keyword} {i}:** {content}"""
            else:
                report += f"""
      • **{keyword}:** {content}"""
    
    # Add all technical analysis summaries
    if all_tech_summaries:
        report += f"""

   • **Detailed Analysis (Technical Indicators)**"""
        for i, summary_data in enumerate(all_tech_summaries, 1):
            keyword = summary_data['keyword'].title()
            content = summary_data['content']
            if len(all_tech_summaries) > 1:
                report += f"""
      • **{keyword} {i}:** {content}"""
            else:
                report += f"""
      • **{keyword}:** {content}"""
    
    # Fallback: If no conclusions/summaries were extracted, add the old logic
    if not all_price_conclusions and not all_tech_summaries:
        price_conclusion = price_data.get('conclusion')
        tech_summary = tech_data.get('summary')
        
        if price_conclusion:
            report += f"""

        • **Detailed Analysis (Price & EMAs)**
            • {price_conclusion}"""
                
        if tech_summary:
                    report += f"""

        • **Detailed Analysis (Technical Indicators)**
            • {tech_summary}"""
      
    report += f"""
    
    **NOTE:**
    Take both the Detailed Analysis (Price & EMAs) & (Technical Indicators) - into account for your personal trading strategy.
    This report was generated using AI.
    This is not investment advice and hence all actions are the users responsibility. 
    This bot or the creators of it do not bear any responsibility on the results from the AI.
    """

    return report


async def split_and_send_message(update, text, parse_mode=None):
    """Split long messages into chunks and send them safely"""
    if len(text) <= MAX_MESSAGE_LENGTH:
        await update.message.reply_text(text, parse_mode=parse_mode)
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
            await update.message.reply_text(chunk, parse_mode=parse_mode)
        else:
            await update.message.reply_text(f"(Continued...)\n\n{chunk}", parse_mode=parse_mode)
        
        # Small delay between messages
        await asyncio.sleep(0.5)

# TELEGRAM BOT HANDLERS

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = """
🚀 **NSE Stock Technical Analysis Bot**

📈 **Features:**
• 3-month hourly chart analysis
• Comprehensive technical indicators
• Support/resistance level identification
• Trading recommendations with timeframes
• Relative strength vs benchmark analysis

💡 **How to use:**
• Send me any NSE stock symbol (e.g., RELIANCE, TCS, INFY)
• Get detailed technical analysis with clear entry/exit points
• Receive structured trading recommendations

🔥 **Examples:**
• RELIANCE
• TCS
• INFY
• HDFCBANK

📊 Just send the ticker symbol for comprehensive analysis!
    """
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

async def analyze_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle stock analysis requests with new reporting format"""
    symbol = update.message.text.strip().upper()
    
    # Basic validation
    if len(symbol) > 20 or not symbol.isalpha():
        await update.message.reply_text("❌ Please send a valid NSE stock symbol (e.g., RELIANCE, TCS)")
        return
    
    # Send initial message
    status_message = await update.message.reply_text(f"🔍 Analyzing {symbol}...\n⏳ Downloading 3-month charts...")
    
    try:
        # Download charts
        chart_data = fetch_charts(f"NSE:{symbol}", interval='1h')
        
        if not chart_data:
            await status_message.edit_text(f"❌ Could not download charts for {symbol}. Please check the symbol and try again.")
            return
        
        await status_message.edit_text(f"🔍 Analyzing {symbol}...\n🤖 AI processing charts...")
        
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
        
        await status_message.edit_text(f"🔍 Analyzing {symbol}...\n📊 Calculating relative strength...")
        
        # Get relative strength data
        relative_strength_data = None
        try:
            category, benchmark_symbol = get_stock_market_cap_category(symbol)
            if category and benchmark_symbol:
                rel_strength = calculate_relative_strength(symbol, benchmark_symbol)
                if rel_strength:
                    relative_strength_data = {
                        'category': category,
                        'benchmark_name': get_benchmark_name(benchmark_symbol),
                        **rel_strength
                    }
        except Exception as e:
            print(f"⚠️ Skipping relative strength analysis: {e}")
            relative_strength_data = None
        
        await status_message.edit_text(f"📈 Analysis complete for {symbol}!\n📤 Sending results...")
        
        # Send charts first
        for chart in chart_data:
            caption = "📈 3-Month Price & EMAs Chart" if chart['type'] == 'price_emas' else "📊 3-Month RSI & MACD Chart"
            await update.message.reply_photo(
                photo=io.BytesIO(chart['data']),
                caption=caption,
                filename=chart['filename']
            )
        
        # Generate and send new analysis report
        analysis_report = create_new_analysis_report(price_data, tech_data, symbol, relative_strength_data)
        
        # Delete status message
        await status_message.delete()
        
        # Send the analysis report
        await split_and_send_message(update, analysis_report, parse_mode='Markdown')
        
    except Exception as e:
        error_msg = format_error("Analysis", symbol, str(e))
        print(f"Error in analyze_stock: {e}")
        try:
            await status_message.edit_text(error_msg)
        except:
            await update.message.reply_text(error_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message."""
    help_text = """
🚀 **NSE Stock Technical Analysis Bot Help**

📊 **Features:**
• 3-month hourly chart analysis
• Support & resistance level identification
• Chart pattern recognition
• Trading recommendations with timeframes
• Relative strength analysis vs benchmarks

💡 **Commands:**
• /start - Start the bot
• /help - Show this help message
• Just send any NSE stock symbol for analysis

🔥 **Valid Symbols:**
• RELIANCE, TCS, INFY, HDFCBANK
• WIPRO, ICICIBANK, SBIN, LT
• BAJFINANCE, ASIANPAINT, etc.

📈 **Analysis Output:**
• Current price & EMA positions
• RSI and momentum indicators
• 3 support and 3 resistance levels
• Trade triggers and targets
• Entry strategies for different timeframes
• Risk management guidelines

⚠️ **Disclaimer:** This is for educational purposes only. Please do your own research before making investment decisions.
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all text messages as stock symbol requests."""
    await analyze_stock(update, context)

def main():
    """Start the bot."""
    if not validate_api_keys():
        return
    
    print("🚀 Starting NSE Technical Analysis Bot...")
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Run the bot until the user presses Ctrl-C
    print("✅ Bot is running... Press Ctrl+C to stop")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()