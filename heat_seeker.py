import yfinance as yf
import pandas as pd
import datetime
import csv
import json
import requests
from bs4 import BeautifulSoup
import os
from google import genai
import MetaTrader5 as mt5
import numpy as np
import schedule
import time
from datetime import datetime, timedelta
import pytz
import re

printed_next_repoll = False

# Custom JSON encoder to handle numpy data types and other non-serializable objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def is_market_open(symbol=None):
    """Check if financial markets are currently open for trading.
    Crypto markets are always open (24/7), forex follows traditional hours."""
    # Crypto markets are always open
    if symbol and symbol in ["BTC-USD", "ETH-USD", "LTC-USD", "BTCUSD", "ETHUSD", "LTCUSD"]:
        return True

    now = datetime.now(pytz.UTC)

    # Get current day of week (0=Monday, 6=Sunday)
    weekday = now.weekday()

    # Forex markets: Sunday 22:00 UTC to Friday 21:00 UTC
    if weekday == 6:  # Sunday
        market_open = now.time() >= datetime.strptime("22:00", "%H:%M").time()
    elif weekday >= 0 and weekday <= 4:  # Monday to Friday
        market_open = True
    elif weekday == 5:  # Saturday
        market_open = now.time() < datetime.strptime("21:00", "%H:%M").time()
    else:
        market_open = False

    if not market_open:
        print(f"Markets are closed (Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}, Day: {weekday})")
    else:
        print(f"Markets are open (Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')})")

    return market_open

# -------------------------
# CONFIGURATION
# -------------------------
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAH1ubvBUbA5s_n5B5EEQos3k12GU01m1E")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it as an environment variable or directly in the script.")

# MT5 Configuration
MT5_LOGIN = int(os.environ.get("MT5_LOGIN", "25247413"))  # Replace with your demo account number
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "6}a0/TwEmb8P")
MT5_SERVER = os.environ.get("MT5_SERVER", "Tickmill-Demo")

# Initialize the Gemini client
client = genai.Client(api_key=API_KEY)
# Use a suitable Gemini model
GEMINI_MODEL = 'gemini-2.5-flash' # A good, fast model for structured tasks

symbols = {
    "crypto": ["BTC-USD","ETH-USD","LTC-USD"]
}

# MT5 symbol mapping
symbols_mt5 = {
    "BTC-USD": "BTCUSD",
    "ETH-USD": "ETHUSD",
    "LTC-USD": "LTCUSD"
}

reverse_symbols_mt5 = {v: k for k, v in symbols_mt5.items()}

def mt5_to_yahoo(mt5_symbol):
    # Crypto: replace USD with -USD
    if mt5_symbol.endswith("USD") and mt5_symbol not in ["US30", "USTEC", "VIX"]:
        return mt5_symbol.replace("USD", "-USD")
    # Forex: add =X
    elif len(mt5_symbol) == 6 and mt5_symbol.isupper() and mt5_symbol not in ["US30", "USTEC", "VIX"]:
        return mt5_symbol + "=X"
    else:
        return None

csv_file = "high_heat_symbols_ai.csv"
trade_performance_file = "trade_performance.csv"
open_trades_file = "open_trades.csv"
close_counts_file = "close_counts.json"
last_check_file = "last_check.json"
heat_threshold = 7  # Only save symbols >= this Heat Score
top_n = 3  # Top N most volatile symbols per category

def load_open_trades():
    """Load open trades from CSV file"""
    open_trades = {}
    try:
        with open(open_trades_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 8:
                    ticket = row[0]
                    open_trades[ticket] = {
                        'symbol': row[1],
                        'rec': row[2],
                        'entry': row[3],
                        'tp': row[4],
                        'sl': row[5],
                        'open_time': row[6],
                        'rationale': row[7],
                        'position_id': row[8] if len(row) > 8 else ''
                    }
    except (FileNotFoundError, StopIteration):
        pass
    return open_trades

def save_open_trades(open_trades):
    """Save open trades to CSV file"""
    print(f"DEBUG: save_open_trades called with {len(open_trades)} trades")
    with open(open_trades_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Ticket", "Symbol", "Recommendation", "Entry Price", "Take Profit", "Stop Loss", "Open Time", "Rationale", "Position ID"])
        for ticket, trade in open_trades.items():
            writer.writerow([
                ticket, trade['symbol'], trade['rec'], trade['entry'],
                trade['tp'], trade['sl'], trade['open_time'], trade['rationale'], trade.get('position_id', '')
            ])
    print(f"DEBUG: save_open_trades completed, wrote {len(open_trades)} trades to {open_trades_file}")

# Multipliers for each symbol (user-configurable)
multipliers = {
    "BTCUSD": 5,
    "ETHUSD": 100,
    "LTCUSD": 500
}

# -------------------------
# FUNCTION: Fetch Recent Headlines
# -------------------------
def fetch_headlines(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = " ".join([h.text for h in soup.find_all("h3")])
        return headlines
    except:
        return ""

# -------------------------
# FUNCTION: Calculate Volatility (ATR / Close)
# -------------------------
def calculate_volatility(df):
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
    atr = df['TR'].rolling(7).mean().iloc[-1]
    volatility = atr / df['Close'].iloc[-1]  # Normalized ATR
    return volatility

# -------------------------
# FUNCTION: Ask AI for Intraday Buy/Sell + Levels + Heat Score
# -------------------------
def get_ai_recommendation(symbol, recent_intraday_data, recent_news_alerts, current_bid=None, current_ask=None):
    prompt = f"""
You are an elite intra-day market analyst and algorithmic trading assistant. Your expertise lies in identifying actionable short-term (1-2 hour) and medium-term (up to 6 hour) trading opportunities based on real-time market dynamics and breaking news.

Here is the current data for {symbol}:

-   **Recent Intraday Price Data (last 6-12 hours, ordered oldest to newest, e.g., 15-min or 1-hour intervals):** {recent_intraday_data}
    *   This is a JSON string of candlestick data with OHLCV (Open, High, Low, Close, Volume) for multiple timeframes (H1 and M15).
    *   Analyze this granular data for current trend direction, momentum strength, volatility levels, significant support/resistance zones, and potential chart patterns (e.g., breakouts, reversals, consolidations).
    *   Pay close attention to recent highs/lows and price action around key levels.
-   **Recent and Breaking Financial News/Alerts (last 1-6 hours):** {recent_news_alerts}
    *   Assess the immediate sentiment (highly positive, positive, neutral, negative, highly negative) and potential market impact of these news items. Identify any specific catalysts (e.g., economic data releases, corporate announcements, unexpected geopolitical events) that could rapidly influence price.
{f"-   **Current Market Prices:** Bid: {current_bid}, Ask: {current_ask}" if current_bid and current_ask else ""}

Based on an advanced synthesis of real-time technical analysis from the intraday data and immediate fundamental impact from news alerts:

-   **Prediction Horizon:** Focus on actionable moves within the next **1-2 hours (short-term)** and up to **6 hours (medium-term)**.
-   **Give a single, definitive Recommendation:** "Buy" or "Sell". If the market is extremely choppy with no clear edge, or if conflicting signals lead to high uncertainty for an actionable trade, prioritize "Neutral" with a very low Heat Score.
-   **Give a Heat Score from 0-10 (confidence in recommendation AND suggested levels):**
    *   **0-2 (Very Low Confidence):** Extremely mixed or weak signals, high market noise. Avoid trading.
    *   **3-5 (Low-Moderate Confidence):** Some directional bias, but with significant conflicting factors or moderate risk/uncertainty in levels.
    *   **6-8 (Moderate-High Confidence):** Clearer signals aligning across multiple indicators, good conviction, with reasonably reliable price levels.
    *   **9-10 (Very High Confidence):** Strong, unequivocal signals across technicals and fundamentals, high conviction, and high confidence in the accuracy of the suggested trading levels.

**Specific Rules for Heat Score Allocation & Trading Levels:**
-   **Crypto Assets (BTC, ETH, LTC):** Assign a Heat Score > 8 when multiple robust intraday technical indicators (e.g., strong trend continuation/reversal, significant breakout from a pattern, high momentum on volume) are unequivocally supported by immediate, impactful news sentiment. Entry/Exit/TP/SL should account for higher potential volatility.
-   **Entry Price:** Suggest the optimal price point to enter the trade. This should be based on identifying a favorable risk-reward entry (e.g., near support for a buy, near resistance for a sell, or after a confirmed breakout/retest).
-   **Exit Price:** Suggest a target price for closing the position if the market moves against the recommendation. This should function as a "Stop Loss" if the trade is moving unfavorably. For positive exits, it is implied by the Take Profit.
-   **Take Profit (TP):** Suggest a realistic price target where profit should be secured. This should be based on technical targets (e.g., next resistance/support, Fibonacci levels, projected pattern completion) and market structure.
-   **Stop Loss (SL):** Suggest a critical price level to exit the trade if it moves adversely, designed to limit potential losses. This should be placed at a logical technical point (e.g., below a key support for a long, above a key resistance for a short) considering current volatility.

**Return ONLY in this exact format:**
Symbol: {symbol}, Recommendation: <Buy/Sell/Neutral>, Heat Score: <0-10>, Entry Price: <numerical_value>, Exit Price: <numerical_value>, Take Profit: <numerical_value>, Stop Loss: <numerical_value>, Rationale: <A concise explanation (3-5 sentences) of the primary intraday technical and immediate fundamental factors driving your recommendation, confidence level, and the reasoning behind the suggested entry, exit, TP, and SL levels. State the expected time frame (1-2 hours or up to 6 hours).>
"""
    try:
        # -------------------------
        # NEW GEMINI API CALL
        # -------------------------
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=0.0 # equivalent to temperature=0
            )
        )

        text = response.text.strip()
        return text
    except Exception as e:
        print(f"AI error for {symbol}: {e}")
        return None

def connect_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        print("MT5 login failed")
        mt5.shutdown()
        return False
    # Removed print "MT5 connected successfully" to reduce noise
    return True

def open_trade(symbol, direction, lot=0.01, sl=None, tp=None):
    # Skip if markets are closed
    if not is_market_open(symbol):
        print(f"Skipping trade open for {symbol} - markets are closed")
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        return False
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Symbol {symbol} selection failed")
            return False
    # Get close count and multiplier
    multiplier = get_multiplier(symbol)
    min_volume = symbol_info.volume_min
    volume = multiplier * min_volume
    tick = mt5.symbol_info_tick(symbol)
    entry_price = tick.ask if direction == 0 else tick.bid
    print(f"Opening {symbol} with multiplier {multiplier}, volume: {volume}, SL: {sl}, TP: {tp}")
    
    # Validate stops
    if sl is not None and tp is not None:
        if direction == 0:  # Buy
            if sl >= entry_price or tp <= entry_price:
                print(f"Invalid stops for buy {symbol}: SL {sl} must be < entry {entry_price}, TP {tp} must be > entry {entry_price}")
                return False
        else:  # Sell
            if sl <= entry_price or tp >= entry_price:
                print(f"Invalid stops for sell {symbol}: SL {sl} must be > entry {entry_price}, TP {tp} must be < entry {entry_price}")
                return False
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if direction == 0 else mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Heat Seeker",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed for {symbol}: {result.comment}")
        return False
    print(f"Order placed for {symbol}: {'Buy' if direction == 0 else 'Sell'} (Volume: {volume})")
    # Get position_id from the created position
    positions = mt5.positions_get(ticket=result.order)
    if positions:
        position_id = positions[0].identifier  # position_id
    else:
        print(f"Warning: Could not get position_id for order {result.order}")
        position_id = 0
    return result.order, position_id

def modify_position_sl_tp(ticket, new_sl=None, new_tp=None):
    """Modify stop loss and take profit for an existing position"""
    if not connect_mt5():
        print("MT5 connection failed for position modification")
        return False

    # Get the position details
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        print(f"Position with ticket {ticket} not found")
        return False

    position = positions[0]
    current_sl = position.sl
    current_tp = position.tp

    # Determine the best SL and TP values
    final_sl = new_sl if new_sl is not None else current_sl
    final_tp = new_tp if new_tp is not None else current_tp

    # If both are None or unchanged, no modification needed
    if (final_sl == current_sl or final_sl is None) and (final_tp == current_tp or final_tp is None):
        print(f"No changes needed for position {ticket}")
        return True

    print(f"Modifying position {ticket} ({position.symbol}): SL {current_sl} -> {final_sl}, TP {current_tp} -> {final_tp}")

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": position.symbol,
        "sl": final_sl,
        "tp": final_tp,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Successfully modified position {ticket}: SL={final_sl}, TP={final_tp}")
        return True
    else:
        print(f"Failed to modify position {ticket}: {result.comment}")
        return False

def update_positions_from_recommendations():
    """Update TP/SL for open positions based on latest AI recommendations"""
    if not connect_mt5():
        print("MT5 connection failed for position updates")
        return

    # Get open positions
    positions = mt5.positions_get()
    if not positions:
        # Removed print "No open positions to update"
        return

    # Read latest recommendations from CSV
    recommendations = {}
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 9:  # Ensure we have all columns
                    symbol = row[2]  # Symbol column
                    try:
                        tp = float(row[7])  # Take Profit column
                        sl = float(row[8])  # Stop Loss column
                        recommendations[symbol] = {"tp": tp, "sl": sl}
                    except (ValueError, IndexError):
                        continue
    except (FileNotFoundError, StopIteration):
        print("No recommendations file found")
        return

    print(f"Found {len(recommendations)} recommendations and {len(positions)} open positions")

    # Update each position if it has a recommendation
    for position in positions:
        mt5_symbol = position.symbol
        yahoo_symbol = reverse_symbols_mt5.get(mt5_symbol, mt5_symbol)

        if yahoo_symbol in recommendations:
            rec = recommendations[yahoo_symbol]
            new_tp = rec["tp"]
            new_sl = rec["sl"]

            current_tp = position.tp
            current_sl = position.sl

            # Determine better values: lowest SL, highest TP
            final_sl = min(current_sl, new_sl) if current_sl and new_sl else (new_sl if new_sl else current_sl)
            final_tp = max(current_tp, new_tp) if current_tp and new_tp else (new_tp if new_tp else current_tp)

            # Only modify if there are actual changes
            if final_sl != current_sl or final_tp != current_tp:
                print(f"Updating {mt5_symbol} (ticket {position.ticket}): SL {current_sl} -> {final_sl}, TP {current_tp} -> {final_tp}")
                modify_position_sl_tp(position.ticket, final_sl, final_tp)
            else:
                print(f"No update needed for {mt5_symbol} (ticket {position.ticket})")
        else:
            print(f"No recommendation found for {mt5_symbol}")

def close_all_positions():
    """Close all positions if account equity exceeds balance by 100 units"""
    if not connect_mt5():
        print("MT5 connection failed for profit check")
        return

    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return

    floating_profit = account_info.equity - account_info.balance
    print(f"Floating profit (equity - balance): {floating_profit:.2f}, Target: 100.00")

    if floating_profit >= 100:
        print("Profit target reached (100+), closing all positions.")
        positions = mt5.positions_get()
        if not positions:
            print("No positions to close.")
            return

        for pos in positions:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 10,
                "magic": 234000,
                "comment": "Heat Seeker Close",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Closed position for {pos.symbol}")
            else:
                print(f"Failed to close {pos.symbol}: {result.comment}")
    else:
        print("Profit target not reached, positions remain open.")

def update_single_symbol(yahoo_symbol):
    try:
        mt5_symbol = symbols_mt5.get(yahoo_symbol, yahoo_symbol)
        # Fetch intraday data from MT5 for real-time accuracy
        if connect_mt5():
            print("MT5 connected successfully")
            timezone = pytz.timezone("Etc/UTC")  # Assuming UTC; adjust if needed
            
            # Select the symbol
            if not mt5.symbol_select(mt5_symbol, True):
                print(f"Failed to select symbol {mt5_symbol}")
                return
            print(f"Symbol {mt5_symbol} selected")
            
            # Fetch 1-Hour Data (last 7 days)
            h1_from = datetime.now(timezone) - timedelta(days=7)
            h1_rates = mt5.copy_rates_range(mt5_symbol, mt5.TIMEFRAME_H1, h1_from, datetime.now(timezone))
            print(f"H1 rates for {mt5_symbol}: {len(h1_rates) if h1_rates is not None else 'None'}")
            h1_data_list = []
            if h1_rates is not None and len(h1_rates) > 0:
                for i, rate in enumerate(h1_rates[-12:]):  # Take last 12 bars
                    try:
                        # Direct field access for structured arrays
                        print(f"Rate type: {type(rate)}, Fields: {rate.dtype.names if hasattr(rate, 'dtype') else 'No dtype'}")
                        
                        # MT5 returns a structured array with named fields
                        # Convert numpy types to standard Python types
                        time_val = int(rate[0])      # time field as integer
                        open_val = float(rate[1])    # open field as float
                        high_val = float(rate[2])    # high field as float
                        low_val = float(rate[3])     # low field as float
                        close_val = float(rate[4])   # close field as float
                        volume_val = int(rate[5])    # tick_volume field as integer
                        
                        h1_data_list.append({
                            "time": datetime.fromtimestamp(time_val, tz=timezone).strftime("%Y-%m-%d %H:%M"),
                            "timeframe": "H1",
                            "open": round(open_val, 5),
                            "high": round(high_val, 5),
                            "low": round(low_val, 5),
                            "close": round(close_val, 5),
                            "volume": int(volume_val)
                        })
                        print(f"Successfully processed H1 bar {i+1} for {mt5_symbol}")
                    except (AttributeError, TypeError, KeyError, ValueError) as e:
                        print(f"Invalid rate data for {mt5_symbol} H1: {e}, skipping")
                        continue
            
            # Fetch 15-Minute Data (last 1 day)
            m15_from = datetime.now(timezone) - timedelta(days=1)
            m15_rates = mt5.copy_rates_range(mt5_symbol, mt5.TIMEFRAME_M15, m15_from, datetime.now(timezone))
            print(f"M15 rates for {mt5_symbol}: {len(m15_rates) if m15_rates is not None else 'None'}")
            m15_data_list = []
            if m15_rates is not None and len(m15_rates) > 0:
                for i, rate in enumerate(m15_rates[-12:]):  # Take last 12 bars
                    try:
                        # Direct field access for structured arrays
                        print(f"M15 Rate type: {type(rate)}, Fields: {rate.dtype.names if hasattr(rate, 'dtype') else 'No dtype'}")
                        
                        # MT5 returns a structured array with named fields
                        # Convert numpy types to standard Python types
                        time_val = int(rate[0])      # time field as integer
                        open_val = float(rate[1])    # open field as float
                        high_val = float(rate[2])    # high field as float
                        low_val = float(rate[3])     # low field as float
                        close_val = float(rate[4])   # close field as float
                        volume_val = int(rate[5])    # tick_volume field as integer
                        
                        m15_data_list.append({
                            "time": datetime.fromtimestamp(time_val, tz=timezone).strftime("%Y-%m-%d %H:%M"),
                            "timeframe": "M15",
                            "open": round(open_val, 5),
                            "high": round(high_val, 5),
                            "low": round(low_val, 5),
                            "close": round(close_val, 5),
                            "volume": int(volume_val)
                        })
                        print(f"Successfully processed M15 bar {i+1} for {mt5_symbol}")
                    except (AttributeError, TypeError, KeyError, ValueError) as e:
                        print(f"Invalid rate data for {mt5_symbol} M15: {e}, skipping")
                        continue
            
            # Combine and sort by time
            combined_data = h1_data_list + m15_data_list
            if not combined_data:
                print(f"No valid MT5 data for {mt5_symbol}, skipping")
                return
            combined_data_sorted = sorted(combined_data, key=lambda x: datetime.strptime(x['time'], "%Y-%m-%d %H:%M"))
            
            try:
                # Use the custom encoder for json.dumps
                recent_intraday_data = json.dumps(combined_data_sorted, cls=NumpyEncoder)
                print(f"JSON serialization successful for {mt5_symbol}")
            except TypeError as e:
                print(f"JSON serialization error for {mt5_symbol}: {e}")
                print(f"Attempting fallback serialization...")
                # Fallback: manual conversion of problematic types
                for item in combined_data_sorted:
                    for key, value in item.items():
                        if isinstance(value, np.integer):
                            item[key] = int(value)
                        elif isinstance(value, np.floating):
                            item[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            item[key] = value.tolist()
                        elif isinstance(value, np.bool_):
                            item[key] = bool(value)
                recent_intraday_data = json.dumps(combined_data_sorted)
            
            mt5.shutdown()
        else:
            print(f"MT5 not connected for {mt5_symbol}, skipping")
            return
        
        recent_news_alerts = fetch_headlines(yahoo_symbol)
        result = get_ai_recommendation(yahoo_symbol, recent_intraday_data, recent_news_alerts)
        if result:
            print(f"AI Response for {yahoo_symbol}: {result[:200]}...")  # Debug: show first 200 chars
            # Parse the new format using regex
            try:
                # Extract fields using regex
                symbol_match = re.search(r'Symbol:\s*([^,]+)', result)
                rec_match = re.search(r'Recommendation:\s*([^,]+)', result)
                heat_match = re.search(r'Heat Score:\s*([^,]+)', result)
                entry_match = re.search(r'Entry Price:\s*([^,]+)', result)
                exit_match = re.search(r'Exit Price:\s*([^,]+)', result)
                tp_match = re.search(r'Take Profit:\s*([^,]+)', result)
                sl_match = re.search(r'Stop Loss:\s*([^,]+)', result)
                rationale_match = re.search(r'Rationale:\s*(.+)', result)
                
                if all([symbol_match, rec_match, heat_match, entry_match, exit_match, tp_match, sl_match, rationale_match]):
                    rec_symbol = symbol_match.group(1).strip()
                    recommendation = rec_match.group(1).strip()
                    ai_heat_score = float(heat_match.group(1).strip())
                    entry_price = float(entry_match.group(1).strip())
                    exit_price = float(exit_match.group(1).strip())
                    take_profit = float(tp_match.group(1).strip())
                    stop_loss = float(sl_match.group(1).strip())
                    rationale = rationale_match.group(1).strip()
                else:
                    raise ValueError("Missing required fields in AI response")
            except (AttributeError, ValueError) as e:
                print(f"Failed to parse AI response for {yahoo_symbol}: {e}")
                print(f"Response: {result}")
                return
            
            heat_score = ai_heat_score
            # All symbols are crypto now - no special heat score adjustments needed
            
            print(f"{rec_symbol} -> {recommendation} | Heat Score: {heat_score} | Entry: {entry_price} | TP: {take_profit} | SL: {stop_loss}")
            
            # Update CSV with unique entries per symbol, include new fields
            rows_dict = {}
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row in reader:
                        if len(row) >= 5:
                            symbol = row[2]  # Symbol is in column 2
                            rows_dict[symbol] = row
            except FileNotFoundError:
                pass
            if heat_score >= heat_threshold and recommendation != "Neutral" and entry_price > 0 and exit_price > 0 and take_profit > 0 and stop_loss > 0 and rationale.strip():
                today = datetime.now().strftime("%Y-%m-%d")
                time_stamp = datetime.now().strftime("%H:%M:%S")
                rows_dict[yahoo_symbol] = [today, time_stamp, yahoo_symbol, recommendation, str(heat_score), str(entry_price), str(exit_price), str(take_profit), str(stop_loss), rationale]
            else:
                if yahoo_symbol in rows_dict:
                    del rows_dict[yahoo_symbol]
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time", "Symbol", "Recommendation", "Heat Score", "Entry Price", "Exit Price", "Take Profit", "Stop Loss", "Rationale"])
                for symbol in sorted(rows_dict.keys()):
                    writer.writerow(rows_dict[symbol])
        else:
            print(f"No AI response for {yahoo_symbol}, skipping update.")
    except Exception as e:
        print(f"Error updating {yahoo_symbol}: {e}")

def close_position(pos):
    # Ensure symbol is selected
    if not mt5.symbol_select(pos.symbol, True):
        print(f"Failed to select symbol {pos.symbol} for closing")
        return

    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        print(f"Failed to get tick for {pos.symbol}")
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos.ticket,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "price": tick.bid if pos.type == 0 else tick.ask,
        "deviation": 10,
        "magic": 234000,
        "comment": "Heat Seeker Close Change",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Closed position for {pos.symbol} due to recommendation change")
        # Close count functionality removed
    else:
        print(f"Failed to close {pos.symbol}: {result.comment}")

def get_multiplier(symbol):
    return multipliers.get(symbol, 5)  # Default to 5 if not found

def test_get_ticket_details(ticket_number):
    """Test function to get details of a specific ticket from MT5 history"""
    if not connect_mt5():
        print("MT5 connection failed")
        return
    
    # Try to get order details
    orders = mt5.history_orders_get(ticket=ticket_number)
    if orders:
        order = orders[0]
        print(f"Order Details for Ticket {ticket_number}:")
        print(f"  Symbol: {order.symbol}")
        print(f"  Type: {'Buy' if order.type == mt5.ORDER_TYPE_BUY else 'Sell'}")
        print(f"  Volume: {order.volume_initial}")
        print(f"  Price: {order.price_open}")
        print(f"  SL: {order.sl}")
        print(f"  TP: {order.tp}")
        print(f"  Time: {datetime.fromtimestamp(order.time_setup)}")
        print(f"  State: {order.state}")
        print(f"  Comment: {order.comment}")
        print(f"  Position ID: {order.position_id}")
        
        # If order is filled, get deals for this position
        if order.state == mt5.ORDER_STATE_FILLED:
            deals = mt5.history_deals_get(position_id=order.position_id)
            if deals:
                total_profit = sum(deal.profit for deal in deals if deal.profit is not None)
                print(f"  Total Profit/Loss for Position: {total_profit}")
                for deal in deals:
                    print(f"  Deal: Type={'Buy' if deal.type == mt5.DEAL_TYPE_BUY else 'Sell'}, Volume={deal.volume}, Price={deal.price}, Profit={deal.profit}, Time={datetime.fromtimestamp(deal.time)}")
            else:
                pass  # No deals found
    else:
        print(f"No order found for ticket {ticket_number}")
    
    mt5.shutdown()

def log_closed_trades():
    """Log performance of closed trades"""
    if not connect_mt5():
        return
    
    # Load open trades
    open_trades = load_open_trades()
    
    if not open_trades:
        mt5.shutdown()
        return
    
    # Get deals from last 24 hours
    from_date = datetime.now() - timedelta(days=1)
    deals = mt5.history_deals_get(from_date, datetime.now())
    if deals is None:
        mt5.shutdown()
        return
    
    logged = set()
    try:
        with open(trade_performance_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Check if file has at least one line (header)
            try:
                next(reader)
            except StopIteration:
                # File is empty, no header
                pass
            for row in reader:
                if len(row) > 7:
                    logged.add(row[7])  # ticket
    except FileNotFoundError:
        pass
    
    for deal in deals:
        if deal.entry == mt5.DEAL_ENTRY_OUT and str(deal.position_id) in open_trades and str(deal.position_id) not in logged:
            trade = open_trades[str(deal.position_id)]
            # Get all deals for this position to sum profit
            pos_deals = mt5.history_deals_get(position_id=deal.position_id)
            total_profit = sum(d.profit for d in pos_deals if d.profit is not None)
            close_time = datetime.fromtimestamp(deal.time)
            # Ensure CSV exists with header
            try:
                with open(trade_performance_file, 'r') as f:
                    pass
            except FileNotFoundError:
                with open(trade_performance_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Date Opened", "Time Opened", "Symbol", "Recommendation", "Entry Price", "Take Profit", "Stop Loss", "Ticket", "Date Closed", "Time Closed", "Close Price", "Profit/Loss", "Rationale"])
            
            with open(trade_performance_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade['open_time'][:10], trade['open_time'][11:19], trade['symbol'], trade['rec'],
                    trade['entry'], trade['tp'], trade['sl'], str(deal.position_id),
                    close_time.strftime('%Y-%m-%d'), close_time.strftime('%H:%M:%S'),
                    deal.price, total_profit, trade['rationale']
                ])
            # Remove from open_trades
            del open_trades[str(deal.position_id)]
    
    save_open_trades(open_trades)
    
    mt5.shutdown()

def track_automatic_closures(current_positions):
    """Check for trades in open_trades.csv that are no longer open in MT5 and log them as closed"""
    print("DEBUG: track_automatic_closures called")
    # Load open trades from CSV
    open_trades = load_open_trades()
    print(f"DEBUG: loaded {len(open_trades)} trades from CSV")

    if not open_trades:
        print("DEBUG: no open trades in CSV")
        return

    # Get current open position ticket IDs
    current_tickets = set()
    if current_positions:
        current_tickets = {str(pos.ticket) for pos in current_positions}

    # Find trades that are in CSV but not in current positions (they were closed)
    closed_trades = []
    removed_old = []
    for ticket, trade in open_trades.items():
        if ticket not in current_tickets:
            position_id = trade.get('position_id', '')
            if position_id:
                closed_trades.append((ticket, position_id))
            else:
                # Old trade without position_id, just remove
                removed_old.append(ticket)
                del open_trades[ticket]

    if removed_old:
        print(f"DEBUG: Removed {len(removed_old)} old trades without position_id: {removed_old}")
        save_open_trades(open_trades)

    if not closed_trades:
        print("DEBUG: no closed trades detected")
        return

    print(f"DEBUG: Found {len(closed_trades)} automatically closed trades to log: {[t[0] for t in closed_trades]}")

    # For each closed trade, we need to find the closing deal to get the close details
    from_date = datetime.now() - timedelta(days=30)  # Look back 30 days for closing deals
    print(f"DEBUG: Getting deal history from {from_date} to {datetime.now()}")
    deals = mt5.history_deals_get(from_date, datetime.now())
    deals = sorted(deals, key=lambda d: d.time, reverse=True) if deals else []
    print(f"DEBUG: mt5.history_deals_get returned: {deals}")
    if deals is None:
        print("Failed to get deal history for closed trades")
        return

    print(f"DEBUG: Found {len(deals)} deals in history")
    if len(deals) == 0:
        print("DEBUG: No deals found in history, cannot log closed trades")
        # Still remove from open_trades since they're not open
        for ticket, _ in closed_trades:
            if ticket in open_trades:
                del open_trades[ticket]
        save_open_trades(open_trades)
        return

    # Group deals by position_id
    deals_by_position = {}
    for deal in deals:
        pos_id = str(deal.position_id)
        if pos_id not in deals_by_position:
            deals_by_position[pos_id] = []
        deals_by_position[pos_id].append(deal)

    # Check if performance file exists and get already logged tickets
    logged = set()
    try:
        with open(trade_performance_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Check if file has at least one line (header)
            try:
                next(reader)  # Skip header
            except StopIteration:
                # File is empty, no header
                pass
            for row in reader:
                if len(row) > 7:
                    logged.add(row[7])  # ticket column
    except FileNotFoundError:
        # Create file with header if it doesn't exist
        with open(trade_performance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Date Opened", "Time Opened", "Symbol", "Recommendation", "Entry Price", "Take Profit", "Stop Loss", "Ticket", "Date Closed", "Time Closed", "Close Price", "Profit/Loss", "Rationale"])

    # Process each closed trade
    for ticket, position_id in closed_trades:
        if ticket in logged:
            # Already logged, just remove from open_trades
            del open_trades[ticket]
            continue

        trade = open_trades[ticket]
        symbol = trade['symbol'].replace('-', '')  # Normalize symbol to match MT5 deal symbols
        open_time_str = trade['open_time']
        open_time = datetime.strptime(open_time_str, '%Y-%m-%dT%H:%M:%S.%f')
        closing_deals = [deal for deal in deals if deal.symbol == symbol and deal.entry == mt5.DEAL_ENTRY_OUT and deal.time > open_time.timestamp()]
        if not closing_deals:
            print(f"No closing deals found for ticket {ticket}")
            continue

        total_profit = sum(deal.profit for deal in closing_deals)
        close_time = datetime.fromtimestamp(closing_deals[0].time)
        close_price = closing_deals[0].price

        # Log to performance CSV
        with open(trade_performance_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                open_time_str[:10], open_time_str[11:19], trade['symbol'], trade['rec'],
                trade['entry'], trade['tp'], trade['sl'], ticket,
                close_time.strftime('%Y-%m-%d'), close_time.strftime('%H:%M:%S'),
                close_price, total_profit, trade['rationale']
            ])

        print(f"Logged closed trade: {trade['symbol']} ticket {ticket}, P/L: {total_profit}")

        # Remove from open_trades
        del open_trades[ticket]

    # Save updated open_trades
    save_open_trades(open_trades)

def scan_high_volatility_symbols():
    # Disabled - focusing only on the 3 crypto symbols
    return
    # Get all symbols
    mt5_symbols = mt5.symbols_get()
    if mt5_symbols is None:
        print("No symbols retrieved")
        mt5.shutdown()
        return
    # Filter by categories based on name patterns - only crypto now
    crypto_symbols = [s.name for s in mt5_symbols if any(crypto in s.name for crypto in ['BTC', 'ETH', 'LTC'])]
    all_symbols = crypto_symbols
    # Timeframe: 1 hour
    timeframe = mt5.TIMEFRAME_H1
    high_vol_symbols = []
    for symbol in all_symbols:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
        if rates is None or len(rates) < 20:
            continue
        df = pd.DataFrame(rates)
        # Calculate ATR for volatility
        df['H-L'] = df['high'] - df['low']
        df['H-C'] = abs(df['high'] - df['close'].shift(1))
        df['L-C'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
        atr = df['TR'].rolling(14).mean().iloc[-1]
        volatility = atr / df['close'].iloc[-1] if df['close'].iloc[-1] != 0 else 0
        # Momentum: RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['tick_volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + (positive_flow / negative_flow))).iloc[-1]
        high_vol_symbols.append({
            'symbol': symbol,
            'volatility': round(volatility, 4),
            'momentum': round(rsi, 2),
            'money_flow': round(mfi, 2)
        })
    # Sort by volatility descending, take top 10
    high_vol_symbols.sort(key=lambda x: x['volatility'], reverse=True)
    top_symbols = high_vol_symbols[:10]
    # Save to JSON
    data = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symbols': top_symbols
    }
    with open('high_volatility_symbols.json', 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    print(f"Scanned {len(all_symbols)} symbols, saved top 10 high-volatility symbols to high_volatility_symbols.json")
    # Check top symbol with AI, skipping already checked symbols
    original_symbols = set()
    for cat, syms in symbols.items():
        original_symbols.update(syms)
    for top in top_symbols:
        top_mt5 = top['symbol']
        yahoo = mt5_to_yahoo(top_mt5)
        if yahoo and yahoo not in original_symbols:
            print(f"Checking top volatility symbol {top_mt5} ({yahoo}) with AI")
            update_single_symbol(yahoo)
            break
        else:
            print(f"Skipping {top_mt5} ({yahoo or 'no mapping'}), already in main check list")
    mt5.shutdown()

def run_heat_seeker():
    global printed_next_repoll
    printed_next_repoll = False
    # Close counts functionality removed

    # Check if cached recommendations are recent (<6 hours)
    use_cache = True  # Temporarily use cache to test trade opening without polling
    cached_symbols = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                rows = list(reader)
                if rows:
                    latest_row = max(rows, key=lambda r: datetime.strptime(r[0] + ' ' + r[1], "%Y-%m-%d %H:%M:%S"))
                    latest_time = datetime.strptime(latest_row[0] + ' ' + latest_row[1], "%Y-%m-%d %H:%M:%S")
                    now_dt = datetime.now()
                    if False:  # Temporarily force AI poll
                        # Check if all cached data has complete price information
                        complete_data = True
                        for row in rows:
                            if len(row) < 9 or not row[7] or not row[8] or row[7] == "" or row[8] == "":
                                complete_data = False
                                break
                        if complete_data:
                            print("Cached recommendations are recent and complete (<6 hours), skipping AI poll and using existing data.")
                            use_cache = True
                            cached_symbols = [(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]
                        else:
                            print("Cached recommendations are recent but incomplete, polling AI to update.")
                    else:
                        print("Cached recommendations are old, polling AI.")
                else:
                    print("No cached data, polling AI.")
            except StopIteration:
                print("CSV file is empty, polling AI.")
    except (FileNotFoundError, ValueError):
        print("No valid cached data, polling AI.")

    # -------------------------
    # CREATE CSV (overwrite each run)
    # -------------------------
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Symbol", "Recommendation", "Heat Score", "Entry Price", "Exit Price", "Take Profit", "Stop Loss", "Rationale"])

    # -------------------------
    # PROCESS SYMBOLS
    # -------------------------
    today = datetime.now().strftime("%Y-%m-%d")
    time_stamp = datetime.now().strftime("%H:%M:%S")
    high_heat_symbols = []

    if use_cache:
        high_heat_symbols = cached_symbols
        print(f"Using {len(high_heat_symbols)} cached symbols.")
    else:
        # Process only crypto symbols (24/7 market)
        for category in ["crypto"]:  # Only crypto now
            if category in symbols:
                for symbol in symbols[category]:
                    try:
                        yahoo_symbol = symbol
                        mt5_symbol = symbols_mt5.get(yahoo_symbol, yahoo_symbol)
                        
                        # Check if market is open before processing
                        if not is_market_open(mt5_symbol):
                            print(f"Skipping {mt5_symbol} - market is closed")
                            continue
                            
                        # Fetch intraday data from MT5
                        if connect_mt5():
                            print("MT5 connected successfully")
                            timezone = pytz.timezone("Etc/UTC")
                            
                            # Select the symbol
                            if not mt5.symbol_select(mt5_symbol, True):
                                print(f"Failed to select symbol {mt5_symbol}")
                                continue
                            print(f"Symbol {mt5_symbol} selected")
                        
                        # Fetch 1-Hour Data (last 7 days)
                        h1_from = datetime.now(timezone) - timedelta(days=7)
                        h1_rates = mt5.copy_rates_range(mt5_symbol, mt5.TIMEFRAME_H1, h1_from, datetime.now(timezone))
                        print(f"H1 rates for {mt5_symbol}: {len(h1_rates) if h1_rates is not None else 'None'}")
                        h1_data_list = []
                        if h1_rates is not None and len(h1_rates) > 0:
                            for i, rate in enumerate(h1_rates[-12:]):  # Take last 12 bars
                                try:
                                    # Direct field access for structured arrays
                                    # MT5 returns a structured array with named fields
                                    time_val = rate[0]    # time field
                                    open_val = rate[1]    # open field
                                    high_val = rate[2]    # high field
                                    low_val = rate[3]     # low field
                                    close_val = rate[4]   # close field
                                    volume_val = rate[5]  # tick_volume field
                                    
                                    # Convert all values to native Python types to ensure JSON serialization works
                                    h1_data_list.append({
                                        "time": datetime.fromtimestamp(int(time_val), tz=timezone).strftime("%Y-%m-%d %H:%M"),
                                        "timeframe": "H1",
                                        "open": float(round(open_val, 5)),
                                        "high": float(round(high_val, 5)),
                                        "low": float(round(low_val, 5)),
                                        "close": float(round(close_val, 5)),
                                        "volume": int(volume_val)
                                    })
                                    print(f"Successfully processed H1 bar {i+1} for {mt5_symbol}")
                                except (AttributeError, TypeError, KeyError, ValueError) as e:
                                    print(f"Invalid rate data for {mt5_symbol} H1: {e}, skipping")
                                    continue
                        
                        # Fetch 15-Minute Data (last 1 day)
                        m15_from = datetime.now(timezone) - timedelta(days=1)
                        m15_rates = mt5.copy_rates_range(mt5_symbol, mt5.TIMEFRAME_M15, m15_from, datetime.now(timezone))
                        print(f"M15 rates for {mt5_symbol}: {len(m15_rates) if m15_rates is not None else 'None'}")
                        m15_data_list = []
                        if m15_rates is not None and len(m15_rates) > 0:
                            for i, rate in enumerate(m15_rates[-12:]):  # Take last 12 bars
                                try:
                                    # Direct field access for structured arrays
                                    # MT5 returns a structured array with named fields
                                    time_val = rate[0]    # time field
                                    open_val = rate[1]    # open field
                                    high_val = rate[2]    # high field
                                    low_val = rate[3]     # low field
                                    close_val = rate[4]   # close field
                                    volume_val = rate[5]  # tick_volume field
                                    
                                    # Convert all values to native Python types to ensure JSON serialization works
                                    m15_data_list.append({
                                        "time": datetime.fromtimestamp(int(time_val), tz=timezone).strftime("%Y-%m-%d %H:%M"),
                                        "timeframe": "M15",
                                        "open": float(round(open_val, 5)),
                                        "high": float(round(high_val, 5)),
                                        "low": float(round(low_val, 5)),
                                        "close": float(round(close_val, 5)),
                                        "volume": int(volume_val)
                                    })
                                    print(f"Successfully processed M15 bar {i+1} for {mt5_symbol}")
                                except (AttributeError, TypeError, KeyError, ValueError) as e:
                                    print(f"Invalid rate data for {mt5_symbol} M15: {e}, skipping")
                                    continue
                        
                        # Combine and sort by time
                        combined_data = h1_data_list + m15_data_list
                        if not combined_data:
                            print(f"No valid MT5 data for {mt5_symbol}, skipping")
                            continue
                        combined_data_sorted = sorted(combined_data, key=lambda x: datetime.strptime(x['time'], "%Y-%m-%d %H:%M"))
                        try:
                            # Use the custom encoder for json.dumps
                            recent_intraday_data = json.dumps(combined_data_sorted, cls=NumpyEncoder)
                            print(f"JSON serialization successful for {mt5_symbol}")
                        except Exception as e:
                            print(f"JSON serialization error for {mt5_symbol}: {e}")
                            print(f"Attempting fallback serialization...")
                            # Fallback: manual conversion of problematic types
                            for item in combined_data_sorted:
                                for key, value in item.items():
                                    # Convert any numpy values to Python native types
                                    if hasattr(value, 'dtype'):  # Check if it's a numpy type
                                        if np.issubdtype(value.dtype, np.integer):
                                            item[key] = int(value)
                                        elif np.issubdtype(value.dtype, np.floating):
                                            item[key] = float(value)
                                        else:
                                            item[key] = str(value)
                            recent_intraday_data = json.dumps(combined_data_sorted)
                        
                        # Get current market prices
                        tick = mt5.symbol_info_tick(mt5_symbol)
                        current_bid = tick.bid if tick else None
                        current_ask = tick.ask if tick else None
                        
                        mt5.shutdown()
                        recent_news_alerts = fetch_headlines(yahoo_symbol)
                        result = get_ai_recommendation(yahoo_symbol, recent_intraday_data, recent_news_alerts, current_bid, current_ask)
                        if result:
                            print(f"AI Response for {yahoo_symbol}: {result[:200]}...")  # Debug: show first 200 chars
                            # Parse AI output using regex for robustness
                            import re
                            try:
                                # Extract fields using regex
                                symbol_match = re.search(r'Symbol:\s*([^,]+)', result)
                                rec_match = re.search(r'Recommendation:\s*([^,]+)', result)
                                heat_match = re.search(r'Heat Score:\s*([^,]+)', result)
                                entry_match = re.search(r'Entry Price:\s*([^,]+)', result)
                                exit_match = re.search(r'Exit Price:\s*([^,]+)', result)
                                tp_match = re.search(r'Take Profit:\s*([^,]+)', result)
                                sl_match = re.search(r'Stop Loss:\s*([^,]+)', result)
                                rationale_match = re.search(r'Rationale:\s*(.+)', result)
                                
                                if all([symbol_match, rec_match, heat_match, entry_match, exit_match, tp_match, sl_match, rationale_match]):
                                    rec_symbol = symbol_match.group(1).strip()
                                    recommendation = rec_match.group(1).strip()
                                    ai_heat_score = float(heat_match.group(1).strip())
                                    entry_price = float(entry_match.group(1).strip())
                                    exit_price = float(exit_match.group(1).strip())
                                    take_profit = float(tp_match.group(1).strip())
                                    stop_loss = float(sl_match.group(1).strip())
                                    rationale = rationale_match.group(1).strip()
                                else:
                                    raise ValueError("Missing required fields in AI response")
                            except (AttributeError, ValueError) as e:
                                print(f"Failed to parse AI response for {yahoo_symbol}: {e}")
                                print(f"Response: {result}")
                                continue

                            # Boost Heat Score for forex & indices
                            heat_score = ai_heat_score  # Default to AI score
                            if rec_symbol in ["EURUSD=X","JPYUSD=X","ZARUSD=X","^IXIC","^DJI","^VIX"]:
                                heat_score = max(heat_score, ai_heat_score)  # Same as ai_heat_score

                            print(f"{rec_symbol} -> {recommendation} | Heat Score: {heat_score} | Entry: {entry_price} | Exit: {exit_price} | TP: {take_profit} | SL: {stop_loss}")

                            # Temporary: force Buy for BTC-USD to test saving
                            if yahoo_symbol == 'BTC-USD':
                                recommendation = 'Buy'
                                heat_score = 10
                                take_profit = entry_price + 1000  # Ensure TP > entry for buy
                                stop_loss = entry_price - 500    # Ensure SL < entry for buy
                                rationale = 'Forced Buy for testing'

                            # Save if Heat Score ≥ threshold and not Neutral, AND all price fields are valid
                            if (heat_score >= heat_threshold and recommendation != "Neutral" and
                                entry_price > 0 and exit_price > 0 and take_profit > 0 and stop_loss > 0 and rationale.strip()):
                                high_heat_symbols.append([today, time_stamp, rec_symbol, recommendation, heat_score, entry_price, exit_price, take_profit, stop_loss, rationale])
                            elif heat_score >= heat_threshold and recommendation != "Neutral":
                                print(f"Skipping {rec_symbol} - incomplete or invalid price data (Entry: {entry_price}, Exit: {exit_price}, TP: {take_profit}, SL: {stop_loss})")
                        else:
                            print(f"MT5 not connected for {mt5_symbol}, skipping")
                            continue

                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")

    # -------------------------
    # WRITE TO CSV
    # -------------------------
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in high_heat_symbols:
            writer.writerow(row)

    if not use_cache:
        print(f"\n{len(high_heat_symbols)} symbols added to {csv_file} today.")

    # Update last check times for all symbols
    last_check = {}
    try:
        with open(last_check_file, 'r') as f:
            last_check = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    now = datetime.now().timestamp()
    for category, syms in symbols.items():
        for symbol in syms:
            last_check[symbol] = now
    with open(last_check_file, 'w') as f:
        json.dump(last_check, f)

    # -------------------------
    # OPEN TRADES IN MT5
    # -------------------------
    if connect_mt5():
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                row_data = row + [""] * (10 - len(row))
                date, _, symbol, rec, heat, entry, exit_p, tp, sl, rationale = row_data
                mt5_symbol = symbols_mt5.get(symbol, symbol)
                direction = 0 if rec == "Buy" else 1
                positions = mt5.positions_get(symbol=mt5_symbol)
                if positions:
                    pos = positions[0]
                    current_direction = 0 if pos.type == mt5.POSITION_TYPE_BUY else 1
                    if current_direction != direction:
                        close_position(pos)
                    # Removed print for matching positions to reduce noise
                else:
                    # Only open trades if we have complete price data
                    if tp and sl and tp != "" and sl != "":
                        result = open_trade(mt5_symbol, direction, sl=float(sl) if sl else None, tp=float(tp) if tp else None)
                        if result:
                            ticket, position_id = result
                            print(f"DEBUG: Trade opened successfully, ticket: {ticket}, position_id: {position_id}")
                            # Record the trade
                            open_trades = load_open_trades()
                            open_trades[str(ticket)] = {
                                'symbol': symbol,
                                'rec': rec,
                                'entry': entry,
                                'tp': tp,
                                'sl': sl,
                                'rationale': rationale,
                                'open_time': datetime.now().isoformat(),
                                'position_id': str(position_id)
                            }
                            save_open_trades(open_trades)
                    else:
                        print(f"Skipping trade open for {mt5_symbol} - incomplete price data (TP: {tp}, SL: {sl})")
        close_all_positions()
        mt5.shutdown()
    else:
        print("MT5 connection failed, skipping trade opening")

# File to track previous positions for detecting automatic closures - DISABLED
# previous_positions_file = "previous_positions.json"



def check_profits():
    global printed_next_repoll
    if connect_mt5():
        positions = mt5.positions_get()
        if positions is None:
            print("Failed to get positions from MT5")
            mt5.shutdown()
            return

        # Track automatic closures (SL/TP hits)
        track_automatic_closures(positions)

        # Update TP/SL for open positions based on latest recommendations
        update_positions_from_recommendations()

        # Check for profit target closure (100 units)
        close_all_positions()

        if not positions:
            # Load last check times
            try:
                with open(last_check_file, 'r') as f:
                    last_check = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                last_check = {}
            if last_check:
                next_repoll = min(last_check.values()) + 30 * 60
            else:
                next_repoll = datetime.now().timestamp() + 30 * 60
            if not printed_next_repoll:
                print(f"Next repoll at {datetime.fromtimestamp(next_repoll).strftime('%Y-%m-%d %H:%M:%S')}")
                printed_next_repoll = True
            mt5.shutdown()
            return

        balance = mt5.account_info().balance
        # individual_threshold = 1  # Fixed take profit for single positions - DISABLED since we have AI TP/SL
        total_profit = sum(p.profit for p in positions)
        # total_threshold = 10  # Fixed take profit for all positions - DISABLED since we have AI TP/SL

        # 1. Check overall profit first - DISABLED since we have AI TP/SL
        # if total_profit >= total_threshold:
        #     print(f"Total profit threshold reached ({total_profit:.2f} >= {total_threshold:.2f}), closing all positions.")
        #     close_all_positions()
        #     # Refresh positions after closing all
        #     positions = mt5.positions_get()
        #     if not positions:
        #         mt5.shutdown()
        #         return

        # 2. Check recommendations
        rec_dict = {}
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) < 9:  # Skip incomplete rows
                        print(f"Skipping incomplete row: {row}")
                        continue
                    row_data = row + [""] * (10 - len(row))
                    yahoo_symbol = row_data[2]
                    rec = row_data[3]
                    heat = float(row_data[4]) if row_data[4] else 0
                    entry = row_data[5]
                    exit_p = row_data[6]
                    tp = row_data[7]
                    sl = row_data[8]
                    # Only include if we have valid TP and SL
                    if tp and sl and tp != "" and sl != "":
                        rec_dict[yahoo_symbol] = (rec, heat, entry, exit_p, tp, sl)
                    else:
                        print(f"Skipping {yahoo_symbol} - incomplete price data (TP: {tp}, SL: {sl})")
        except FileNotFoundError:
            pass  # CSV not yet created

        # Collect recent symbols (<6 hours) to keep positions open even if incomplete
        recent_symbols = set()
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                now = datetime.now()
                for row in reader:
                    if len(row) >= 3:
                        try:
                            row_time = datetime.strptime(row[0] + ' ' + row[1], "%Y-%m-%d %H:%M:%S")
                            if (now - row_time).total_seconds() < 6 * 3600:
                                recent_symbols.add(row[2])
                        except ValueError:
                            pass
        except FileNotFoundError:
            pass

        for pos in positions:
            yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
            if yahoo_symbol in rec_dict:
                rec, heat, entry, exit_p, tp, sl = rec_dict[yahoo_symbol]
                direction = 0 if rec == "Buy" else 1
                current_direction = 0 if pos.type == mt5.POSITION_TYPE_BUY else 1
                if current_direction != direction:
                    print(f"Recommendation changed for {pos.symbol} ({yahoo_symbol}), closing and reopening.")
                    close_position(pos)
                    yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
                    update_single_symbol(yahoo_symbol)
                    open_trade(pos.symbol, direction, sl=float(sl) if sl else None, tp=float(tp) if tp else None)

        # 3. Check individual take profits and losses - DISABLED since we have AI TP/SL
        # for pos in positions:
        #     multiplier = get_multiplier(pos.symbol)
        #     individual_threshold = 1 * multiplier
        #     yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
        #     if pos.profit >= individual_threshold:
        #         print(f"Individual take profit reached for {pos.symbol} (Profit: {pos.profit:.2f} >= {individual_threshold:.2f}), closing position.")
        #         close_position(pos)
        #     elif pos.profit <= -50:
        #         # Check if 15 minutes have passed since last recheck
        #         try:
        #             with open(close_counts_file, 'r') as f:
        #                 counts = json.load(f)
        #         except (FileNotFoundError, json.JSONDecodeError):
        #             counts = {}
        #         now = datetime.now().timestamp()
        #         last_recheck = counts.get(pos.symbol, {}).get("last_recheck", 0)
        #         if now - last_recheck > 30 * 60:  # 30 minutes
        #             print(f"Individual loss threshold reached for {pos.symbol} (Profit: {pos.profit:.2f} <= -50), updating recommendation.")
        #             update_single_symbol(yahoo_symbol)
        #             # Update last recheck time
        #             counts.setdefault(pos.symbol, {})["last_recheck"] = now
        #             with open(close_counts_file, 'w') as f:
        #                 json.dump(counts, f)
        #         else:
        #             print(f"Skipping recheck for {pos.symbol}, last recheck was {int((now - last_recheck) / 60)} minutes ago.")

        # Reopen positions for symbols still in CSV without open positions
        positions = mt5.positions_get()  # Refresh after closes
        if positions is None:
            print("Failed to get positions from MT5")
            return
        current_mt5_symbols = {pos.symbol for pos in positions}
        for yahoo_symbol, (rec, heat, entry, exit_p, tp, sl) in rec_dict.items():
            mt5_symbol = symbols_mt5.get(yahoo_symbol, yahoo_symbol)
            if mt5_symbol not in current_mt5_symbols:
                # Skip reopening if we don't have complete price data
                if not tp or not sl or tp == "" or sl == "":
                    print(f"Skipping reopening {mt5_symbol} - incomplete price data (TP: {tp}, SL: {sl})")
                    continue
                direction = 0 if rec == "Buy" else 1
                print(f"Reopening position for {mt5_symbol} as it's still in high heat list.")
                open_trade(mt5_symbol, direction, sl=float(sl) if sl else None, tp=float(tp) if tp else None)

        # Close positions for symbols not in high heat list
        for pos in positions:
            yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
            if yahoo_symbol not in rec_dict and yahoo_symbol not in recent_symbols:
                print(f"Closing position for {pos.symbol} ({yahoo_symbol}) as it's no longer in high heat list.")
                close_position(pos)

        # Load last check times
        try:
            with open(last_check_file, 'r') as f:
                last_check = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            last_check = {}

        now = datetime.now().timestamp()

        # Check and repoll symbols with open positions based on price bounds
        for pos in positions:
            yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
            if yahoo_symbol in rec_dict:
                # Get current price
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick:
                    current_price = tick.last if tick.last > 0 else (tick.bid + tick.ask) / 2
                    sl = float(rec_dict[yahoo_symbol][5])  # SL
                    tp = float(rec_dict[yahoo_symbol][4])  # TP
                    price_min = min(sl, tp)
                    price_max = max(sl, tp)
                    last_update = last_check.get(yahoo_symbol, 0)
                    if price_min <= current_price <= price_max:
                        # Within bounds, repoll if >6 hours since last update
                        if now - last_update > 6 * 3600:
                            print(f"Repolling symbol {yahoo_symbol} (within bounds, >6 hours)")
                            update_single_symbol(yahoo_symbol)
                            last_check[yahoo_symbol] = now
                    else:
                        # Outside bounds, remove from CSV and repoll immediately
                        print(f"Removing {yahoo_symbol} from high heat list (outside bounds) and repolling")
                        # Remove from CSV
                        rows = []
                        try:
                            with open(csv_file, 'r', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                header = next(reader)
                                for row in reader:
                                    if len(row) >= 3 and row[2] != yahoo_symbol:  # Symbol is column 2
                                        rows.append(row)
                            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerow(header)
                                writer.writerows(rows)
                        except FileNotFoundError:
                            pass
                        # Remove from rec_dict
                        if yahoo_symbol in rec_dict:
                            del rec_dict[yahoo_symbol]
                        # Repoll
                        update_single_symbol(yahoo_symbol)
                        last_check[yahoo_symbol] = now

        # Recheck non-high-heat symbols every 30 minutes
        for category, syms in symbols.items():
            for yahoo_symbol in syms:
                if yahoo_symbol not in rec_dict and yahoo_symbol not in recent_symbols and now - last_check.get(yahoo_symbol, 0) > 30 * 60:
                    print(f"Rechecking non-high-heat symbol {yahoo_symbol}")
                    update_single_symbol(yahoo_symbol)
                    last_check[yahoo_symbol] = now
        with open(last_check_file, 'w') as f:
            json.dump(last_check, f)

        mt5.shutdown()
        
        # Log closed trades
        log_closed_trades()

if __name__ == "__main__":
    schedule.every(5).seconds.do(check_profits)
    schedule.every(6).hours.do(run_heat_seeker)
    schedule.every(1).hours.do(scan_high_volatility_symbols)
    run_heat_seeker()  # Run immediately on start
    scan_high_volatility_symbols()  # Run volatility scan immediately on start
    test_get_ticket_details(127012277)  # Test getting ticket details
    track_automatic_closures([])  # Test logging closed trades
    while True:
        schedule.run_pending()
        time.sleep(1)
