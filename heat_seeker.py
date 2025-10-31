import yfinance as yf
import pandas as pd
import datetime
import csv
import json
import requests
from bs4 import BeautifulSoup
import os
from openai import OpenAI
import MetaTrader5 as mt5
import numpy as np
import schedule
import time
import threading
from datetime import datetime, timedelta
import pytz
import re
import uuid
import hashlib

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

# ------------------------- 
# AI CACHE CLASS FOR COST OPTIMIZATION
# -------------------------
class AICache:
    def __init__(self, cache_duration_minutes=30):
        self.cache_duration = cache_duration_minutes * 60  # Convert to seconds
        self.cache_file = "ai_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from file"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_cache_key(self, symbol, data_hash):
        """Generate cache key from symbol and data hash"""
        return f"{symbol}_{data_hash}"
    
    def get(self, symbol, data_hash):
        """Get cached response if valid"""
        cache_key = self._get_cache_key(symbol, data_hash)
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_duration:
                print(f"✅ Using cached AI response for {symbol}")
                return cached_item['response']
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
                self._save_cache()
        return None
    
    def set(self, symbol, data_hash, response):
        """Cache AI response"""
        cache_key = self._get_cache_key(symbol, data_hash)
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'response': response
        }
        self._save_cache()
        print(f"💾 Cached AI response for {symbol}")

# Global cache instance
ai_cache = AICache(cache_duration_minutes=5)  # Reduced from 30 to 5 minutes for local AI

# ------------------------- 
# REAL TIME MARKET WATCHER CLASS
# -------------------------
class RealTimeMarketWatcher:
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = ['BTC-USD', 'ETH-USD', 'LTC-USD']
        self.symbols = symbols
        self.last_analysis = {}  # Track last analysis time per symbol
        self.price_thresholds = {
            'BTC-USD': 0.005,  # 0.5% price change triggers analysis
            'ETH-USD': 0.008,  # 0.8% price change
            'LTC-USD': 0.012   # 1.2% price change
        }
        self.volume_threshold = 1.5  # 50% volume increase triggers analysis
        self.last_prices = {}
        self.last_volumes = {}
        self.monitoring_active = False

        # MT5 symbol mapping (same as heat_seeker.py)
        self.symbols_mt5 = {
            "BTC-USD": "BTCUSD",
            "ETH-USD": "ETHUSD",
            "LTC-USD": "LTCUSD"
        }

    def start_monitoring(self):
        """Start the real-time monitoring thread"""
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self.monitor_loop)
        monitor_thread.start()  # Removed daemon=True to keep program running
        print("🕐 Real-time market monitoring started...")

    def stop_monitoring(self):
        """Stop the monitoring"""
        self.monitoring_active = False
        print("🛑 Real-time market monitoring stopped")

    def monitor_loop(self):
        """Main monitoring loop that runs continuously"""
        last_profit_check = 0
        last_trade_check = 0
        while self.monitoring_active:
            try:
                for symbol in self.symbols:
                    self.check_symbol_events(symbol)
                
                # Check profits every 5 seconds
                now = time.time()
                if now - last_profit_check > 5:  # 5 seconds
                    print("🔄 Running periodic profit check...")
                    check_profits()
                    last_profit_check = now
                
                # Check for trades to open from CSV every 2 minutes
                if now - last_trade_check > 120:  # 2 minutes
                    print("📊 Checking for pending trades to open...")
                    self.check_and_open_trades_from_csv()
                    last_trade_check = now
                
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(30)  # Wait longer on errors

    def check_symbol_events(self, symbol):
        """Check if symbol has significant market events"""

        # Get current market data from MT5
        current_data = self.get_current_market_data(symbol)
        if not current_data:
            return

        current_price = current_data['price']
        current_volume = current_data['volume']

        # Initialize tracking if first time
        if symbol not in self.last_prices:
            self.last_prices[symbol] = current_price
            self.last_volumes[symbol] = current_volume
            return

        # Check for significant price movement
        price_change = abs(current_price - self.last_prices[symbol]) / self.last_prices[symbol]
        price_threshold = self.price_thresholds.get(symbol, 0.01)

        # Check for volume spike
        volume_change = current_volume / self.last_volumes[symbol] if self.last_volumes[symbol] > 0 else 1

        # Check time since last analysis
        last_analysis_time = self.last_analysis.get(symbol)
        time_since_analysis = datetime.now() - last_analysis_time if last_analysis_time else timedelta(hours=1)

        # Trigger conditions
        significant_price_move = price_change >= price_threshold
        volume_spike = volume_change >= self.volume_threshold
        time_overdue = time_since_analysis >= timedelta(minutes=20)  # Force analysis every 20 min minimum

        if significant_price_move or volume_spike or time_overdue:
            print(f"🎯 Market event detected for {symbol}:")
            if significant_price_move:
                print(".1%")
            if volume_spike:
                print(".1f")
            if time_overdue:
                print(f"   ⏰ Time since last analysis: {time_since_analysis}")

            # Trigger AI analysis
            self.trigger_ai_analysis(symbol, current_data)

            # Update tracking
            self.last_prices[symbol] = current_price
            self.last_volumes[symbol] = current_volume
            self.last_analysis[symbol] = datetime.now()

    def get_current_market_data(self, symbol):
        """Get current market data from MT5"""
        try:
            import MetaTrader5 as mt5

            # Make sure MT5 is initialized
            if not mt5.initialize():
                print("MT5 initialization failed")
                return None

            mt5_symbol = self.symbols_mt5.get(symbol, symbol)

            # Get current tick data
            tick = mt5.symbol_info_tick(mt5_symbol)
            if not tick:
                print(f"Failed to get tick data for {mt5_symbol}")
                return None

            # Get recent volume (from last candle)
            rates = mt5.copy_rates_from_pos(mt5_symbol, mt5.TIMEFRAME_M1, 0, 1)
            volume = rates[0]['real_volume'] if rates else 0

            return {
                'price': tick.last if tick.last > 0 else (tick.bid + tick.ask) / 2,
                'volume': volume,
                'bid': tick.bid,
                'ask': tick.ask,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            return None

    def trigger_ai_analysis(self, symbol, market_data):
        """Trigger AI analysis for the symbol"""
        print(f"🤖 Triggering AI analysis for {symbol}...")

        try:
            # Run full analysis and update CSV
            update_single_symbol(symbol)
            print(f"✅ AI analysis completed for {symbol}")

            # Now check if we should open trades from updated recommendations
            self.check_and_open_trades_from_csv()

        except Exception as e:
            print(f"AI analysis error for {symbol}: {e}")

    def check_and_open_trades_from_csv(self):
        """Check CSV recommendations and open trades if needed"""
        try:
            import MetaTrader5 as mt5

            # Make sure MT5 is connected
            if not mt5.initialize():
                print("MT5 initialization failed for trade opening")
                return

            mt5_login = int(os.environ.get("MT5_LOGIN", "25247413"))
            mt5_password = os.environ.get("MT5_PASSWORD", "6}a0/TwEmb8P")
            mt5_server = os.environ.get("MT5_SERVER", "Tickmill-Demo")

            if not mt5.login(mt5_login, mt5_password, mt5_server):
                print("MT5 login failed for trade opening")
                mt5.shutdown()
                return

            csv_file = "high_heat_symbols_ai.csv"

            # Read recommendations from CSV
            recommendations = []
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip header
                    for row in reader:
                        row_data = row + [""] * (11 - len(row))
                        date, _, symbol, rec, heat, entry, exit_p, tp, sl, rationale, recommendation_id = row_data
                        if tp and sl and tp != "" and sl != "":
                            recommendations.append({
                                'symbol': symbol,
                                'rec': rec,
                                'tp': tp,
                                'sl': sl,
                                'entry': entry,
                                'rationale': rationale,
                                'recommendation_id': recommendation_id,
                                'heat_score': heat
                            })
            except (FileNotFoundError, IndexError):
                pass

            if not recommendations:
                mt5.shutdown()
                return

            # Check each recommendation and open trade if needed
            for rec in recommendations:
                yahoo_symbol = rec['symbol']
                mt5_symbol = self.symbols_mt5.get(yahoo_symbol, yahoo_symbol)
                direction = 0 if rec['rec'].upper() == "BUY" else 1

                # Check if position already exists
                positions = mt5.positions_get(symbol=mt5_symbol)
                if positions:
                    # Position exists, skip
                    continue

                # Open the trade
                print(f"📈 Opening {rec['rec']} position for {mt5_symbol} from CSV recommendation...")
                result = self.open_trade_from_csv(mt5_symbol, direction, rec)
                if result:
                    ticket, position_id = result
                    print(f"✅ Trade opened successfully! Ticket: {ticket}")

            mt5.shutdown()

        except Exception as e:
            print(f"Error in check_and_open_trades_from_csv: {e}")
            try:
                mt5.shutdown()
            except:
                pass

    def open_trade_from_csv(self, symbol, direction, recommendation):
        """Open a trade based on CSV recommendation data"""
        try:
            import MetaTrader5 as mt5

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

            # Multipliers for each symbol
            multipliers = {
                "BTCUSD": 5,
                "ETHUSD": 100,
                "LTCUSD": 500
            }
            multiplier = multipliers.get(symbol, 5)
            min_volume = symbol_info.volume_min
            volume = multiplier * min_volume

            tick = mt5.symbol_info_tick(symbol)
            entry_price = tick.ask if direction == 0 else tick.bid

            # Recalculate SL/TP based on current entry price using ratios from stored values
            stored_entry = float(recommendation['entry'])
            stored_sl = float(recommendation['sl'])
            stored_tp = float(recommendation['tp'])
            
            if stored_entry > 0:
                sl_ratio = stored_sl / stored_entry
                tp_ratio = stored_tp / stored_entry
                sl = entry_price * sl_ratio
                tp = entry_price * tp_ratio
            else:
                sl = stored_sl
                tp = stored_tp

            print(f"Opening {symbol} with multiplier {multiplier}, volume: {volume}, SL: {sl}, TP: {tp}")

            # Validate stops
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
                "comment": "Heat Seeker Auto",
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC,
                "sl": sl,
                "tp": tp
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Order failed for {symbol}: {result.comment}")
                return False

            print(f"Order placed for {symbol}: {'Buy' if direction == 0 else 'Sell'} (Volume: {volume})")

            # Get position_id from the created position
            positions = mt5.positions_get(ticket=result.order)
            if positions:
                position_id = positions[0].identifier
            else:
                position_id = 0

            # Record the trade with updated SL/TP
            open_trades = load_open_trades()
            open_trades[str(result.order)] = {
                'symbol': recommendation['symbol'],
                'rec': recommendation['rec'],
                'entry': str(entry_price),  # Use actual entry price
                'tp': str(tp),
                'sl': str(sl),
                'rationale': recommendation['rationale'],
                'open_time': datetime.now().isoformat(),
                'position_id': str(position_id),
                'recommendation_id': recommendation['recommendation_id'],
                'heat_score': recommendation['heat_score']
            }
            save_open_trades(open_trades)

            return result.order, position_id

        except Exception as e:
            print(f"Error opening trade for {symbol}: {e}")
            return False

    def save_recommendation_to_csv(self, symbol, recommendation, market_data):
        """Save AI recommendation to high_heat_symbols_ai.csv"""

        # Create recommendation row
        row = {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Time': datetime.now().strftime('%H:%M:%S'),
            'Symbol': symbol,
            'Recommendation': recommendation['recommendation'],
            'Heat Score': recommendation['confidence'],
            'Entry Price': recommendation['entry'],
            'Exit Price': recommendation['sl'],  # Stop loss as exit
            'Take Profit': recommendation['tp'],
            'Stop Loss': recommendation['sl'],
            'Rationale': recommendation['rationale'],
            'Recommendation_ID': str(uuid.uuid4())
        }

        # Append to CSV (create if doesn't exist)
        csv_file = 'high_heat_symbols_ai.csv'
        file_exists = os.path.exists(csv_file)

        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def evaluate_trade_opportunity(self, symbol, recommendation):
        """Evaluate if we should open a trade based on recommendation"""

        try:
            import MetaTrader5 as mt5

            # Check if we already have a position
            positions = mt5.positions_get()
            if positions is None:
                print("Failed to get positions from MT5")
                return

            symbol_positions = [p for p in positions if p.symbol == self.symbols_mt5.get(symbol, symbol)]

            if symbol_positions:
                print(f"⚠️ Already have position(s) for {symbol}, skipping trade evaluation")
                return

            # Check confidence threshold
            if recommendation['confidence'] >= 7.0:  # High confidence only
                print(f"🎯 High-confidence recommendation for {symbol}, evaluating trade...")

                # Check account risk limits
                if self.check_risk_limits(symbol, recommendation):
                    # Open the trade
                    self.open_recommended_trade(symbol, recommendation)

        except Exception as e:
            print(f"Error evaluating trade opportunity for {symbol}: {e}")

    def check_risk_limits(self, symbol, recommendation):
        """Check if trade fits within risk limits"""
        try:
            import MetaTrader5 as mt5

            # Get account info
            account = mt5.account_info()
            if not account:
                print("Failed to get account info")
                return False

            # Basic risk checks
            margin_used_pct = (account.margin / account.balance) * 100 if account.balance > 0 else 0

            # Don't open new trades if margin usage > 70%
            if margin_used_pct > 70:
                print(f"⚠️ Margin usage too high ({margin_used_pct:.1f}%), skipping trade")
                return False

            # Check if single position would be > 20% of account
            # This is a simplified check - you might want more sophisticated risk management
            estimated_position_size = account.balance * 0.02  # 2% of account
            if estimated_position_size > account.balance * 0.20:
                print(f"⚠️ Position size too large for {symbol}")
                return False

            return True

        except Exception as e:
            print(f"Error checking risk limits: {e}")
            return False

    def start(self):
        """Alias for start_monitoring()"""
        self.start_monitoring()

    def stop(self):
        """Alias for stop_monitoring()"""
        self.stop_monitoring()

# ------------------------- 
# OPTIMIZED PROMPT FUNCTION
# -------------------------
def create_optimized_prompt(symbol, recent_intraday_data, recent_news_alerts, current_bid=None, current_ask=None):
    """Create ultra-optimized prompt to reduce token usage by 85%"""
    
    # Extract key market data points
    try:
        data = json.loads(recent_intraday_data)
        if not data:
            raise ValueError("No data available")
        
        # Get current price
        current_price = (current_bid + current_ask) / 2 if current_bid and current_ask else data[-1]['close']
        
        # Calculate trend direction and strength
        closes = [item['close'] for item in data[-6:]]  # Last 6 periods
        if len(closes) >= 2:
            trend_direction = "up" if closes[-1] > closes[0] else "down"
            trend_strength = "strong" if abs(closes[-1] - closes[0]) / closes[0] > 0.01 else "weak"
        else:
            trend_direction = "sideways"
            trend_strength = "weak"
        
        # Find support and resistance levels
        highs = [item['high'] for item in data[-12:]]
        lows = [item['low'] for item in data[-12:]]
        support = min(lows) if lows else current_price * 0.98
        resistance = max(highs) if highs else current_price * 1.02
        
        # Volume trend
        volumes = [item['volume'] for item in data[-6:]]
        volume_trend = "increasing" if volumes and len(volumes) >= 2 and volumes[-1] > volumes[0] else "decreasing"
        
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        # Fallback values
        current_price = 50000
        trend_direction = "unknown"
        trend_strength = "unknown"
        support = current_price * 0.95
        resistance = current_price * 1.05
        volume_trend = "unknown"
    
    # Create ultra-optimized prompt
    prompt = f"""Analyze {symbol} for trading opportunity:

PRICE: ${current_price:.2f}
TREND: {trend_direction} ({trend_strength})
SUPPORT: ${support:.2f}
RESISTANCE: ${resistance:.2f}
VOLUME: {volume_trend}

Return ONLY the following JSON object, no additional text or explanation:
{{"recommendation":"BUY/SELL/HOLD", "confidence":1-10, "entry":price, "tp":price, "sl":price, "rationale":"brief explanation"}}"""
    
    return prompt
    
    return prompt

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
# LM Studio Configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
API_KEY = "lm-studio"  # Dummy API key required by OpenAI library but ignored by LM Studio

# MT5 Configuration
MT5_LOGIN = int(os.environ.get("MT5_LOGIN", "25247413"))  # Replace with your demo account number
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "6}a0/TwEmb8P")
MT5_SERVER = os.environ.get("MT5_SERVER", "Tickmill-Demo")

# Initialize the OpenAI client, pointing it to your local LM Studio server
client = OpenAI(
    base_url=LM_STUDIO_URL,
    api_key=API_KEY
)

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
account_management_actions_file = "account_management_actions.csv"
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
                        'position_id': row[8] if len(row) > 8 else '',
                        'recommendation_id': row[9] if len(row) > 9 else '',
                        'heat_score': row[10] if len(row) > 10 else ''
                    }
    except (FileNotFoundError, StopIteration):
        pass
    return open_trades

def save_open_trades(open_trades):
    """Save open trades to CSV file"""
    print(f"DEBUG: save_open_trades called with {len(open_trades)} trades")
    with open(open_trades_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Ticket", "Symbol", "Recommendation", "Entry Price", "Take Profit", "Stop Loss", "Open Time", "Rationale", "Position ID", "Recommendation_ID", "Heat_Score"])
        for ticket, trade in open_trades.items():
            writer.writerow([
                ticket, trade['symbol'], trade['rec'], trade['entry'],
                trade['tp'], trade['sl'], trade['open_time'], trade['rationale'], 
                trade.get('position_id', ''), trade.get('recommendation_id', ''), 
                trade.get('heat_score', '')
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
    # Create optimized prompt
    prompt = create_optimized_prompt(symbol, recent_intraday_data, recent_news_alerts, current_bid, current_ask)
    
    # Create hash of input data for caching
    data_hash = hashlib.md5(f"{symbol}_{recent_intraday_data}_{recent_news_alerts}_{current_bid}_{current_ask}".encode()).hexdigest()
    
    # TEMPORARILY DISABLE CACHE FOR FRESH AI RECOMMENDATIONS
    # Check cache first
    # cached_response = ai_cache.get(symbol, data_hash)
    # if cached_response:
    #     return cached_response
    
    try:
        print(f"🤖 Sending optimized AI prompt for {symbol}...")
        # ------------------------- 
        # LM STUDIO API CALL
        # -------------------------
        # Combine system prompt with user prompt since Mistral only supports user/assistant roles
        full_prompt = f"""You are an elite intra-day market analyst and algorithmic trading assistant. Your expertise lies in identifying actionable short-term (1-2 hour) and medium-term (up to 6 hour) trading opportunities based on real-time market dynamics and breaking news.

{prompt}"""
        
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0
        )

        text = completion.choices[0].message.content.strip()
        print(f"✅ AI Response received for {symbol}: {text[:300]}...")
        
        # Cache the response
        ai_cache.set(symbol, data_hash, text)
        
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
    min_volume = float(symbol_info.volume_min)  # Ensure it's a float
    volume = float(multiplier) * min_volume  # Ensure volume is a float
    tick = mt5.symbol_info_tick(symbol)
    entry_price = tick.ask if direction == 0 else tick.bid
    print(f"Opening {symbol} with multiplier {multiplier}, volume: {volume}, SL: {sl}, TP: {tp}")
    
    # Validate stops
    if sl is not None and tp is not None:
        sl = float(sl)  # Ensure sl is float
        tp = float(tp)  # Ensure tp is float
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
        request["sl"] = float(sl)
    if tp is not None:
        request["tp"] = float(tp)
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

def modify_position_sl_tp(position_id, new_sl=None, new_tp=None):
    """Modify stop loss and take profit for an existing position"""
    if not connect_mt5():
        print("MT5 connection failed for position modification")
        return False

    # Get all positions
    positions = mt5.positions_get()
    if not positions:
        print(f"No positions found")
        return False

    # Ensure position_id is a valid identifier
    try:
        if isinstance(position_id, str):
            position_id_int = int(position_id)
        elif isinstance(position_id, int):
            position_id_int = position_id
        else:
            print(f"Invalid position_id type: {type(position_id)}, value: {position_id}")
            return False
    except (ValueError, TypeError) as e:
        print(f"Error converting position_id to int: {e}, position_id: {position_id}")
        return False

    # Find position by position_id
    pos = next((p for p in positions if p.identifier == position_id_int), None)
    if not pos:
        print(f"Position with position_id {position_id_int} not found")
        return False

    current_sl = pos.sl
    current_tp = pos.tp

    # Determine the best SL and TP values
    final_sl = new_sl if new_sl is not None else current_sl
    final_tp = new_tp if new_tp is not None else current_tp

    # If both are None or unchanged, no modification needed
    if (final_sl == current_sl or final_sl is None) and (final_tp == current_tp or final_tp is None):
        print(f"No changes needed for position {position_id}")
        return True

    print(f"Modifying position {position_id} ({pos.symbol}): SL {current_sl} -> {final_sl}, TP {current_tp} -> {final_tp}")

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "symbol": pos.symbol,
        "sl": final_sl,
        "tp": final_tp,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Successfully modified position {position_id}: SL={final_sl}, TP={final_tp}")
        return True
    else:
        print(f"Failed to modify position {position_id}: {result.comment}")
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
                modify_position_sl_tp(position.identifier, final_sl, final_tp)
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
            # Parse the new JSON format
            ai_data = {}
            try:
                # Extract JSON from response (it might be embedded in text)
                import re
                json_match = re.search(r'\{.*\}', result.strip())
                if json_match:
                    json_str = json_match.group(0)
                    ai_data = json.loads(json_str)
                else:
                    # Try parsing the whole response as JSON
                    ai_data = json.loads(result.strip())
            except (json.JSONDecodeError, ValueError):
                # Fallback: extract individual fields from text format
                rec_match = re.search(r'(?:recommend|Recommendation).*?["\']?([^"\',\s\n]+)', result, re.IGNORECASE)
                conf_match = re.search(r'confidence level \((\d+)\)', result, re.IGNORECASE)
                entry_match = re.search(r'support level at \$([0-9.]+)', result, re.IGNORECASE)
                # More flexible TP/SL matching
                tp_match = re.search(r'(?:Take Profit|target price|tp).*?[\$]?(\d+\.\d+)', result, re.IGNORECASE)
                sl_match = re.search(r'(?:Stop Loss|stop loss|sl).*?[\$]?(\d+\.\d+)', result, re.IGNORECASE)
                rationale_match = re.search(r'(?:Rationale|Reasoning):\s*(.+)', result, re.IGNORECASE | re.DOTALL)
                
                if rec_match:
                    ai_data['recommendation'] = rec_match.group(1).strip().upper()
                if conf_match:
                    ai_data['confidence'] = int(conf_match.group(1))
                if entry_match:
                    ai_data['entry'] = float(entry_match.group(1))
                if tp_match:
                    ai_data['tp'] = float(tp_match.group(1))
                if sl_match:
                    ai_data['sl'] = float(sl_match.group(1))
                if rationale_match:
                    ai_data['rationale'] = rationale_match.group(1).strip()
            
            # Map JSON fields to our variables
            recommendation = ai_data.get('recommendation', 'HOLD').upper()
            # Convert HOLD to Neutral for consistency
            if recommendation == 'HOLD':
                recommendation = 'Neutral'
            elif recommendation not in ['BUY', 'SELL', 'NEUTRAL']:
                recommendation = 'Neutral'
            
            heat_score = float(ai_data.get('confidence', 3))
            entry_price = float(ai_data.get('entry', 0)) if ai_data.get('entry') else 0
            take_profit = float(ai_data.get('tp', 0)) if ai_data.get('tp') else 0
            stop_loss = float(ai_data.get('sl', 0)) if ai_data.get('sl') else 0
            rationale = ai_data.get('rationale', 'No rationale provided')
            
            # Generate default TP/SL if not provided but we have entry price
            if entry_price > 0 and take_profit == 0 and stop_loss == 0:
                if recommendation == 'BUY':
                    take_profit = entry_price * 1.02  # 2% target
                    stop_loss = entry_price * 0.98    # 2% stop
                elif recommendation == 'SELL':
                    take_profit = entry_price * 0.98  # 2% target
                    stop_loss = entry_price * 1.02    # 2% stop
            
            # For neutral recommendations, set prices to 0
            if recommendation == 'NEUTRAL':
                entry_price = take_profit = stop_loss = 0
            
            # Set exit price same as entry for now (can be refined later)
            exit_price = entry_price
            
            heat_score = heat_score
            # All symbols are crypto now - no special heat score adjustments needed
            
            print(f"{yahoo_symbol} -> {recommendation} | Heat Score: {heat_score} | Entry: {entry_price} | TP: {take_profit} | SL: {stop_loss}")
            
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
                # Generate unique Recommendation_ID
                recommendation_id = str(uuid.uuid4())
                # Keep existing recommendation_id if symbol is already in dict
                if yahoo_symbol in rows_dict and len(rows_dict[yahoo_symbol]) > 10:
                    recommendation_id = rows_dict[yahoo_symbol][10]
                rows_dict[yahoo_symbol] = [today, time_stamp, yahoo_symbol, recommendation, str(heat_score), str(entry_price), str(exit_price), str(take_profit), str(stop_loss), rationale, recommendation_id]
            else:
                if yahoo_symbol in rows_dict:
                    del rows_dict[yahoo_symbol]
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time", "Symbol", "Recommendation", "Heat Score", "Entry Price", "Exit Price", "Take Profit", "Stop Loss", "Rationale", "Recommendation_ID"])
                for symbol in sorted(rows_dict.keys()):
                    writer.writerow(rows_dict[symbol])
        else:
            print(f"No AI response for {yahoo_symbol}, skipping update.")
    except Exception as e:
        print(f"Error updating {yahoo_symbol}: {e}")

def close_position(position_id):
    """Close a position by position identifier"""
    if not connect_mt5():
        print("MT5 connection failed for position closing")
        return

    # Get all positions
    positions = mt5.positions_get()
    if not positions:
        print("No positions found")
        return

    # Ensure position_id is a valid identifier
    try:
        if isinstance(position_id, str):
            position_id_int = int(position_id)
        elif isinstance(position_id, int):
            position_id_int = position_id
        else:
            print(f"Invalid position_id type: {type(position_id)}, value: {position_id}")
            return
    except (ValueError, TypeError) as e:
        print(f"Error converting position_id to int: {e}, position_id: {position_id}")
        return

    pos = next((p for p in positions if p.identifier == position_id_int), None)
    if not pos:
        print(f"Position with position_id {position_id_int} not found for closing")
        return

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
    """Get volume multiplier for symbol (default 1)"""
    return 1

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
                    writer.writerow(["Recommendation_ID", "Date Opened", "Time Opened", "Symbol", "Recommendation", "Heat_Score", "Entry Price", "Take Profit", "Stop Loss", "Ticket", "Date Closed", "Time Closed", "Close Price", "Profit/Loss", "Original_Rationale", "Duration_Hours"])
            
            # Parse open_time and calculate duration
            try:
                open_time = datetime.fromisoformat(trade['open_time'])
            except (ValueError, KeyError):
                # Fallback for old format
                open_time = datetime.strptime(trade['open_time'][:19], '%Y-%m-%d %H:%M:%S') if 'T' not in trade['open_time'] else datetime.fromisoformat(trade['open_time'])
            
            duration_hours = round((close_time - open_time).total_seconds() / 3600, 2)
            
            with open(trade_performance_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.get('recommendation_id', ''),
                    open_time.strftime('%Y-%m-%d'), 
                    open_time.strftime('%H:%M:%S'), 
                    trade['symbol'], 
                    trade['rec'],
                    trade.get('heat_score', ''),
                    trade['entry'], 
                    trade['tp'], 
                    trade['sl'], 
                    str(deal.position_id),
                    close_time.strftime('%Y-%m-%d'), 
                    close_time.strftime('%H:%M:%S'),
                    deal.price, 
                    total_profit, 
                    trade['rationale'],
                    duration_hours
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
    for ticket, trade in list(open_trades.items()):
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
    if deals is None:
        print("Failed to get deal history for closed trades")
        # Still remove from open_trades since they're not open
        for ticket, _ in closed_trades:
            if ticket in open_trades:
                del open_trades[ticket]
        save_open_trades(open_trades)
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

    # Group deals by position_id for easier lookup
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
                if len(row) > 9:
                    logged.add(row[9])  # ticket column (now at index 9)
    except FileNotFoundError:
        # Create file with header if it doesn't exist
        with open(trade_performance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Recommendation_ID", "Date Opened", "Time Opened", "Symbol", "Recommendation", "Heat_Score", "Entry Price", "Take Profit", "Stop Loss", "Ticket", "Date Closed", "Time Closed", "Close Price", "Profit/Loss", "Original_Rationale", "Duration_Hours"])

    # Process each closed trade
    for ticket, position_id in closed_trades:
        if ticket in logged:
            # Already logged, just remove from open_trades
            if ticket in open_trades:
                del open_trades[ticket]
            continue

        trade = open_trades.get(ticket)
        if not trade:
            continue
            
        # Get MT5 symbol (normalize Yahoo symbol)
        yahoo_symbol = trade['symbol']
        mt5_symbol = symbols_mt5.get(yahoo_symbol, yahoo_symbol.replace('-', ''))
        
        # Parse open time
        open_time_str = trade['open_time']
        try:
            open_time = datetime.fromisoformat(open_time_str)
        except ValueError:
            # Try alternative format
            try:
                open_time = datetime.strptime(open_time_str[:19], '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                print(f"Could not parse open_time for ticket {ticket}: {open_time_str}")
                continue
        
        # Find closing deals for this position using position_id
        position_deals = deals_by_position.get(position_id, [])
        closing_deals = [deal for deal in position_deals 
                        if deal.entry == mt5.DEAL_ENTRY_OUT 
                        and deal.time >= open_time.timestamp()]
        
        if not closing_deals:
            print(f"No closing deals found for position_id {position_id} (ticket {ticket})")
            # Still remove from open trades as it's not open anymore
            if ticket in open_trades:
                del open_trades[ticket]
            continue

        # Calculate total profit from all deals for this position
        total_profit = sum(deal.profit for deal in position_deals if deal.profit is not None)
        
        # Use the first closing deal for close time and price
        close_deal = closing_deals[0]
        close_time = datetime.fromtimestamp(close_deal.time)
        close_price = close_deal.price
        
        # Calculate duration
        duration_hours = round((close_time - open_time).total_seconds() / 3600, 2)

        # Log to performance CSV
        with open(trade_performance_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.get('recommendation_id', ''),
                open_time.strftime('%Y-%m-%d'), 
                open_time.strftime('%H:%M:%S'),
                trade['symbol'], 
                trade['rec'],
                trade.get('heat_score', ''),
                trade['entry'], 
                trade['tp'], 
                trade['sl'], 
                ticket,
                close_time.strftime('%Y-%m-%d'), 
                close_time.strftime('%H:%M:%S'),
                close_price, 
                total_profit, 
                trade['rationale'],
                duration_hours
            ])

        print(f"Logged closed trade: {trade['symbol']} ticket {ticket}, P/L: {total_profit}")

        # Remove from open_trades
        if ticket in open_trades:
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
    use_cache = False  # Enable AI polling for actual recommendations
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
        writer.writerow(["Date", "Time", "Symbol", "Recommendation", "Heat Score", "Entry Price", "Exit Price", "Take Profit", "Stop Loss", "Rationale", "Recommendation_ID"])

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
                print(f"🔄 Processing {len(symbols[category])} crypto symbols: {symbols[category]}")
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
                        
                        recent_news_alerts = fetch_headlines(yahoo_symbol)
                        print(f"🔍 Processing {yahoo_symbol} -> {mt5_symbol}")
                        print(f"📊 Recent news alerts: {recent_news_alerts[:100]}...")
                        print(f"📈 Intraday data points: {len(combined_data_sorted)}")
                        result = get_ai_recommendation(yahoo_symbol, recent_intraday_data, recent_news_alerts, current_bid, current_ask)
                        if result:
                            print(f"AI Response for {yahoo_symbol}: {result[:200]}...")  # Debug: show first 200 chars
                            # Parse the new JSON format (or extract from mixed text)
                            ai_data = {}
                            try:
                                # First try to extract JSON from response
                                import re
                                json_match = re.search(r'\{.*\}', result.strip())
                                if json_match:
                                    json_str = json_match.group(0)
                                    ai_data = json.loads(json_str)
                                else:
                                    # Try parsing the whole response as JSON
                                    ai_data = json.loads(result.strip())
                            except (json.JSONDecodeError, ValueError):
                                # Fallback: extract individual fields from text format
                                rec_match = re.search(r'(?:recommend|Recommendation).*?["\']?([^"\',\s\n]+)', result, re.IGNORECASE)
                                conf_match = re.search(r'confidence level \((\d+)\)', result, re.IGNORECASE)
                                entry_match = re.search(r'support level at \$([0-9.]+)', result, re.IGNORECASE)
                                # More flexible TP/SL matching
                                tp_match = re.search(r'(?:Take Profit|target price|tp).*?[\$]?(\d+\.\d+)', result, re.IGNORECASE)
                                sl_match = re.search(r'(?:Stop Loss|stop loss|sl).*?[\$]?(\d+\.\d+)', result, re.IGNORECASE)
                                rationale_match = re.search(r'(?:Rationale|Reasoning):\s*(.+)', result, re.IGNORECASE | re.DOTALL)
                                
                                if rec_match:
                                    ai_data['recommendation'] = rec_match.group(1).strip().upper()
                                if conf_match:
                                    ai_data['confidence'] = int(conf_match.group(1))
                                if entry_match:
                                    ai_data['entry'] = float(entry_match.group(1))
                                if tp_match:
                                    ai_data['tp'] = float(tp_match.group(1))
                                if sl_match:
                                    ai_data['sl'] = float(sl_match.group(1))
                                if rationale_match:
                                    ai_data['rationale'] = rationale_match.group(1).strip()
                            
                            # Map JSON fields to our variables
                            recommendation = ai_data.get('recommendation', 'HOLD').upper()
                            # Convert HOLD to Neutral for consistency
                            if recommendation == 'HOLD':
                                recommendation = 'Neutral'
                            elif recommendation not in ['BUY', 'SELL', 'NEUTRAL']:
                                recommendation = 'Neutral'
                            
                            heat_score = float(ai_data.get('confidence', 3))
                            entry_price = float(ai_data.get('entry', 0)) if ai_data.get('entry') else 0
                            take_profit = float(ai_data.get('tp', 0)) if ai_data.get('tp') else 0
                            stop_loss = float(ai_data.get('sl', 0)) if ai_data.get('sl') else 0
                            rationale = ai_data.get('rationale', 'No rationale provided')
                            
                            # Generate default TP/SL if not provided but we have entry price
                            if entry_price > 0 and take_profit == 0 and stop_loss == 0:
                                if recommendation == 'BUY':
                                    take_profit = entry_price * 1.02  # 2% target
                                    stop_loss = entry_price * 0.98    # 2% stop
                                elif recommendation == 'SELL':
                                    take_profit = entry_price * 0.98  # 2% target
                                    stop_loss = entry_price * 1.02    # 2% stop
                            
                            # For neutral recommendations, set prices to 0
                            if recommendation == 'NEUTRAL':
                                entry_price = take_profit = stop_loss = 0
                            
                            # Set exit price same as entry for now (can be refined later)
                            exit_price = entry_price

                            # Boost Heat Score for forex & indices
                            heat_score = heat_score  # Default to AI score
                            if yahoo_symbol in ["EURUSD=X","JPYUSD=X","ZARUSD=X","^IXIC","^DJI","^VIX"]:
                                heat_score = max(heat_score, heat_score)  # Same as ai_heat_score

                            print(f"{yahoo_symbol} -> {recommendation} | Heat Score: {heat_score} | Entry: {entry_price} | Exit: {exit_price} | TP: {take_profit} | SL: {stop_loss}")

                            # Save if Heat Score ≥ threshold and not Neutral, AND all price fields are valid
                            if (heat_score >= heat_threshold and recommendation != "Neutral" and
                                entry_price > 0 and exit_price > 0 and take_profit > 0 and stop_loss > 0 and rationale.strip()):
                                recommendation_id = str(uuid.uuid4())
                                high_heat_symbols.append([today, time_stamp, yahoo_symbol, recommendation, heat_score, entry_price, exit_price, take_profit, stop_loss, rationale, recommendation_id])
                            elif heat_score >= heat_threshold and recommendation != "Neutral":
                                print(f"Skipping {yahoo_symbol} - incomplete or invalid price data (Entry: {entry_price}, Exit: {exit_price}, TP: {take_profit}, SL: {stop_loss})")
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
        # Load recommendations with their IDs and heat scores
        recommendations_dict = {}
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 11:
                        symbol = row[2]
                        recommendations_dict[symbol] = {
                            'recommendation_id': row[10],
                            'heat_score': row[4]
                        }
        except (FileNotFoundError, IndexError):
            pass
            
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                row_data = row + [""] * (11 - len(row))
                date, _, symbol, rec, heat, entry, exit_p, tp, sl, rationale, recommendation_id = row_data
                mt5_symbol = symbols_mt5.get(symbol, symbol)
                direction = 0 if rec.upper() == "BUY" else 1
                positions = mt5.positions_get(symbol=mt5_symbol)
                if positions:
                    pos = positions[0]
                    current_direction = 0 if pos.type == mt5.POSITION_TYPE_BUY else 1
                    if current_direction != direction:
                        close_position(pos.identifier)
                    # Removed print for matching positions to reduce noise
                else:
                    # Only open trades if we have complete price data
                    if tp and sl and tp != "" and sl != "":
                        # Recalculate SL/TP based on current price
                        tick = mt5.symbol_info_tick(mt5_symbol)
                        if tick:
                            current_price = tick.ask if direction == 0 else tick.bid
                            stored_entry = float(entry)
                            stored_sl = float(sl)
                            stored_tp = float(tp)
                            if stored_entry > 0:
                                sl_ratio = stored_sl / stored_entry
                                tp_ratio = stored_tp / stored_entry
                                sl = current_price * sl_ratio
                                tp = current_price * tp_ratio
                            # Now sl and tp are recalculated
                        result = open_trade(mt5_symbol, direction, sl=float(sl) if sl else None, tp=float(tp) if tp else None)
                        if result:
                            ticket, position_id = result
                            print(f"DEBUG: Trade opened successfully, ticket: {ticket}, position_id: {position_id}")
                            # Record the trade
                            open_trades = load_open_trades()
                            open_trades[str(ticket)] = {
                                'symbol': symbol,
                                'rec': rec,
                                'entry': str(current_price) if 'current_price' in locals() else entry,  # Use actual entry price
                                'tp': str(tp),
                                'sl': str(sl),
                                'rationale': rationale,
                                'open_time': datetime.now().isoformat(),
                                'position_id': str(position_id),
                                'recommendation_id': recommendation_id,
                                'heat_score': heat
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

        # Track automatic closures (SL/TP hits) - only if we have positions
        if positions:
            track_automatic_closures(positions)

        # Update TP/SL for open positions based on latest recommendations
        update_positions_from_recommendations()

        # Check for profit target closure (100 units)
        close_all_positions()

        if not positions:
            # Check if CSV has any recommendations
            has_recommendations = False
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    # Check if there's at least one data row
                    for row in reader:
                        if len(row) >= 4:  # Valid recommendation row
                            has_recommendations = True
                            break
            except (FileNotFoundError, StopIteration):
                has_recommendations = False
            
            # If no recommendations, run immediate AI poll
            if not has_recommendations:
                print("No positions and no recommendations found - running immediate AI poll")
                mt5.shutdown()
                run_heat_seeker()  # Immediate poll for new opportunities
                return
            
            # Load last check times for scheduling next poll
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
                            if (now - row_time).total_seconds() < 1 * 3600:  # Consider recent if < 1 hour
                                recent_symbols.add(row[2])
                        except ValueError:
                            pass
        except FileNotFoundError:
            pass

        for pos in positions:
            yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
            if yahoo_symbol in rec_dict:
                rec, heat, entry, exit_p, tp, sl = rec_dict[yahoo_symbol]
                direction = 0 if rec.upper() == "BUY" else 1
                current_direction = 0 if pos.type == mt5.POSITION_TYPE_BUY else 1
                if current_direction != direction:
                    print(f"Recommendation changed for {pos.symbol} ({yahoo_symbol}), closing and reopening.")
                    close_position(pos.identifier)
                    yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
                    update_single_symbol(yahoo_symbol)
                    # Recalculate SL/TP based on current price
                    tick = mt5.symbol_info_tick(pos.symbol)
                    if tick:
                        current_price = tick.ask if direction == 0 else tick.bid
                        stored_entry = float(entry)
                        stored_sl = float(sl)
                        stored_tp = float(tp)
                        if stored_entry > 0:
                            sl_ratio = stored_sl / stored_entry
                            tp_ratio = stored_tp / stored_entry
                            sl = current_price * sl_ratio
                            tp = current_price * tp_ratio
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
                direction = 0 if rec.upper() == "BUY" else 1
                print(f"Reopening position for {mt5_symbol} as it's still in high heat list.")
                # Recalculate SL/TP based on current price
                tick = mt5.symbol_info_tick(mt5_symbol)
                if tick:
                    current_price = tick.ask if direction == 0 else tick.bid
                    stored_entry = float(entry)
                    stored_sl = float(sl)
                    stored_tp = float(tp)
                    if stored_entry > 0:
                        sl_ratio = stored_sl / stored_entry
                        tp_ratio = stored_tp / stored_entry
                        sl = current_price * sl_ratio
                        tp = current_price * tp_ratio
                open_trade(mt5_symbol, direction, sl=float(sl) if sl else None, tp=float(tp) if tp else None)

        # Close positions for symbols not in high heat list
        for pos in positions:
            yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
            if yahoo_symbol not in rec_dict and yahoo_symbol not in recent_symbols:
                print(f"Closing position for {pos.symbol} ({yahoo_symbol}) as it's no longer in high heat list.")
                close_position(pos.identifier)

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
                        # Within bounds, repoll if >1 hour since last update
                        if now - last_update > 1 * 3600:
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

        # Recheck non-high-heat symbols every 5 minutes
        for category, syms in symbols.items():
            for yahoo_symbol in syms:
                if yahoo_symbol not in rec_dict and yahoo_symbol not in recent_symbols and now - last_check.get(yahoo_symbol, 0) > 5 * 60:
                    print(f"Rechecking non-high-heat symbol {yahoo_symbol}")
                    update_single_symbol(yahoo_symbol)
                    last_check[yahoo_symbol] = now
        with open(last_check_file, 'w') as f:
            json.dump(last_check, f)

        mt5.shutdown()
        
        # Log closed trades
        log_closed_trades()
        
        # Sync open trades from MT5 to CSV
        sync_open_trades_from_mt5()
        
        # Run AI account management for profit optimization
        manage_account()
        
        # Execute pending account management actions from CSV
        execute_account_management_actions()

def sync_open_trades_from_mt5():
    """Sync open trades from MT5 to CSV file"""
    if not connect_mt5():
        print("MT5 connection failed for trade sync")
        return
    
    positions = mt5.positions_get()
    if positions is None:
        print("Failed to get positions from MT5")
        mt5.shutdown()
        return
    
    # Load existing open trades
    open_trades = load_open_trades()
    
    # Create new trades dict from MT5 positions
    mt5_trades = {}
    for pos in positions:
        ticket = str(pos.ticket)
        yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
        
        # Get recommendation data from CSV if available
        rec_data = {}
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 11 and row[2] == yahoo_symbol:
                        rec_data = {
                            'rec': row[3],
                            'entry': row[5],
                            'tp': row[7],
                            'sl': row[8],
                            'rationale': row[9],
                            'recommendation_id': row[10],
                            'heat_score': row[4]
                        }
                        break
        except (FileNotFoundError, IndexError):
            pass
        
        # Use existing data if available, otherwise create from position
        if ticket in open_trades:
            # Keep existing data but update current TP/SL
            trade_data = open_trades[ticket].copy()
            trade_data['tp'] = str(pos.tp) if pos.tp > 0 else trade_data.get('tp', '')
            trade_data['sl'] = str(pos.sl) if pos.sl > 0 else trade_data.get('sl', '')
        else:
            # Create new entry
            trade_data = {
                'symbol': yahoo_symbol,
                'rec': rec_data.get('rec', 'BUY' if pos.type == 0 else 'SELL'),
                'entry': str(pos.price_open),
                'tp': str(pos.tp) if pos.tp > 0 else rec_data.get('tp', ''),
                'sl': str(pos.sl) if pos.sl > 0 else rec_data.get('sl', ''),
                'rationale': rec_data.get('rationale', 'Position opened by system'),
                'open_time': datetime.fromtimestamp(pos.time).isoformat(),
                'position_id': str(pos.identifier),
                'recommendation_id': rec_data.get('recommendation_id', ''),
                'heat_score': rec_data.get('heat_score', '')
            }
        
        mt5_trades[ticket] = trade_data
    
    # Save the synced trades
    save_open_trades(mt5_trades)
    print(f"Synced {len(mt5_trades)} open trades from MT5 to CSV")
    
    mt5.shutdown()

def save_account_management_actions(actions):
    """Save account management actions to CSV file"""
    import uuid
    from datetime import datetime
    
    # Ensure CSV exists with header
    try:
        with open(account_management_actions_file, 'r') as f:
            pass
    except FileNotFoundError:
        with open(account_management_actions_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Date", "Time", "Action_ID", "Type", "Position_ID", "Ticket", "Symbol", "Direction", "New_SL", "New_TP", "SL", "TP", "Rationale", "Status"])
    
    # Append new actions
    with open(account_management_actions_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        for action in actions:
            action_id = str(uuid.uuid4())
            action_type = action.get('type', '')
            
            if action_type == 'modify':
                row = [
                    date_str, time_str, action_id, action_type,
                    action.get('position_id', ''), action.get('ticket', ''), '', '',  # position_id, ticket, symbol, direction
                    action.get('new_sl', ''), action.get('new_tp', ''),  # new_sl, new_tp
                    '', '',  # sl, tp (not used for modify)
                    action.get('rationale', ''), 'pending'
                ]
            elif action_type == 'close':
                row = [
                    date_str, time_str, action_id, action_type,
                    action.get('position_id', ''), action.get('ticket', ''), '', '',  # position_id, ticket, symbol, direction
                    '', '', '', '',  # new_sl, new_tp, sl, tp
                    action.get('rationale', ''), 'pending'
                ]
            elif action_type == 'open':
                row = [
                    date_str, time_str, action_id, action_type,
                    '', action.get('symbol', ''), action.get('direction', ''),  # ticket, symbol, direction
                    '', '', action.get('sl', ''), action.get('tp', ''),  # new_sl, new_tp, sl, tp
                    action.get('rationale', ''), 'pending'
                ]
            elif action_type == 'analyze':
                row = [
                    date_str, time_str, action_id, action_type,
                    '', '', '',  # ticket, symbol, direction
                    '', '', '', '',  # new_sl, new_tp, sl, tp
                    str(action), 'pending'  # rationale as full action dict
                ]
            else:
                # Unknown action type
                row = [
                    date_str, time_str, action_id, action_type,
                    '', '', '',  # ticket, symbol, direction
                    '', '', '', '',  # new_sl, new_tp, sl, tp
                    str(action), 'pending'  # rationale as full action dict
                ]
            
            writer.writerow(row)

def manage_account():
    """AI-powered account management to maximize profits"""
    if not connect_mt5():
        print("MT5 connection failed for account management")
        return
    
    # Get account info
    account = mt5.account_info()
    if not account:
        print("Failed to get account info")
        mt5.shutdown()
        return
    
    balance = account.balance
    equity = account.equity
    floating_pl = equity - balance
    
    # Get current positions with P/L
    positions = mt5.positions_get()
    pos_dict = {pos.ticket: {'symbol': pos.symbol, 'pl': pos.profit, 'type': 'buy' if pos.type == 0 else 'sell'} for pos in positions} if positions else {}
    
    # Load open trades
    open_trades = load_open_trades()
    
    # Load recent performance (last 10 trades)
    performance_summary = ""
    try:
        with open(trade_performance_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            trades = list(reader)[-10:]  # last 10
            total_pl = sum(float(row[13]) for row in trades if row[13])
            wins = sum(1 for row in trades if float(row[13]) > 0)
            win_rate = wins / len(trades) if trades else 0
            performance_summary = f"Last {len(trades)} trades: Total P/L {total_pl:.2f}, Win Rate {win_rate:.1%}"
    except (FileNotFoundError, ValueError):
        performance_summary = "No recent performance data"
    
    # Load current recommendations
    recommendations = {}
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 9:
                    symbol = row[2]
                    rec = row[3]
                    heat = float(row[4])
                    entry = float(row[5])
                    tp = float(row[7])
                    sl = float(row[8])
                    recommendations[symbol] = {'rec': rec, 'heat': heat, 'entry': entry, 'tp': tp, 'sl': sl}
    except FileNotFoundError:
        pass
    
    # Build prompt data using actual MT5 position data instead of CSV data
    open_pos_text = ""
    if positions:
        for pos in positions:
            position_id = str(pos.identifier)
            ticket = str(pos.ticket)
            yahoo_symbol = reverse_symbols_mt5.get(pos.symbol, pos.symbol)
            direction = 'buy' if pos.type == 0 else 'sell'
            entry_price = pos.price_open
            current_sl = pos.sl if pos.sl > 0 else 'None'
            current_tp = pos.tp if pos.tp > 0 else 'None'
            pl = pos.profit
            open_pos_text += f"Position_ID {position_id} (Ticket {ticket}): {yahoo_symbol} {direction} Entry:{entry_price:.5f} SL:{current_sl} TP:{current_tp} P/L:{pl:.2f}\n"
    else:
        open_pos_text = "No open positions"
    
    rec_text = "\n".join([f"{s}: {d['rec']} Heat:{d['heat']} Entry:{d['entry']} TP:{d['tp']} SL:{d['sl']}" for s, d in recommendations.items()])
    
    prompt = f"""Aggressive Account Management Analysis for Maximum Profit:

ACCOUNT STATUS: Balance ${balance:.2f}, Equity ${equity:.2f}, Floating P/L ${floating_pl:.2f}

CURRENT OPEN POSITIONS (CRITICAL: Use Position_ID for actions, not Ticket):
{open_pos_text}

RECENT PERFORMANCE ANALYSIS:
{performance_summary}

ACTIVE RECOMMENDATIONS:
{rec_text}

CRITICAL RULES FOR ACTION GENERATION:
1. ONLY use Position_ID values that appear in "CURRENT OPEN POSITIONS" above - do not invent or guess position IDs
2. For MODIFY actions: use "position_id" field (not "ticket"), new_sl must be below entry price for buys, above entry price for sells; new_tp must be above entry price for buys, below entry price for sells
3. For CLOSE actions: use "position_id" field to identify positions to close
4. For OPEN actions: only open new positions if no position exists for that symbol and risk/reward ratio > 2:1
5. Never risk more than 2% of account equity per position

VALID ACTION TYPES:
- modify: Change stop loss/take profit for existing positions
- close: Close specific positions by position_id
- open: Open new positions (rare, only when opportunity is exceptional)
- analyze: Request deeper analysis (no execution required)

EXAMPLE VALID ACTIONS:
{{"actions": [
  {{"type":"modify","position_id":"12345678","new_sl":95.0,"new_tp":110.0,"rationale":"Trailing stop on winning position"}},
  {{"type":"close","position_id":"12345679","rationale":"Cutting losses on losing position"}},
  {{"type":"open","symbol":"ETH-USD","direction":"buy","sl":100.0,"tp":120.0,"rationale":"Strong bullish setup with 2:1 reward ratio"}}
]}}

STRATEGY GUIDELINES:
- For losing positions: Cut losses quickly, reduce position size, or reverse direction if fundamentals change
- For winning positions: Trail stops aggressively, compound profits by adding to winners
- For new positions: Only open if entry has clear edge and stop loss is tight
- Overall: Prioritize capital preservation while aggressively pursuing profits

Return only JSON with specific actions to execute."""
    
    # AI call
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content.strip()
        print(f"Account management AI response: {response}")
        
        # Parse JSON
        try:
            # Extract JSON from response - handle markdown code blocks
            import re
            # First try to extract from markdown code block
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Fallback: extract JSON directly
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response.strip()
            
            # Clean JSON: remove // comments and extra text
            json_str = re.sub(r'//.*', '', json_str)
            # Remove trailing commas or invalid
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r',\s*\}', '}', json_str)
            data = json.loads(json_str)
            actions = data.get('actions', [])
            
            # Add position_id to actions based on current positions
            position_id_map = {str(pos.ticket): str(pos.identifier) for pos in positions} if positions else {}
            
            for action in actions:
                if 'ticket' in action and action['ticket'] in position_id_map:
                    action['position_id'] = position_id_map[action['ticket']]
                elif 'position_id' not in action:
                    # If no position_id and no ticket, or ticket not found, skip this action
                    print(f"Warning: Could not find position_id for action: {action}")
                    continue
            
            # Save actions to CSV instead of executing immediately
            save_account_management_actions(actions)
            print(f"Saved {len(actions)} account management actions to CSV")
            
            # Legacy execution code (commented out)
            # for action in actions:
            #     action_type = action.get('type')
            #     if action_type == 'modify':
            #         ticket = action['ticket']
            #         new_sl = action.get('new_sl')
            #         new_tp = action.get('new_tp')
            #         if modify_position_sl_tp(ticket, new_sl, new_tp):
            #             print(f"Modified position {ticket}: SL={new_sl}, TP={new_tp}")
            #     elif action_type == 'close':
            #         ticket = action['ticket']
            #         # Find position and close
            #         positions = mt5.positions_get(ticket=ticket)
            #         if positions:
            #             pos = positions[0]
            #             close_position(pos)
            #             print(f"Closed position {ticket}")
            #         else:
            #             print(f"Position {ticket} not found for closing")
            #     elif action_type == 'open':
            #         symbol = action['symbol']
            #         mt5_symbol = symbols_mt5.get(symbol, symbol)
            #         direction = 0 if action['direction'].lower() == 'buy' else 1
            #         sl = action.get('sl')
            #         tp = action.get('tp')
            #         result = open_trade(mt5_symbol, direction, sl=sl, tp=tp)
            #         if result:
            #             print(f"Opened {action['direction']} position for {symbol}")
            #     elif action_type == 'analyze':
            #         # Just log the analysis request
            #         print(f"AI requested analysis: {action}")
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse AI response as JSON: {e}")
            print(f"Response was: {response[:500]}...")
    
    except Exception as e:
        print(f"Account management AI error: {e}")
    
    mt5.shutdown()

def cleanup_old_failed_actions():
    """Remove failed actions older than 24 hours to prevent CSV bloat"""
    try:
        # Read all actions
        actions = []
        try:
            with open(account_management_actions_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                actions = list(reader)
        except FileNotFoundError:
            return
        
        if not actions:
            return
        
        # Filter out failed actions older than 24 hours
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        filtered_actions = []
        removed_count = 0
        
        for row in actions:
            if len(row) >= 14:
                status = row[13]  # status column (now at index 13)
                if status == 'failed':
                    try:
                        # Parse date and time
                        date_str = row[0]
                        time_str = row[1]
                        action_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                        
                        if action_datetime < cutoff_time:
                            removed_count += 1
                            continue  # skip this failed action
                    except ValueError:
                        pass  # if we can't parse date, keep the action
                
                filtered_actions.append(row)
        
        if removed_count > 0:
            # Write back filtered actions
            with open(account_management_actions_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(header)
                writer.writerows(filtered_actions)
            print(f"Cleaned up {removed_count} old failed account management actions")
            
    except Exception as e:
        print(f"Error cleaning up old failed actions: {e}")

def execute_account_management_actions():
    """Execute pending account management actions from CSV"""
    print("🔄 Executing account management actions...")
    if not connect_mt5():
        print("MT5 connection failed for executing account management actions")
        return
    
    try:
        # Get current positions for validation
        current_positions = mt5.positions_get()
        current_position_ids = {str(pos.identifier): pos for pos in current_positions} if current_positions else {}
        current_tickets = {str(pos.ticket): pos for pos in current_positions} if current_positions else {}
        
        # Read pending actions
        pending_actions = []
        try:
            with open(account_management_actions_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 14 and row[13] == 'pending':  # status column (now at index 13)
                        # Normalize row to exactly 14 columns
                        normalized_row = row + [''] * (14 - len(row)) if len(row) < 14 else row[:14]
                        try:
                            pending_actions.append({
                                'date': normalized_row[0],
                                'time': normalized_row[1],
                                'action_id': normalized_row[2],
                                'type': normalized_row[3],
                                'position_id': normalized_row[4],  # new position_id column
                                'ticket': normalized_row[5],      # ticket column (now at index 5)
                                'symbol': normalized_row[6],
                                'direction': normalized_row[7],
                                'new_sl': normalized_row[8],
                                'new_tp': normalized_row[9],
                                'sl': normalized_row[10],
                                'tp': normalized_row[11],
                                'rationale': normalized_row[12],
                                'status': normalized_row[13],
                                'row': normalized_row  # keep normalized row for updating
                            })
                        except (IndexError, ValueError) as e:
                            print(f"Skipping malformed CSV row: {row[:5]}... Error: {e}")
                            continue
        except FileNotFoundError:
            print("No account management actions file found")
            mt5.shutdown()
            return
        
        if not pending_actions:
            print("No pending account management actions to execute")
            mt5.shutdown()
            return
        
        print(f"Executing {len(pending_actions)} pending account management actions")
        
        # Execute actions and update status
        updated_rows = []
        try:
            with open(account_management_actions_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Normalize row to exactly 14 columns
                    row = row + [''] * (14 - len(row)) if len(row) < 14 else row[:14]
                    updated_rows.append(row)
        except FileNotFoundError:
            mt5.shutdown()
            return
        
        # Process each pending action
        for action in pending_actions:
            action_type = action['type']
            success = False
            
            try:
                if action_type == 'modify':
                    position_id = action['position_id']
                    ticket = action['ticket']
                    
                    # Try to find position by position_id first, then by ticket as fallback
                    pos = None
                    if position_id and position_id in current_position_ids:
                        pos = current_position_ids[position_id]
                        print(f"Found position {position_id} by position_id")
                    elif ticket and ticket in current_tickets:
                        pos = current_tickets[ticket]
                        print(f"Found position {ticket} by ticket (fallback)")
                    else:
                        print(f"Skipping modify action {action['action_id']}: Position not found (position_id: {position_id}, ticket: {ticket})")
                        continue
                    
                    new_sl = float(action['new_sl']) if action['new_sl'] else None
                    new_tp = float(action['new_tp']) if action['new_tp'] else None
                    
                    # Validate stop levels
                    if new_sl is not None and new_tp is not None:
                        if pos.type == 0:  # Buy position
                            # For buy positions: SL should be below entry, TP should be above entry
                            if new_sl >= pos.price_open or new_tp <= pos.price_open:
                                print(f"Skipping modify action {action['action_id']}: Invalid stops for buy position (SL: {new_sl}, TP: {new_tp}, Entry: {pos.price_open})")
                                continue
                        else:  # Sell position
                            # For sell positions: SL should be above entry, TP should be below entry
                            if new_sl <= pos.price_open or new_tp >= pos.price_open:
                                print(f"Skipping modify action {action['action_id']}: Invalid stops for sell position (SL: {new_sl}, TP: {new_tp}, Entry: {pos.price_open})")
                                continue
                    
                    if modify_position_sl_tp(pos.identifier, new_sl, new_tp):
                        print(f"Executed: Modified position {pos.identifier}: SL={new_sl}, TP={new_tp}")
                        success = True
                        
                elif action_type == 'close':
                    position_id = action['position_id']
                    ticket = action['ticket']
                    
                    # Try to find position by position_id first, then by ticket as fallback
                    pos = None
                    if position_id and position_id in current_position_ids:
                        pos = current_position_ids[position_id]
                    elif ticket and ticket in current_tickets:
                        pos = current_tickets[ticket]
                    else:
                        print(f"Skipping close action {action['action_id']}: Position not found (position_id: {position_id}, ticket: {ticket})")
                        continue
                    
                    close_position(pos.identifier)
                    print(f"Executed: Closed position {pos.identifier}")
                    success = True
                        
                elif action_type == 'open':
                    symbol = action['symbol']
                    if not symbol:
                        print(f"Skipping open action {action['action_id']}: No symbol specified")
                        continue
                        
                    mt5_symbol = symbols_mt5.get(symbol, symbol)
                    direction = 0 if action['direction'].lower() == 'buy' else 1
                    sl = float(action['sl']) if action['sl'] else None
                    tp = float(action['tp']) if action['tp'] else None
                    
                    # Basic validation - check if we already have a position for this symbol
                    existing_positions = [p for p in current_positions if p.symbol == mt5_symbol] if current_positions else []
                    if existing_positions:
                        print(f"Skipping open action {action['action_id']}: Already have position(s) for {mt5_symbol}")
                        continue
                    
                    result = open_trade(mt5_symbol, direction, sl=sl, tp=tp)
                    if result:
                        print(f"Executed: Opened {action['direction']} position for {symbol}")
                        success = True
                        
                elif action_type == 'analyze':
                    # Just log the analysis request
                    print(f"Executed: AI requested analysis: {action['rationale']}")
                    success = True
                    
                else:
                    print(f"Skipping unknown action type: {action_type}")
                    
            except Exception as e:
                print(f"Error executing action {action['action_id']}: {e}")
            
            # Update status in the CSV data
            for i, row in enumerate(updated_rows):
                if len(row) >= 3 and row[2] == action['action_id']:  # action_id column
                    updated_rows[i][13] = 'executed' if success else 'failed'  # status column
                    break
        
        # Write updated CSV
        with open(account_management_actions_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(updated_rows)
        
        print(f"✅ Updated account management actions CSV with {len(pending_actions)} processed actions")
        
        # Clean up old failed actions
        cleanup_old_failed_actions()
        
    except Exception as e:
        print(f"Error in execute_account_management_actions: {e}")
    
    mt5.shutdown()

if __name__ == "__main__":
    watcher = RealTimeMarketWatcher()
    watcher.start()
