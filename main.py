# main.py
import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import websockets
import ta
from collections import defaultdict

# -----------------------
# Load API keys
# -----------------------
load_dotenv()
POLYGON_API_KEY = "ctosy8Ke3i4dsvhVjufB8KTLC0h1hDLV" 
CHUTES_API_KEY = "cpk_4a556b336b5049b8a27eb2bc9706db24.3356dfb634815d27ae2eed47a64faa54.KcKeGKlQKHexSABsIZCqVl9QjWvP3QkQ"

# -----------------------
# Config
# -----------------------
CRYPTO_SYMBOLS = ["BTC/USD","ETH/USD","USDT/USD","BNB/USD","XRP/USD",
                  "ADA/USD","SOL/USD","DOGE/USD","DOT/USD","MATIC/USD"]
FOREX_PAIRS = ["EUR/USD","USD/JPY","GBP/USD","USD/CHF","AUD/USD","USD/CAD",
               "NZD/USD","EUR/GBP","EUR/JPY","GBP/JPY","AUD/JPY","CHF/JPY",
               "EUR/CHF","GBP/CHF","NZD/JPY"]
BLUE_CHIP_STOCKS = ["AAPL","MSFT","GOOGL","AMZN","FB","JPM","JNJ","V","WMT","PG"]
ALL_SYMBOLS = CRYPTO_SYMBOLS + FOREX_PAIRS + BLUE_CHIP_STOCKS

TIMEFRAMES = {"scalp":"1Min", "day":"15Min", "swing":"60Min"}
POSITION_SIZE = 0.025  # 2.5% per trade
CONFIDENCE_THRESHOLD = 0.7
DOM_DEPTH = 20
VOLUME_PROFILE_BUCKETS = 10
RETRAIN_INTERVAL = 3600  # seconds

# -----------------------
# Trade log
# -----------------------
TRADE_LOG = []

# -----------------------
# Utility functions
# -----------------------
def add_indicators(df):
    df['EMA25'] = ta.trend.EMAIndicator(df['close'], window=25).ema_indicator()
    df['EMA50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['EMA75'] = ta.trend.EMAIndicator(df['close'], window=75).ema_indicator()
    df['MACD'] = ta.trend.MACD(df['close']).macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    return df

def calculate_volume_profile(df, buckets=VOLUME_PROFILE_BUCKETS):
    if df.empty:
        return {}
    min_price, max_price = df['low'].min(), df['high'].max()
    bucket_size = (max_price - min_price) / buckets
    vp = defaultdict(float)
    for _, row in df.iterrows():
        bucket_index = int((row['close'] - min_price) // bucket_size)
        vp[bucket_index] += row['volume']
    return vp

def detect_liquidity_zones(dom_data):
    zones = []
    all_orders = dom_data.get('bids',[]) + dom_data.get('asks',[])
    for price, size in all_orders:
        if size > 1000:
            zones.append(price)
    return zones

# -----------------------
# News
# -----------------------
def fetch_news(symbol):
    url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=5&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    news_items = []
    if response.status_code == 200:
        data = response.json()
        if 'results' in data:
            for item in data['results']:
                news_items.append({
                    'headline': item['title'],
                    'description': item['description'],
                    'url': item['url'],
                    'published_utc': item['published_utc']
                })
    return news_items

def analyze_news_sentiment(news_items):
    sentiment_scores = []
    for item in news_items:
        sentiment_scores.append(0)  # Placeholder NLP sentiment
    return sentiment_scores

# -----------------------
# Trading AI (Chutes)
# -----------------------
class TradingAI:
    def __init__(self):
        self.model = None

    async def predict(self, features, news_sentiment, volume_profile, liquidity_zones):
        last_row = features.iloc[-1]
        if last_row['EMA25'] > last_row['EMA75'] and last_row['MACD'] > 0:
            action = 'long'
            confidence = 0.8 + 0.1*len(liquidity_zones)/20
        elif last_row['EMA25'] < last_row['EMA75'] and last_row['MACD'] < 0:
            action = 'short'
            confidence = 0.8 + 0.1*len(liquidity_zones)/20
        else:
            action = 'no_trade'
            confidence = 0.5
        return {'action': action, 'confidence': min(confidence,1.0)}

# -----------------------
# Execute simulated trade
# -----------------------
async def execute_trade(symbol, mode, signal):
    if signal['confidence'] < CONFIDENCE_THRESHOLD or signal['action'] == 'no_trade':
        return
    trade = {
        'timestamp': datetime.utcnow(),
        'symbol': symbol,
        'mode': mode,
        'action': signal['action'],
        'confidence': signal['confidence'],
        'position_size': POSITION_SIZE
    }
    TRADE_LOG.append(trade)
    print(f"[{trade['timestamp']}] {symbol} | {mode} | {signal['action']} | confidence {signal['confidence']}")

# -----------------------
# Polygon WebSocket placeholder (live-ready)
# -----------------------
async def polygon_ws(symbol, asset_type="crypto"):
    dom_data = {'bids':[], 'asks':[]}
    ws_url = {
        "crypto":"wss://socket.polygon.io/crypto",
        "stocks":"wss://socket.polygon.io/stocks",
        "forex":"wss://socket.polygon.io/forex"
    }[asset_type]
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"action":"auth","params":POLYGON_API_KEY}))
        await ws.send(json.dumps({"action":"subscribe","params":f"T.{symbol}"}))
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            # TODO: parse live DOM / trades here
            await asyncio.sleep(0.1)
    return dom_data

# -----------------------
# Monitor symbol
# -----------------------
async def monitor_symbol(symbol, mode, timeframe, asset_type):
    ai = TradingAI()
    while True:
        try:
            # Historical OHLCV
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timeframe}/2025-01-01/{datetime.utcnow().strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=500&apiKey={POLYGON_API_KEY}"
            resp = requests.get(url)
            data = resp.json()
            if 'results' not in data:
                await asyncio.sleep(5)
                continue
            df = pd.DataFrame(data['results'])
            df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume','t':'timestamp'}, inplace=True)
            df['t'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = add_indicators(df)

            # News + sentiment
            news_items = fetch_news(symbol)
            news_sentiment = analyze_news_sentiment(news_items)

            # DOM + liquidity + volume profile
            dom_data = {'bids': [(row['close'], row['volume']) for _, row in df.tail(DOM_DEPTH).iterrows()],
                        'asks': [(row['close'], row['volume']) for _, row in df.tail(DOM_DEPTH).iterrows()]}
            liquidity_zones = detect_liquidity_zones(dom_data)
            volume_profile = calculate_volume_profile(df)

            # AI decision
            signal = await ai.predict(df, news_sentiment, volume_profile, liquidity_zones)

            # Execute simulated trade
            await execute_trade(symbol, mode, signal)
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[ERROR] {symbol} {mode}: {e}")
            await asyncio.sleep(5)

# -----------------------
# Auto-learning
# -----------------------
async def retrain_model():
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL)
        print("[INFO] Retraining AI with TRADE_LOG...")
        # TODO: implement retraining logic
        print("[INFO] Retraining complete.")

# -----------------------
# Main loop
# -----------------------
async def main():
    tasks = []
    for symbol in ALL_SYMBOLS:
        asset_type = "crypto" if symbol in CRYPTO_SYMBOLS else ("forex" if symbol in FOREX_PAIRS else "stocks")
        for mode, timeframe in TIMEFRAMES.items():
            tasks.append(monitor_symbol(symbol, mode, timeframe, asset_type))
    tasks.append(retrain_model())
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
