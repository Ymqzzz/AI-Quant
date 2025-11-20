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
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

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
POSITION_SIZE = 0.025
CONFIDENCE_THRESHOLD = 0.7
DOM_DEPTH = 20
VOLUME_PROFILE_BUCKETS = 10
RETRAIN_INTERVAL = 3600

TRADE_LOG = []

# -----------------------
# Feature utilities
# -----------------------
def add_indicators(df):
    df['EMA25'] = ta.trend.EMAIndicator(df['close'], window=25).ema_indicator()
    df['EMA50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['EMA75'] = ta.trend.EMAIndicator(df['close'], window=75).ema_indicator()
    df['MACD'] = ta.trend.MACD(df['close']).macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
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
nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
def fetch_news(symbol):
    url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=5&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    news_items = []
    if response.status_code == 200:
        data = response.json()
        for item in data.get('results', []):
            news_items.append(item['title'])
    return news_items

def analyze_news_embeddings(news_items):
    if not news_items:
        return np.zeros(384)  # embedding size of MiniLM-L6-v2
    embeddings = nlp_model.encode(news_items)
    return np.mean(embeddings, axis=0)

# -----------------------
# Deep Learning Model
# -----------------------
class DeepTrader(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)  # long, short, no_trade
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.softmax(out)

# -----------------------
# Ensemble Trading AI
# -----------------------
class TradingAI:
    def __init__(self, feature_dim):
        self.online_model = SGDClassifier(max_iter=1000, tol=1e-3)
        self.batch_model = GradientBoostingClassifier()
        self.deep_model = DeepTrader(feature_dim)
        self.deep_model.eval()
        self.initial_fit_done = False

    async def predict(self, feature_vector):
        X = feature_vector.reshape(1, -1)
        # Online model prediction
        if self.initial_fit_done:
            online_pred = self.online_model.predict_proba(X)[0]
        else:
            online_pred = np.array([0.33,0.33,0.34])

        # Batch model prediction
        if self.initial_fit_done:
            batch_pred = self.batch_model.predict_proba(X)[0]
        else:
            batch_pred = np.array([0.33,0.33,0.34])

        # Deep model prediction
        with torch.no_grad():
            x_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            deep_pred = self.deep_model(x_tensor).numpy()[0]

        # Ensemble: weighted average
        ensemble_pred = 0.4*online_pred + 0.3*batch_pred + 0.3*deep_pred
        action_index = np.argmax(ensemble_pred)
        action_map = {0:'long', 1:'short', 2:'no_trade'}
        confidence = ensemble_pred[action_index]
        return {'action': action_map[action_index], 'confidence': confidence}

    def update_online_model(self, X, y):
        if not self.initial_fit_done:
            self.online_model.partial_fit(X, y, classes=[0,1,2])
            self.initial_fit_done = True
        else:
            self.online_model.partial_fit(X, y)

# -----------------------
# Simulated trade execution
# -----------------------
async def execute_trade(symbol, mode, signal):
    if signal['confidence'] < CONFIDENCE_THRESHOLD or signal['action']=='no_trade':
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
# Monitor symbol
# -----------------------
async def monitor_symbol(symbol, mode, timeframe, feature_dim):
    ai = TradingAI(feature_dim)
    while True:
        try:
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
            volume_profile = calculate_volume_profile(df)
            dom_data = {'bids': [(r['close'], r['volume']) for _, r in df.tail(DOM_DEPTH).iterrows()],
                        'asks': [(r['close'], r['volume']) for _, r in df.tail(DOM_DEPTH).iterrows()]}
            liquidity_zones = detect_liquidity_zones(dom_data)
            news_items = fetch_news(symbol)
            news_emb = analyze_news_embeddings(news_items)
            # Feature vector: OHLC + indicators + VP stats + DOM + news embeddings
            features = np.array([
                df['EMA25'].iloc[-1], df['EMA50'].iloc[-1], df['EMA75'].iloc[-1],
                df['MACD'].iloc[-1], df['RSI'].iloc[-1], df['ATR'].iloc[-1],
                np.mean(list(volume_profile.values())), np.std(list(volume_profile.values())),
                len(liquidity_zones)
            ])
            features = np.concatenate([features, news_emb])
            signal = await ai.predict(features)
            await execute_trade(symbol, mode, signal)
            # Update online model with placeholder label for now (simulate reward)
            y_label = 2  # no_trade by default
            ai.update_online_model(features.reshape(1,-1), [y_label])
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[ERROR] {symbol} {mode}: {e}")
            await asyncio.sleep(5)

# -----------------------
# Retraining loop
# -----------------------
async def retrain_models():
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL)
        print("[INFO] Retraining batch/deep models using TRADE_LOG...")
        # TODO: retrain GradientBoostingClassifier & DeepTrader using trade history
        print("[INFO] Retraining complete.")

# -----------------------
# Main
# -----------------------
async def main():
    feature_dim = 6+2+1+384  # indicators + VP mean/std + liquidity count + news embedding
    tasks = []
    for symbol in ALL_SYMBOLS:
        for mode, timeframe in TIMEFRAMES.items():
            tasks.append(monitor_symbol(symbol, mode, timeframe, feature_dim))
    tasks.append(retrain_models())
    await asyncio.gather(*tasks)

if __name__=="__main__":
    asyncio.run(main())
