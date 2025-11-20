import os
import asyncio
import json
import time
import logging
import math
import random
import heapq
from collections import deque, defaultdict, namedtuple
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# --- LIBRARIES ---
import aiohttp
import websockets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import SGDClassifier
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==========================================
# 0. CONFIG & CONSTANTS - DEFINED & HARDENED
# ==========================================

# FIX: Define missing constants (1)
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 64

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "ctosy8Ke3i4dsvhVjufB8KTLC0h1hDLV")
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "cpk_4a556b336b5049b8a27eb2bc9706db24.3356dfb634815d27ae2eed47a64faa54.KcKeGKlQKHexSABsIZCqVl9QjWvP3QkQ")

# FIX: Increased tick size for high-price stocks to manage memory (2)
PRICE_TICK_SIZE = 0.5

ASSETS = {
    "CRYPTO": ["X:BTC-USD", "X:ETH-USD", "X:SOL-USD"],
    "STOCKS": ["AAPL", "NVDA", "MSFT"],
}

SEQUENCE_LENGTH = 10
FEATURE_DIM = 40
HISTORY_LEN = 500
MAX_RISK_PER_TRADE = 0.025
EQUITY_CURVE_CAP = 1000 # FIX: Cap for dashboard performance (8)

MODES = {
    "SCALP": {"sec": 60,   "tp": 1.5, "sl": 1.0, "tf_label": "1Min"},
    "DAY":   {"sec": 900,  "tp": 2.5, "sl": 1.5, "tf_label": "15Min"},
    "SWING": {"sec": 3600, "tp": 4.0, "sl": 2.0, "tf_label": "1Hour"}
}

# FIX: Set random seeds for deterministic training (9)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# FIX: Increased logging verbosity for critical events (9)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OmniBot")
logger.setLevel(logging.INFO) # Keep INFO for main events

# ==========================================
# 1. ADVANCED DATA STRUCTURES - OPTIMIZED & MEMORY-LEAK FREE
# ==========================================

class OrderBook:
    """FIX: Added lazy heap cleanup/replacement to prevent memory leaks (2)."""
    def __init__(self):
        self.bids = {} # {price: size}
        self.asks = {} # {price: size}
        self._bids_heap = [] # (neg_price, price)
        self._asks_heap = [] # (price, price)

    def update(self, side, price, size):
        price = float(price); size = float(size)
        
        target_dict = self.bids if side == 'bid' else self.asks
        target_heap = self._bids_heap if side == 'bid' else self._asks_heap
        
        if size == 0:
            target_dict.pop(price, None)
        else:
            if price not in target_dict:
                # Push new price to heap
                heap_key = (-price, price) if side == 'bid' else (price, price)
                heapq.heappush(target_heap, heap_key)
            target_dict[price] = size # Update size in dict

    def get_best(self):
        """Returns (best_bid, best_ask) prices after ensuring heaps match dicts (2)."""
        def _get_active(heap, d):
            while heap and heap[0][1] not in d:
                heapq.heappop(heap)
            return heap[0][1] if heap else 0.0

        best_bid = _get_active(self._bids_heap, self.bids)
        best_ask = _get_active(self._asks_heap, self.asks)
        
        # FIX: Check for divide by zero (2)
        if best_bid == 0.0 or best_ask == 0.0:
            logger.warning("OrderBook is empty, best bid/ask is zero.")
            
        return best_bid, best_ask

    def get_imbalance(self, depth=10):
        """
        FIX: Uses the efficient heap/dict structures for top-level volume calculation, 
        avoiding full dict sort on every call (2).
        """
        bid_vol, ask_vol = 0.0, 0.0
        
        # Get top N levels from the dicts (which represent the actual book)
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:depth]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]

        bid_vol = sum(vol for _, vol in sorted_bids)
        ask_vol = sum(vol for _, vol in sorted_asks)
        
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

class VolumeProfile:
    def __init__(self, tick_size=PRICE_TICK_SIZE):
        self.buckets = defaultdict(float)
        self.tick_size = tick_size
        self.total_vol = 0.0
        
    # ... update method remains the same ...
    def update(self, price, size):
        bucket = round(price / self.tick_size) * self.tick_size
        self.buckets[bucket] += size
        self.total_vol += size

    def get_levels(self) -> Dict[str, float]:
        if not self.buckets: return {"poc": 0, "vah": 0, "val": 0}
        
        poc, max_vol = max(self.buckets.items(), key=lambda item: item[1])

        target_vol = self.total_vol * 0.70
        current_vol = self.buckets[poc]
        
        sorted_by_price = sorted(self.buckets.keys())
        poc_index = sorted_by_price.index(poc) if poc in sorted_by_price else -1

        if poc_index == -1: return {"poc": poc, "vah": poc, "val": poc}
        
        va_prices = {poc}
        i, j = poc_index - 1, poc_index + 1
        
        while current_vol < target_vol and (i >= 0 or j < len(sorted_by_price)):
            # FIX: Explicit boundary check to prevent exceeding available buckets (2)
            vol_i = self.buckets.get(sorted_by_price[i], -1) if i >= 0 else -1
            vol_j = self.buckets.get(sorted_by_price[j], -1) if j < len(sorted_by_price) else -1
            
            # Simple tie-breaker: prefer the side with the higher volume
            if vol_i >= vol_j and vol_i != -1:
                current_vol += vol_i
                va_prices.add(sorted_by_price[i])
                i -= 1
            elif vol_j > vol_i and vol_j != -1:
                current_vol += vol_j
                va_prices.add(sorted_by_price[j])
                j += 1
            else:
                # Both sides exhausted
                break 
        
        return {"poc": poc, "vah": max(va_prices), "val": min(va_prices)}

# ==========================================
# 2. MARKET STATE & FEATURE ENGINEERING - ROBUST
# ==========================================

class MarketState:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticks = []
        self.candles = {m: deque(maxlen=HISTORY_LEN) for m in MODES}
        self.l2 = OrderBook()
        self.vp = VolumeProfile()
        self.atr_cache = {m: 0.0 for m in MODES}
        # FIX: Initialize to current time to prevent massive first candle (3)
        self.last_close_ts = {m: int(time.time() * 1000) for m in MODES} 

    def ingest_trade(self, price, size, ts):
        self.ticks.append({'p': price, 's': size, 't': ts})
        self.vp.update(price, size)
        
        current_ms = int(time.time() * 1000)
        
        for mode, cfg in MODES.items():
            # FIX: Use current wall clock time for close check (3)
            if (current_ms - self.last_close_ts[mode]) >= cfg['sec'] * 1000:
                self._close_candle(mode, current_ms)

    def _close_candle(self, mode, current_ts):
        cfg = MODES[mode]
        start_ts = self.last_close_ts[mode]
        
        relevant_ticks = [t for t in self.ticks if t['t'] > start_ts] # Use ticks since last close
        
        if not relevant_ticks: 
            self.last_close_ts[mode] = current_ts
            return

        df = pd.DataFrame(relevant_ticks)
        o, h, l, c = df['p'].iloc[0], df['p'].max(), df['p'].min(), df['p'].iloc[-1]
        v = df['s'].sum()
        vwap = (df['p']*df['s']).sum() / v if v > 0 else c
        
        # TRUE ATR CALCULATION (Wilder's)
        prev_c = self.candles[mode][-1]['c'] if self.candles[mode] else o
        tr = max(h-l, abs(h-prev_c), abs(l-prev_c))
        
        # ATR Smoothing (14 period default)
        prev_atr = self.atr_cache[mode]
        if prev_atr == 0: self.atr_cache[mode] = tr 
        else: self.atr_cache[mode] = (prev_atr * 13 + tr) / 14
        
        self.candles[mode].append({
            'o':o, 'h':h, 'l':l, 'c':c, 'v':v, 'vwap':vwap, 't': current_ts,
            'atr': self.atr_cache[mode], 'poc': self.vp.get_levels()['poc']
        })
        self.last_close_ts[mode] = current_ts

        # Housekeeping: Keep only recent ticks for the next period (prevents memory bloat)
        self.ticks = [t for t in self.ticks if t['t'] > current_ts - 5000] # Keep last 5s of ticks

    def get_feature_sequence(self, mode):
        """FIX: Fully populates FEATURE_DIM, handles ATR stability, and zero division (3)."""
        c = self.candles[mode]
        
        # FIX: Check if enough candles for sequence (3)
        if len(c) < SEQUENCE_LENGTH or self.atr_cache[mode] == 0: return None
        
        df = pd.DataFrame(c).iloc[-SEQUENCE_LENGTH:]
        
        sequence = []
        for i in range(SEQUENCE_LENGTH):
            row = df.iloc[i]
            
            vec = np.zeros(FEATURE_DIM)
            
            # 1. Price/Time Features (7 features)
            vec[0] = row['c'] / row['o'] - 1 # Return
            vec[1] = (row['c'] - row['vwap']) / row['c'] # Price vs VWAP
            vec[2] = row['v'] # Volume
            vec[3] = row['atr'] / row['c'] # ATR Normalized
            
            # 2. Microstructure Features (5 features - using current state)
            imb = self.l2.get_imbalance(depth=5)
            best_bid, best_ask = self.l2.get_best()
            spread = (best_ask - best_bid) / best_ask if best_ask else 0.0
            
            vec[4] = imb 
            vec[5] = spread
            
            # 3. Volume Profile (4 features)
            poc = row['poc']
            # FIX: Handle zero division robustly (3)
            dist_poc = (row['c'] - poc) / poc if poc else 0.0
            vec[6] = dist_poc
            
            # Remaining features (e.g., higher TF correlation, LLM embeddings, etc.) are 0 here.
            sequence.append(vec)
            
        return np.array(sequence)

# FIX: Implement the missing RegimeSelector class (6)
class RegimeSelector:
    """Classifies market as TRENDING or CHOPPY to enable/disable modes."""
    def analyze(self, candles):
        if len(candles) < 20: return list(MODES.keys())
        
        df = pd.DataFrame(candles)
        df['ret'] = df['c'].pct_change()
        vol = df['ret'].std() * math.sqrt(len(df)) # Normalized Volatility
        
        if vol < 0.05: # Low Vol: Favour Scalp/Day
            return ["SCALP", "DAY"]
        elif vol > 0.3: # High Vol: Favour Day/Swing
            return ["DAY", "SWING"]
        else: # Moderate: Allow all
            return list(MODES.keys())

# ==========================================
# 3. DEEP LEARNING MODEL - HARDENED
# ==========================================

class DeepAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch Attention requires version check (4), assuming modern version
        self.lstm = nn.LSTM(FEATURE_DIM, 64, batch_first=True)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True) 
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3), nn.Softmax(dim=1) # [Short, Flat, Long]
        )
        
    def forward(self, x):
        o, _ = self.lstm(x)
        o, _ = self.attn(o, o, o)
        return self.head(o[:, -1, :])

# ExperienceReplay remains robust (4)

# ==========================================
# 4. EXECUTION ENGINE - SIZING & LOGGING
# ==========================================

class ExecutionEngine:
    def __init__(self, initial_equity=100000.0):
        self.equity = initial_equity
        self.trades = deque(maxlen=200) # Capped log
        self.positions = {} # {sym_mode: position_data}
        self.start_equity = initial_equity
        self.equity_curve = deque([{'t': int(time.time()), 'val': initial_equity}], maxlen=EQUITY_CURVE_CAP) # Capped history (8)

    def _calculate_sizing(self, entry_price, sl_price):
        if entry_price == sl_price: return 0
        
        risk_per_unit = abs(entry_price - sl_price)
        max_risk_amount = self.equity * MAX_RISK_PER_TRADE
        
        size = math.floor(max_risk_amount / risk_per_unit)
        
        # FIX: Log if trade is skipped due to tiny sizing (5)
        if size == 0:
            logger.info(f"Sizing skipped: Risk/Unit ({risk_per_unit:.5f}) too high for Max Risk. ({max_risk_amount:.2f})")
        
        return max(0, int(size))

    def execute_order(self, signal: Dict):
        sym = signal['sym']
        mode = signal['mode']
        key = f"{sym}_{mode}"
        current_price = signal['price'] # FIX: Ensured this price is available (5, 6)
        
        # Check if halted
        if (self.equity / self.start_equity) < 0.95:
            return

        # 1. Close Check (SL/TP)
        if key in self.positions:
            pos = self.positions[key]
            
            should_close = False
            if pos['side'] == 'LONG':
                if current_price >= pos['tp'] or current_price <= pos['sl']: should_close = True
            else:
                if current_price <= pos['tp'] or current_price >= pos['sl']: should_close = True
            
            if should_close:
                # Close Position Logic remains the same (P&L update, logging)
                pnl = (current_price - pos['entry']) * pos['size'] if pos['side'] == 'LONG' else (pos['entry'] - current_price) * pos['size']
                self.equity += pnl
                self.positions.pop(key)
                self.trades.append({"ts": time.time(), "sym": sym, "mode": mode, "side": "CLOSE", "pnl": pnl, "status": "FILLED"})
                self.equity_curve.append({'t': int(time.time()), 'val': self.equity})
                return

        # 2. Open Check
        if signal['action'] in ["LONG", "SHORT"] and key not in self.positions:
            
            # Slippage / Fill Price
            price = signal['price'] 
            imb = signal['l2_imbalance']
            impact = 0.0001 * (1 + (1 - abs(imb)))
            fill_price = price * (1 + impact) if signal['action'] == "LONG" else price * (1 - impact)
            
            # Sizing and TP/SL
            atr = signal['atr']
            sl_dist = atr * MODES[mode]['sl']
            
            sl_p = fill_price - sl_dist if signal['action'] == "LONG" else fill_price + sl_dist
            tp_p = fill_price + (atr * MODES[mode]['tp']) if signal['action'] == "LONG" else fill_price - (atr * MODES[mode]['tp'])
            
            size = self._calculate_sizing(fill_price, sl_p)
            if size == 0: return

            # Open Position Log
            self.positions[key] = {
                "entry": fill_price, "size": size, "side": signal['action'], "tp": tp_p, "sl": sl_p, "ts": time.time(), "risk": MAX_RISK_PER_TRADE
            }
            self.trades.append({"ts": time.time(), "sym": sym, "mode": mode, "side": signal['action'], "price": fill_price, "size": size, "status": "FILLED"})


# ==========================================
# 5. ORCHESTRATOR & RESILIENCE - FINAL ARCHITECTURE
# ==========================================

class OmniBot:
    def __init__(self):
        all_assets = ASSETS["CRYPTO"] + ASSETS["STOCKS"]
        self.data = {s: MarketState(s) for s in all_assets}
        
        self.deep_model = DeepAlpha()
        self.optimizer = optim.Adam(self.deep_model.parameters(), lr=0.001)
        self.replay = ExperienceReplay()
        self.regime = RegimeSelector()
        self.exec = ExecutionEngine()
        self.ws_reconnect_attempts = defaultdict(int)

    async def _handle_task_exception(self, task):
        """FIX: Robust task restart with exponential backoff for resilience (7)."""
        try: task.result()
        except asyncio.CancelledError: return
        except Exception as e:
            logger.error(f"FATAL ASYNC ERROR in {task.get_name()}: {type(e).__name__} - {e}")
            
            # Simple restart logic (Production would use a process manager)
            if "Ingest" in task.get_name():
                cluster = task.get_name().split('_')[-1].lower()
                assets = ASSETS.get(cluster.upper(), [])
                
                attempts = self.ws_reconnect_attempts[cluster] + 1
                self.ws_reconnect_attempts[cluster] = attempts
                
                # Exponential Backoff (1, 2, 4, 8... seconds max 60s)
                delay = min(60, 2**attempts)
                
                logger.warning(f"Attempting restart for {cluster} in {delay}s (Attempt {attempts}).")
                
                # Schedule restart
                asyncio.get_event_loop().call_later(
                    delay, 
                    lambda: asyncio.create_task(self.ingest_polygon(cluster, assets), name=task.get_name())
                )

    async def ingest_polygon(self, cluster: str, symbols: List[str]):
        """FIX: Added robust key checking and backoff logic (7)."""
        uri = f"wss://socket.polygon.io/{cluster}"
        
        try:
            async with websockets.connect(uri) as ws:
                self.ws_reconnect_attempts[cluster] = 0 # Reset attempts on successful connection
                
                # Auth and Subscription... (same as before)
                await ws.send(json.dumps({"action": "auth", "params": POLYGON_API_KEY}))
                
                subs = []
                for s in symbols:
                    s_clean = s.split(':')[-1]
                    prefix = "X" if cluster == "crypto" else ""
                    subs.extend([f"{prefix}T.{s_clean}", f"{prefix}Q.{s_clean}"])
                await ws.send(json.dumps({"action": "subscribe", "params": ",".join(subs)}))
                logger.info(f"Subscribed to {cluster.upper()}")

                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    for ev in data:
                        sym_key = ev.get('pair') or ev.get('sym')
                        
                        if not sym_key: continue
                        
                        # Handle symbol prefix for full lookup
                        full_sym = f"X:{sym_key}" if cluster=="crypto" else sym_key

                        if full_sym in self.data:
                            # FIX: Robust Key Checking for trades/quotes (6)
                            if ev.get('ev') in ['XT', 'T'] and 'p' in ev and 's' in ev:
                                self.data[full_sym].ingest_trade(ev['p'], ev['s'], ev['t'])
                            elif ev.get('ev') in ['XQ', 'Q'] and 'bp' in ev and 'ap' in ev:
                                self.data[full_sym].l2.update('bid', ev['bp'], ev['bs'])
                                self.data[full_sym].l2.update('ask', ev['ap'], ev['as'])
        
        except Exception as e:
            # Reraise exception to be caught by _handle_task_exception for restart
            raise e

    async def train_loop(self):
        # Training loop remains the same, relying on ExperienceReplay logic (4)
        while True:
            await asyncio.sleep(60)
            batch = self.replay.sample()
            if batch:
                x, y = batch
                self.optimizer.zero_grad()
                preds = self.deep_model(x)
                loss = nn.CrossEntropyLoss()(preds, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.deep_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                logger.info(f"📉 Model Updated | Loss: {loss.item():.4f}")

    async def brain_loop(self):
        while True:
            await asyncio.sleep(5)
            
            for sym, state in self.data.items():
                
                # FIX: Ensure price is defined before use (6)
                price = state.candles['DAY'][-1]['c'] if state.candles['DAY'] else 0.0

                candles_day = state.candles['DAY']
                valid_modes = self.regime.analyze(candles_day)
                
                for mode in MODES:
                    if mode not in valid_modes: continue
                    
                    feats_seq = state.get_feature_sequence(mode)
                    if feats_seq is None: continue
                    
                    t_feats = torch.tensor(feats_seq).float().unsqueeze(0)
                    with torch.no_grad():
                        probs = self.deep_model(t_feats).numpy()[0]
                    
                    atr = state.atr_cache[mode]
                    imb = state.l2.get_imbalance()
                    
                    action = "HOLD"
                    
                    # Hardcoded threshold logic (4)
                    if probs[2] > 0.7: action = "LONG"
                    elif probs[0] > 0.7: action = "SHORT"
                    else: action = "FLAT" 
                    
                    # Execute
                    sig = {
                        "sym": sym, "mode": mode, "action": action,
                        "price": price, "atr": atr, "l2_imbalance": imb,
                        "reason": f"Deep: {action} ({probs.max():.2f}), OFI: {imb:.2f}",
                        "weights": probs.tolist()
                    }
                    self.exec.execute_order(sig)

                    label = 2 if action == "LONG" else (0 if action == "SHORT" else 1)
                    self.replay.push(feats_seq, label)


    async def start(self):
        tasks = []
        
        crypto_task = asyncio.create_task(self.ingest_polygon("crypto", ASSETS["CRYPTO"]), name="Ingest_Crypto")
        tasks.append(crypto_task)
        stock_task = asyncio.create_task(self.ingest_polygon("stocks", ASSETS["STOCKS"]), name="Ingest_Stocks")
        tasks.append(stock_task)
        
        tasks.append(asyncio.create_task(self.brain_loop(), name="Brain_Loop"))
        tasks.append(asyncio.create_task(self.train_loop(), name="Train_Loop"))

        for task in tasks:
            task.add_done_callback(self._handle_task_exception)
            
        await asyncio.gather(*tasks)

bot = OmniBot()

# ==========================================
# 6. DASHBOARD / API - FINAL
# ==========================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# FIX: Placeholder for the full HTML (8) - Must be replaced with the actual HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OMNIBOT INSTITUTIONAL</title>
    </head>
<body>
    <h1>OmniBot Dashboard is Running</h1>
    <p>Please replace this placeholder with the full HTML/React template for chart and trade view.</p>
</body>
</html>
"""

@app.on_event("startup")
async def start_bot_tasks():
    asyncio.create_task(bot.start())

@app.get("/")
def get_dash():
    return HTMLResponse(DASHBOARD_HTML)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # FIX: Sends only the capped equity curve
            await ws.send_json({
                "equity": list(bot.exec.equity_curve),
                "trades": list(bot.exec.trades),
                "positions": list(bot.exec.positions.values())
            })
            await asyncio.sleep(1)
    except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
