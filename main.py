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
# 0. CONFIG & CONSTANTS
# ==========================================
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 64
POLYGON_API_KEY = "ctosy8Ke3i4dsvhVjufB8KTLC0h1hDLV"
CHUTES_API_KEY = "cpk_4a556b336b5049b8a27eb2bc9706db24.3356dfb634815d27ae2eed47a64faa54.KcKeGKlQKHexSABsIZCqVl9QjWvP3QkQ"
PRICE_TICK_SIZE = 0.5

ASSETS = {
    "CRYPTO": ["X:BTC-USD", "X:ETH-USD", "X:SOL-USD"],
    "STOCKS": ["AAPL", "NVDA", "MSFT"],
}

SEQUENCE_LENGTH = 10
FEATURE_DIM = 40
HISTORY_LEN = 500
MAX_RISK_PER_TRADE = 0.025
EQUITY_CURVE_CAP = 1000

MODES = {
    "SCALP": {"sec": 60, "tp": 1.5, "sl": 1.0, "tf_label": "1Min"},
    "DAY": {"sec": 900, "tp": 2.5, "sl": 1.5, "tf_label": "15Min"},
    "SWING": {"sec": 3600, "tp": 4.0, "sl": 2.0, "tf_label": "1Hour"}
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OmniBot")
logger.setLevel(logging.INFO)

# ==========================================
# 1. EXPERIENCE REPLAY (FIXED)
# ==========================================
class ExperienceReplay:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, x, y):
        self.buffer.append((x, y))

    def sample(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        x_batch = torch.tensor([b[0] for b in batch]).float()
        y_batch = torch.tensor([b[1] for b in batch]).long()
        return x_batch, y_batch

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 2. ORDERBOOK & VOLUME PROFILE
# ==========================================
class OrderBook:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self._bids_heap = []
        self._asks_heap = []

    def update(self, side, price, size):
        price = float(price)
        size = float(size)
        target_dict = self.bids if side == "bid" else self.asks
        target_heap = self._bids_heap if side == "bid" else self._asks_heap

        if size == 0:
            target_dict.pop(price, None)
        else:
            if price not in target_dict:
                heap_key = (-price, price) if side == "bid" else (price, price)
                heapq.heappush(target_heap, heap_key)
            target_dict[price] = size

    def get_best(self):
        def _get_active(heap, d):
            while heap and heap[0][1] not in d:
                heapq.heappop(heap)
            return heap[0][1] if heap else 0.0

        best_bid = _get_active(self._bids_heap, self.bids)
        best_ask = _get_active(self._asks_heap, self.asks)
        if best_bid == 0.0 or best_ask == 0.0:
            logger.warning("OrderBook empty: best bid/ask = 0")
        return best_bid, best_ask

    def get_imbalance(self, depth=10):
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

    def update(self, price, size):
        bucket = round(price / self.tick_size) * self.tick_size
        self.buckets[bucket] += size
        self.total_vol += size

    def get_levels(self) -> Dict[str, float]:
        if not self.buckets:
            return {"poc": 0, "vah": 0, "val": 0}
        poc, _ = max(self.buckets.items(), key=lambda item: item[1])
        target_vol = self.total_vol * 0.70
        current_vol = self.buckets[poc]
        sorted_prices = sorted(self.buckets.keys())
        poc_index = sorted_prices.index(poc)
        va_prices = {poc}
        i, j = poc_index - 1, poc_index + 1
        while current_vol < target_vol and (i >= 0 or j < len(sorted_prices)):
            vol_i = self.buckets.get(sorted_prices[i], -1) if i >= 0 else -1
            vol_j = self.buckets.get(sorted_prices[j], -1) if j < len(sorted_prices) else -1
            if vol_i >= vol_j and vol_i != -1:
                current_vol += vol_i
                va_prices.add(sorted_prices[i])
                i -= 1
            elif vol_j > vol_i and vol_j != -1:
                current_vol += vol_j
                va_prices.add(sorted_prices[j])
                j += 1
            else:
                break
        return {"poc": poc, "vah": max(va_prices), "val": min(va_prices)}

# ==========================================
# 3. MARKET STATE
# ==========================================
class MarketState:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticks = []
        self.candles = {m: deque(maxlen=HISTORY_LEN) for m in MODES}
        self.l2 = OrderBook()
        self.vp = VolumeProfile()
        self.atr_cache = {m: 0.0 for m in MODES}
        self.last_close_ts = {m: int(time.time() * 1000) for m in MODES}

    def ingest_trade(self, price, size, ts):
        self.ticks.append({"p": price, "s": size, "t": ts})
        self.vp.update(price, size)
        current_ms = int(time.time() * 1000)
        for mode, cfg in MODES.items():
            if (current_ms - self.last_close_ts[mode]) >= cfg["sec"] * 1000:
                self._close_candle(mode, current_ms)

    def _close_candle(self, mode, current_ts):
        cfg = MODES[mode]
        start_ts = self.last_close_ts[mode]
        relevant_ticks = [t for t in self.ticks if t["t"] > start_ts]
        if not relevant_ticks:
            self.last_close_ts[mode] = current_ts
            return
        df = pd.DataFrame(relevant_ticks)
        o, h, l, c = df["p"].iloc[0], df["p"].max(), df["p"].min(), df["p"].iloc[-1]
        v = df["s"].sum()
        vwap = (df["p"] * df["s"]).sum() / v if v > 0 else c
        prev_c = self.candles[mode][-1]["c"] if self.candles[mode] else o
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        prev_atr = self.atr_cache[mode]
        self.atr_cache[mode] = tr if prev_atr == 0 else (prev_atr * 13 + tr) / 14
        self.candles[mode].append({
            "o": o, "h": h, "l": l, "c": c, "v": v, "vwap": vwap,
            "t": current_ts, "atr": self.atr_cache[mode], "poc": self.vp.get_levels()["poc"]
        })
        self.last_close_ts[mode] = current_ts
        self.ticks = [t for t in self.ticks if t["t"] > current_ts - 5000]

    def get_feature_sequence(self, mode):
        c = self.candles[mode]
        if len(c) < SEQUENCE_LENGTH or self.atr_cache[mode] == 0:
            return None
        df = pd.DataFrame(c).iloc[-SEQUENCE_LENGTH:]
        sequence = []
        for i in range(SEQUENCE_LENGTH):
            row = df.iloc[i]
            vec = np.zeros(FEATURE_DIM)
            vec[0] = row["c"] / row["o"] - 1
            vec[1] = (row["c"] - row["vwap"]) / row["c"]
            vec[2] = row["v"]
            vec[3] = row["atr"] / row["c"]
            imb = self.l2.get_imbalance(depth=5)
            best_bid, best_ask = self.l2.get_best()
            spread = (best_ask - best_bid) / best_ask if best_ask else 0.0
            vec[4] = imb
            vec[5] = spread
            poc = row["poc"]
            vec[6] = (row["c"] - poc) / poc if poc else 0.0
            sequence.append(vec)
        return np.array(sequence)

# ==========================================
# 4. REGIME SELECTOR
# ==========================================
class RegimeSelector:
    def analyze(self, candles):
        if len(candles) < 20: return list(MODES.keys())
        df = pd.DataFrame(candles)
        df["ret"] = df["c"].pct_change()
        vol = df["ret"].std() * math.sqrt(len(df))
        if vol < 0.05: return ["SCALP", "DAY"]
        elif vol > 0.3: return ["DAY", "SWING"]
        else: return list(MODES.keys())

# ==========================================
# 5. DEEP LEARNING MODEL
# ==========================================
class DeepAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(FEATURE_DIM, 64, batch_first=True)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3), nn.Softmax(dim=1)
        )

    def forward(self, x):
        o, _ = self.lstm(x)
        o, _ = self.attn(o, o, o)
        return self.head(o[:, -1, :])

# ==========================================
# 6. EXECUTION ENGINE
# ==========================================
class ExecutionEngine:
    def __init__(self, initial_equity=100000.0):
        self.equity = initial_equity
        self.positions = {}
        self.trades = deque(maxlen=200)
        self.start_equity = initial_equity
        self.equity_curve = deque([{"t": int(time.time()), "val": initial_equity}], maxlen=EQUITY_CURVE_CAP)

    def _calculate_sizing(self, entry_price, sl_price):
        if entry_price == sl_price: return 0
        risk_per_unit = abs(entry_price - sl_price)
        max_risk_amount = self.equity * MAX_RISK_PER_TRADE
        size = math.floor(max_risk_amount / risk_per_unit)
        return max(0, int(size))

    def execute_order(self, signal: Dict):
        sym = signal["sym"]
        mode = signal["mode"]
        key = f"{sym}_{mode}"
        current_price = signal["price"]

        # Close existing position
        if key in self.positions:
            pos = self.positions[key]
            should_close = False
            if pos["side"] == "LONG":
                if current_price >= pos["tp"] or current_price <= pos["sl"]:
                    should_close = True
            else:
                if current_price <= pos["tp"] or current_price >= pos["sl"]:
                    should_close = True
            if should_close:
                pnl = (current_price - pos["entry"]) * pos["size"] if pos["side"] == "LONG" else (pos["entry"] - current_price) * pos["size"]
                self.equity += pnl
                self.positions.pop(key)
                self.trades.append({"ts": time.time(), "sym": sym, "mode": mode, "side": "CLOSE", "pnl": pnl, "status": "FILLED"})
                self.equity_curve.append({"t": int(time.time()), "val": self.equity})
                return

        # Open new position
        if signal["action"] in ["LONG", "SHORT"] and key not in self.positions:
            price = signal["price"]
            imb = signal["l2_imbalance"]
            impact = 0.0001 * (1 + (1 - abs(imb)))
            fill_price = price * (1 + impact) if signal["action"] == "LONG" else price * (1 - impact)
            atr = signal["atr"]
            sl_dist = atr * MODES[mode]["sl"]
            sl_p = fill_price - sl_dist if signal["action"] == "LONG" else fill_price + sl_dist
            tp_p = fill_price + atr * MODES[mode]["tp"] if signal["action"] == "LONG" else fill_price - atr * MODES[mode]["tp"]
            size = self._calculate_sizing(fill_price, sl_p)
            if size == 0: return
            self.positions[key] = {"entry": fill_price, "size": size, "side": signal["action"], "tp": tp_p, "sl": sl_p, "ts": time.time(), "risk": MAX_RISK_PER_TRADE}
            self.trades.append({"ts": time.time(), "sym": sym, "mode": mode, "side": signal["action"], "price": fill_price, "size": size, "status": "FILLED"})

# ==========================================
# 7. OMNIBOT ORCHESTRATOR
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
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"FATAL ASYNC ERROR in {task.get_name()}: {type(e).__name__} - {e}")
            if "Ingest" in task.get_name():
                cluster = task.get_name().split("_")[-1].lower()
                assets = ASSETS.get(cluster.upper(), [])
                attempts = self.ws_reconnect_attempts[cluster] + 1
                self.ws_reconnect_attempts[cluster] = attempts
                delay = min(60, 2**attempts)
                logger.warning(f"Attempting restart for {cluster} in {delay}s (Attempt {attempts}).")
                asyncio.get_event_loop().call_later(delay, lambda: asyncio.create_task(self.ingest_polygon(cluster, assets), name=task.get_name()))

    async def ingest_polygon(self, cluster: str, symbols: List[str]):
        uri = f"wss://socket.polygon.io/{cluster}"
        try:
            async with websockets.connect(uri) as ws:
                self.ws_reconnect_attempts[cluster] = 0
                await ws.send(json.dumps({"action": "auth", "params": POLYGON_API_KEY}))
                subs = []
                for s in symbols:
                    s_clean = s.split(":")[-1]
                    prefix = "X" if cluster == "crypto" else ""
                    subs.extend([f"{prefix}T.{s_clean}", f"{prefix}Q.{s_clean}"])
                await ws.send(json.dumps({"action": "subscribe", "params": ",".join(subs)}))
                logger.info(f"Subscribed to {cluster.upper()}")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    for ev in data:
                        sym_key = ev.get("pair") or ev.get("sym")
                        if not sym_key: continue
                        full_sym = f"X:{sym_key}" if cluster == "crypto" else sym_key
                        if full_sym in self.data:
                            if ev.get("ev") in ["XT", "T"] and "p" in ev and "s" in ev:
                                self.data[full_sym].ingest_trade(ev["p"], ev["s"], ev["t"])
                            elif ev.get("ev") in ["XQ", "Q"] and "bp" in ev and "ap" in ev:
                                self.data[full_sym].l2.update("bid", ev["bp"], ev["bs"])
                                self.data[full_sym].l2.update("ask", ev["ap"], ev["as"])
        except Exception as e:
            raise e

    async def train_loop(self):
        while True:
            await asyncio.sleep(1)
            for sym, ms in self.data.items():
                for mode in MODES:
                    seq = ms.get_feature_sequence(mode)
                    if seq is not None:
                        action = random.randint(0, 2)
                        self.replay.push(seq, action)
            batch = self.replay.sample()
            if batch is not None:
                x_batch, y_batch = batch
                self.optimizer.zero_grad()
                preds = self.deep_model(x_batch)
                loss = nn.CrossEntropyLoss()(preds, y_batch)
                loss.backward()
                self.optimizer.step()

# ==========================================
# 8. MAIN
# ==========================================
if __name__ == "__main__":
    bot = OmniBot()
    loop = asyncio.get_event_loop()
    tasks = []
    for cluster, symbols in ASSETS.items():
        task_name = f"Ingest_{cluster.lower()}"
        task = loop.create_task(bot.ingest_polygon(cluster.lower(), symbols), name=task_name)
        task.add_done_callback(lambda t: asyncio.create_task(bot._handle_task_exception(t)))
        tasks.append(task)
    tasks.append(loop.create_task(bot.train_loop(), name="TrainLoop"))
    loop.run_until_complete(asyncio.wait(tasks))
