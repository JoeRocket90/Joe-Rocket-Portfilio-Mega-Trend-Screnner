import os
import time
import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo


# ---------------------------
# Run control (Berlin time + manual override)
# ---------------------------
def should_run_now() -> bool:
    # Allow manual runs from GitHub Actions UI
    if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":
        return True
    # Optional override
    if os.getenv("FORCE_RUN") == "1":
        return True

    now = dt.datetime.now(ZoneInfo("Europe/Berlin"))
    return (now.weekday() < 5) and (now.hour in (9, 16))


# ---------------------------
# Indicators
# ---------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]; low = df["Low"]; close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# ---------------------------
# Watchlist
# ---------------------------
def load_watchlist(path: str = "watchlist.txt") -> List[str]:
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line)
    return tickers


# ---------------------------
# Data validation / normalization
# ---------------------------
def utc_today() -> dt.date:
    return dt.datetime.utcnow().date()

def drop_incomplete_today_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Remove partial daily candle if yfinance included today's in-progress bar."""
    if df is None or df.empty:
        return df
    last_date = pd.to_datetime(df.index[-1]).date()
    if last_date == utc_today():
        return df.iloc[:-1]
    return df

def normalize_yfinance_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance can return MultiIndex columns e.g. ('Close','MSFT').
    This normalizes to single-level OHLCV columns: Open/High/Low/Close/Volume
    """
    if df is None or df.empty:
        return df

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # common: level0 = OHLCV, level1 = ticker
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            # fallback: keep first level
            df.columns = df.columns.get_level_values(0)

    # Keep standard columns only
    wanted = ["Open", "High", "Low", "Close", "Volume"]
    cols = [c for c in wanted if c in df.columns]
    return df[cols].copy()

def last_full_bar_date(df: pd.DataFrame) -> Optional[dt.date]:
    if df is None or df.empty:
        return None
    return pd.to_datetime(df.index[-1]).date()

def trading_day_age(last_date: dt.date, now_date: dt.date) -> int:
    a = np.datetime64(last_date)
    b = np.datetime64(now_date)
    return int(np.busday_count(a, b))

def is_data_current(last_date: Optional[dt.date], max_trading_days: int = 2) -> bool:
    if last_date is None:
        return False
    return trading_day_age(last_date, utc_today()) <= max_trading_days


# ---------------------------
# Feature computation + setup rules
# ---------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA10"] = ema(df["Close"], 10)
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["HH20"] = df["High"].rolling(20).max().shift(1)      # 20T-Hoch (ohne heute)
    df["HH252"] = df["High"].rolling(252).max().shift(1)    # ~52W-Hoch (ohne heute)
    return df

def trend_filter_ok(row: pd.Series) -> bool:
    return (row["Close"] > row["EMA50"]) and (row["EMA20"] > row["EMA50"])

def setup_pullback_reclaim(df: pd.DataFrame) -> bool:
    if len(df) < 60:
        return False
    prev = df.iloc[-2]
    cur = df.iloc[-1]
    if not trend_filter_ok(cur):
        return False
    return (prev["Close"] <= prev["EMA20"]) and (cur["Close"] > cur["EMA20"]) and (cur["RSI14"] >= 45)

def setup_breakout_20d(df: pd.DataFrame) -> bool:
    if len(df) < 60:
        return False
    cur = df.iloc[-1]
    if not trend_filter_ok(cur):
        return False
    if pd.isna(cur["HH20"]) or cur["HH20"] <= 0:
        return False
    vol_ok = (cur["Volume"] > 1.2 * cur["VOL_MA20"]) if (not pd.isna(cur["VOL_MA20"])) else False
    return (cur["Close"] > cur["HH20"]) and vol_ok

def setup_newcomer_radar(df: pd.DataFrame) -> bool:
    if len(df) < 260:
        return False
    cur = df.iloc[-1]
    vol_ok = (cur["Volume"] > 1.2 * cur["VOL_MA20"]) if (not pd.isna(cur["VOL_MA20"])) else False

    near_high = (not pd.isna(cur["HH252"])) and (cur["Close"] >= 0.98 * cur["HH252"])
    cond_a = trend_filter_ok(cur) and near_high and vol_ok

    recent = df.iloc[-15:]                 # ~3 Wochen
    sustain = (recent["EMA20"] > recent["EMA50"]).all()
    past = df.iloc[-90:-15]                # ~3 Monate davor
    used_to_be_false = (past["EMA20"] <= past["EMA50"]).mean() > 0.6
    cond_b = (cur["Close"] > cur["EMA50"]) and sustain and used_to_be_false and vol_ok

    return cond_a or cond_b


# ---------------------------
# Output (Signal-Cards)
# ---------------------------
@dataclass
class SignalCard:
    ticker: str
    theme: str
    setup: str
    data_source: str
    last_bar: str
    close: float
    ema20: float
    ema50: float
    ema200: float
    rsi14: float
    vol: float
    vol_ma20: float
    trigger: str
    stop: str
    notes: str

def classify_theme(ticker: str) -> str:
    mapping = {
        "NVDA":"AI / Semis","AMD":"AI / Semis","AVGO":"AI / Semis","ASML":"AI / Semis","TSM":"AI / Semis",
        "MSFT":"AI / Cloud","AMZN":"AI / Cloud","GOOGL":"AI / Cloud","META":"AI / Platforms","AAPL":"Consumer Tech",
        "CRWD":"Cybersecurity","PANW":"Cybersecurity","FTNT":"Cybersecurity","ZS":"Cybersecurity",
        "SYM":"Robotik / Automation","TER":"Automation / Test","ROK":"Industrial Automation","HON":"Industrial Tech","ABB":"Industrial Automation","ISRG":"MedTech / Robotik",
        "NEE":"Energy / Grid","FSLR":"Solar","ENPH":"Solar / Inverters",
        "LMT":"Defense","RTX":"Defense",
        "LLY":"Health / GLP-1","NVO":"Health / GLP-1",
    }
    base = ticker.split(".")[0]
    return mapping.get(ticker, mapping.get(base, "Megatrend"))

def make_signal_card(ticker: str, df: pd.DataFrame, setup: str, data_source: str) -> SignalCard:
    cur = df.iloc[-1]
    last_bar = str(pd.to_datetime(df.index[-1]).date())
    theme = classify_theme(ticker)

    atr14 = float(cur["ATR14"]) if not pd.isna(cur["ATR14"]) else 0.0

    if setup == "Breakout (20T Hoch + Volumen)":
        trigger_level = float(cur["HH20"])
        stop_level = max(float(cur["EMA20"]), float(cur["Close"]) - 2.0 * atr14) if atr14 > 0 else float(cur["EMA20"])
        trigger = f"Tagesschluss > {trigger_level:.2f} (Breakout über 20T-Hoch) UND Volumen > 1.2×MA20"
        stop = f"Invalidation: Tagesschluss < {stop_level:.2f} (EMA20/ATR-Stop)"
    elif setup == "Trend-Pullback (EMA20 Reclaim)":
        stop_level = max(float(cur["EMA50"]), float(cur["Close"]) - 2.0 * atr14) if atr14 > 0 else float(cur["EMA50"])
        trigger = f"Bestätigung: nächster Tag über Tageshoch ODER erneuter Close > EMA20 ({cur['EMA20']:.2f})"
        stop = f"Invalidation: Tagesschluss < {stop_level:.2f} (unter EMA50/ATR)"
    else:
        hh20 = float(cur["HH20"]) if not pd.isna(cur["HH20"]) else float(cur["Close"])
        stop_level = float(cur["EMA50"])
        trigger = f"Bestätigung: Tagesschluss > {hh20:.2f} UND Volumen > 1.2×MA20"
        stop = f"Invalidation: Tagesschluss < {stop_level:.2f} (unter EMA50-Struktur)"

    notes = (
        "KO-/OS-Hinweis: Signal basiert auf dem Underlying. "
        "KO-Level deutlich hinter dem Underlying-Stop wählen (nicht zu eng). "
        "Bei OS eher 3–6 Monate Laufzeit (Theta/IV), moderate Hebel bevorzugen. "
        "Finanzen.Zero: Order manuell nach diesen Levels."
    )

    return SignalCard(
        ticker=ticker,
        theme=theme,
        setup=setup,
        data_source=data_source,
        last_bar=last_bar,
        close=float(cur["Close"]),
        ema20=float(cur["EMA20"]),
        ema50=float(cur["EMA50"]),
        ema200=float(cur["EMA200"]),
        rsi14=float(cur["RSI14"]),
        vol=float(cur["Volume"]),
        vol_ma20=float(cur["VOL_MA20"]) if not pd.isna(cur["VOL_MA20"]) else float("nan"),
        trigger=trigger,
        stop=stop,
        notes=notes,
    )

def score_card(card: SignalCard) -> float:
    vol_factor = (card.vol / card.vol_ma20) if (card.vol_ma20 and not np.isnan(card.vol_ma20) and card.vol_ma20 > 0) else 1.0
    dist = (card.close - card.ema50) / card.ema50 if card.ema50 > 0 else 0.0
    rsi_term = max(0.0, min(1.0, (card.rsi14 - 45) / 25))  # 45..70 -> 0..1
    return 1.0 * rsi_term + 1.0 * dist + 0.5 * (vol_factor - 1.0)

def write_report(cards: List[SignalCard], path_md: str, header: str):
    lines = [header, ""]
    if not cards:
        lines.append("**No trade** – Keine Setups erfüllen heute die Kriterien (Trendfilter + Trigger + Volumen) oder Daten sind nicht aktuell.")
        open(path_md, "w", encoding="utf-8").write("\n".join(lines))
        return

    for c in cards:
        lines += [
            f"### ✅ {c.ticker} — {c.theme}",
            f"**Setup:** {c.setup}",
            f"- **Datenquelle:** {c.data_source}",
            f"- **Letzte geschlossene Tageskerze:** {c.last_bar}",
            f"- **Close:** {c.close:.2f} | **EMA20:** {c.ema20:.2f} | **EMA50:** {c.ema50:.2f} | **EMA200:** {c.ema200:.2f} | **RSI14:** {c.rsi14:.1f}",
            f"- **Volumen:** {c.vol:.0f} | **Vol MA20:** {c.vol_ma20:.0f}",
            "",
            f"**Trigger (Entry):** {c.trigger}",
            f"**Invalidation/Stop (Underlying):** {c.stop}",
            f"**Notiz (KO/OS):** {c.notes}",
            "",
            "---",
            ""
        ]
    open(path_md, "w", encoding="utf-8").write("\n".join(lines))


def main():
    if not should_run_now():
        print("Skip: not target hour in Europe/Berlin (09/16).")
        header = f"# Megatrend Screener – Signale (Daily)\n\n**Zeitpunkt:** {dt.datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M')} (Europe/Berlin)\n\n"
        write_report([], "signals_report.md", header)
        pd.DataFrame([{"status":"skip", "reason":"not target hour (Berlin 09/16)"}]).to_csv("signals.csv", index=False)
        return

    tickers = load_watchlist("watchlist.txt")
    data_source = "Yahoo Finance (yfinance, auto_adjust=True)"
    cards: List[SignalCard] = []
    rows: List[Dict] = []

    for t in tickers:
        try:
            df = yf.download(
                t,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            df = normalize_yfinance_ohlcv(df, t)
            df = drop_incomplete_today_bar(df)

            if df is None or df.empty or len(df) < 220:
                rows.append({"ticker": t, "status": "skip", "reason": "zu wenig Daten"})
                continue

            last_date = last_full_bar_date(df)
            if not is_data_current(last_date, max_trading_days=2):
                rows.append({"ticker": t, "status": "no_trade", "reason": f"Daten nicht aktuell (last_bar={last_date})"})
                continue

            df = compute_features(df.dropna())

            setup = None
            if setup_breakout_20d(df):
                setup = "Breakout (20T Hoch + Volumen)"
            elif setup_pullback_reclaim(df):
                setup = "Trend-Pullback (EMA20 Reclaim)"
            elif setup_newcomer_radar(df):
                setup = "Newcomer-Radar (52W High / Trendwechsel + Volumen)"

            if setup is None:
                rows.append({"ticker": t, "status": "no_trade", "reason": "kein Setup"})
                continue

            card = make_signal_card(t, df, setup, data_source)
            cards.append(card)
            rows.append({
                "ticker": t,
                "status": "signal",
                "theme": card.theme,
                "setup": setup,
                "last_bar": card.last_bar,
                "close": card.close,
                "ema20": card.ema20,
                "ema50": card.ema50,
                "ema200": card.ema200,
                "rsi14": card.rsi14,
                "vol": card.vol,
                "vol_ma20": card.vol_ma20,
                "trigger": card.trigger,
                "stop": card.stop,
            })

            time.sleep(0.7)  # kleine Pause (hilft gegen Yahoo Rate-Limits)

        except Exception as e:
            rows.append({"ticker": t, "status": "error", "reason": str(e)})

    # nur Top 0–3 (Anti-Spam)
    cards = sorted(cards, key=score_card, reverse=True)[:3]

    pd.DataFrame(rows).to_csv("signals.csv", index=False)

    header = (
        f"# Megatrend Screener – Signale (Daily)\n\n"
        f"**Zeitpunkt:** {dt.datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M')} (Europe/Berlin)\n\n"
        f"**System:** Trendfilter (Close>EMA50 & EMA20>EMA50) + Pullback/Breakout/Newcomer-Radar\n"
    )
    write_report(cards, "signals_report.md", header)

    if cards:
        print("TOP SETUPS:")
        for c in cards:
            print(f"- {c.ticker} | {c.setup} | last_bar {c.last_bar}")
    else:
        print("No trade")


if __name__ == "__main__":
    main()
