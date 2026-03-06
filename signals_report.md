# Megatrend Screener – Signale (Daily)

**Zeitpunkt:** 2026-03-06 16:56 (Europe/Berlin)

**System:** Trendfilter (Close>EMA50 & EMA20>EMA50) + Pullback/Breakout/Newcomer-Radar


### ✅ TER — Automation / Test
**Setup:** Trend-Pullback (EMA20 Reclaim)
- **Datenquelle:** Yahoo Finance (yfinance, auto_adjust=True)
- **Letzte geschlossene Tageskerze:** 2026-03-05
- **Close:** 305.58 | **EMA20:** 305.48 | **EMA50:** 269.83 | **EMA200:** 186.28 | **RSI14:** 54.0
- **Volumen:** 3936100 | **Vol MA20:** 3440065

**Trigger (Entry):** Bestätigung: nächster Tag über Tageshoch ODER erneuter Close > EMA20 (305.48)
**Invalidation/Stop (Underlying):** Invalidation: Tagesschluss < 272.12 (unter EMA50/ATR)
**Notiz (KO/OS):** KO-/OS-Hinweis: Signal basiert auf dem Underlying. KO-Level deutlich hinter dem Underlying-Stop wählen (nicht zu eng). Bei OS eher 3–6 Monate Laufzeit (Theta/IV), moderate Hebel bevorzugen. Finanzen.Zero: Order manuell nach diesen Levels.

---
