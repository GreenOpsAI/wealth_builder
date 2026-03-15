import yfinance as yf
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

# ── Watchlist ────────────────────────────────────────────────────────────────
SYMBOLS = {
    "US Equity":     ["SPY", "QQQ", "IWM", "RSP"],
    "Diversifier":   ["GLD", "SLV", "DBC", "TLT", "STIP", "EEM", "EFA"],
    "Sector":        ["XLE", "XLV", "XLP", "XLU"],
    "Other":         ["EWT"],
}

EMA_FAST = 10
EMA_SLOW = 20

# ── Fetch + calculate ────────────────────────────────────────────────────────
def get_trend(symbol: str) -> dict:
    try:
        df = yf.download(symbol, period="max", interval="1wk",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < EMA_SLOW + 2:
            return {"symbol": symbol, "error": "no data"}

        close = df["Close"].squeeze()
        ema10 = close.ewm(span=EMA_FAST, adjust=False).mean()
        ema20 = close.ewm(span=EMA_SLOW, adjust=False).mean()

        high = df["High"].squeeze()
        low  = df["Low"].squeeze()
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_pct = round(
            float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
            / float(close.iloc[-1]) * 100, 2
        )

        # Current bar values
        price  = float(close.iloc[-1])
        e10    = float(ema10.iloc[-1])
        e20    = float(ema20.iloc[-1])

        trend  = "Up" if e10 > e20 else "Down"

        # Count consecutive weeks in current trend state + find trend start price
        current_state = e10 > e20
        is_up = ema10 > ema20
        flipped = is_up[is_up != current_state]

        if flipped.empty:
            age_str = f">{len(is_up)}"
            trend_start_price = float(close.iloc[0])
        else:
            last_flip_loc = is_up.index.get_loc(flipped.index[-1])
            age = len(is_up) - 1 - last_flip_loc
            age_str = str(age)
            trend_start_price = float(close.iloc[last_flip_loc + 1])

        performance = round((price - trend_start_price) / trend_start_price * 100, 1)
        perf_str = f"{performance:+.1f}%"

        return {
            "symbol":  symbol,
            "price":   round(price, 2),
            "ema10":   round(e10,   2),
            "ema20":   round(e20,   2),
            "trend":       trend,
            "age":         age_str,
            "atr_pct":     atr_pct,
            "performance": perf_str,
            "error":       None,
        }
    except Exception as ex:
        return {"symbol": symbol, "error": str(ex)}

# ── Print ────────────────────────────────────────────────────────────────────
console = Console()

def make_table(group: str, rows: list[dict]) -> Table:
    table = Table(
        title=f"[bold]{group}[/bold]",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        title_justify="left",
        padding=(0, 1),
    )
    table.add_column("Symbol",    style="bold",    min_width=6)
    table.add_column("ATR%",      justify="right", min_width=6)
    table.add_column("Price",     justify="right", min_width=10)
    table.add_column("EMA10",     justify="right", min_width=9)
    table.add_column("EMA20",     justify="right", min_width=9)
    table.add_column("Trend",     justify="right", min_width=9)
    table.add_column("Age (wks)", justify="right", min_width=6)
    table.add_column("Perf%",     justify="right", min_width=6)

    for r in rows:
        if r.get("error"):
            table.add_row(r["symbol"], "", "", "", "", "", Text(f"error: {r['error']}", style="dim"), "")
        else:
            if r["trend"] == "Up":
                trend_cell = Text("▲ Up",   style="bold green")
            else:
                trend_cell = Text("▼ Down", style="bold red")

            table.add_row(
                r["symbol"],
                f"{r['atr_pct']:.1f}%",
                f"{r['price']:,.2f}",
                f"{r['ema10']:,.2f}",
                f"{r['ema20']:,.2f}",
                trend_cell,
                r["age"],
                r["performance"],
            )
    return table

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    console.print(
        f"\n[bold]Weekly Trend Scan — {datetime.now().strftime('%A %b %d, %Y')}[/bold]\n"
    )

    for group, syms in SYMBOLS.items():
        rows = [get_trend(s) for s in syms]
        console.print(make_table(group, rows))
