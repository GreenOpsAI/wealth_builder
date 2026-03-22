"""
trend_history.py — Trend Episode Analyzer
==========================================
For a given symbol, reconstruct every historical trend episode using the
same indicator definitions as trend_scan.py, then answer:

  1. How long do trends of each direction typically last?
  2. How do Strength / Compression / Dist20% / Perf% evolve week-by-week?
  3. Which past episodes look most like right now (week-1 fingerprint match)?

Usage:
    python trend_history.py QQQ
    python trend_history.py QQQ --min-weeks 3        # skip micro-trends
    python trend_history.py SPY --direction Up        # only Up episodes
"""

import sys
import argparse
import math
import yfinance as yf
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from rich.rule import Rule
from rich.panel import Panel

# ── Same constants as trend_scan.py ─────────────────────────────────────────
EMA_FAST             = 10
EMA_SLOW             = 20
ATR_PERIOD           = 14
SLOPE_FAST_LOOKBACK  = 4
SLOPE_SLOW_LOOKBACK  = 7
COMPRESSION_LOOKBACK = 11

console = Console()

# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class WeekSnap:
    """One week's worth of metrics within a trend episode."""
    week_num:    int          # 1-based age within episode
    date:        str
    price:       float
    ema10:       float
    ema20:       float
    atr_pct:     float
    strength:    str          # Strong / Mixed / Weak
    compression: str          # Expanding / Contracting
    dist20_pct:  float        # (price - EMA20) / EMA20 * 100
    perf_pct:    float        # (price - episode_start_price) / start * 100

@dataclass
class TrendEpisode:
    """A complete trend lifecycle from flip to next flip."""
    symbol:       str
    direction:    str          # Up / Down
    start_date:   str
    end_date:     str          # date of last bar (flip happens after)
    duration:     int          # weeks

    entry:        WeekSnap     # week 1 snapshot
    exit:         WeekSnap     # final week snapshot
    weekly:       list         # list[WeekSnap]

    peak_perf:    float        # best perf% reached intra-trend
    peak_week:    int          # which week that occurred
    trough_perf:  float        # worst perf% (opposite direction pullback)

    strength_seq: list         # e.g. ['Strong','Strong','Mixed','Weak']
    comp_seq:     list         # e.g. ['Contracting','Expanding',...]

    is_current:   bool = False  # True for the still-open episode

# ── Indicator engine (vectorised) ────────────────────────────────────────────
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw weekly OHLCV, attach all derived columns.
    Returns a new DataFrame with extra columns.
    """
    # yfinance sometimes returns MultiIndex columns (metric, ticker) — flatten to 1D
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    ema10 = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema20 = close.ewm(span=EMA_SLOW, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr     = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    atr_pct = atr / close * 100

    # Slopes (same lookbacks as trend_scan.py)
    slope10 = (ema10 - ema10.shift(SLOPE_FAST_LOOKBACK - 1)) / ema10.shift(SLOPE_FAST_LOOKBACK - 1) * 100
    slope20 = (ema20 - ema20.shift(SLOPE_SLOW_LOOKBACK - 1)) / ema20.shift(SLOPE_SLOW_LOOKBACK - 1) * 100

    is_up = ema10 > ema20

    # Strength
    def calc_strength(row):
        up    = row["is_up"]
        s10   = row["slope10"]
        s20   = row["slope20"]
        if up:
            if s10 > 0 and s20 > 0: return "Strong"
            if s10 > 0 or  s20 > 0: return "Mixed"
            return "Weak"
        else:
            if s10 < 0 and s20 < 0: return "Strong"
            if s10 < 0 or  s20 < 0: return "Mixed"
            return "Weak"

    tmp = pd.DataFrame({
        "is_up":   is_up,
        "slope10": slope10,
        "slope20": slope20,
    })
    strength = tmp.apply(calc_strength, axis=1)

    # Compression: gap now vs COMPRESSION_LOOKBACK bars ago
    gap     = (ema10 - ema20).abs()
    gap_old = gap.shift(COMPRESSION_LOOKBACK - 1)
    compression = pd.Series(
        ["Expanding" if g > go else "Contracting"
         for g, go in zip(gap, gap_old)],
        index=close.index
    )

    dist20 = (close - ema20) / ema20 * 100

    out = df.copy()
    out["close"]       = close
    out["ema10"]       = ema10
    out["ema20"]       = ema20
    out["atr_pct"]     = atr_pct
    out["is_up"]       = is_up
    out["strength"]    = strength
    out["compression"] = compression
    out["dist20"]      = dist20
    return out

# ── Episode segmentation ─────────────────────────────────────────────────────
def segment_episodes(ind: pd.DataFrame, symbol: str) -> list:
    """
    Walk the indicator frame and cut it into TrendEpisode objects.
    The last episode (current) is marked is_current=True.
    """
    episodes = []

    # Find flip points: where is_up changes
    is_up   = ind["is_up"]
    flips   = is_up[is_up != is_up.shift(1)].index.tolist()

    # Build slice boundaries: (start_idx, end_idx_exclusive)
    boundaries = []
    for i, flip in enumerate(flips):
        loc = ind.index.get_loc(flip)
        end_loc = ind.index.get_loc(flips[i+1]) if i+1 < len(flips) else len(ind)
        boundaries.append((loc, end_loc))

    for start_loc, end_loc in boundaries:
        slice_df = ind.iloc[start_loc:end_loc]
        if len(slice_df) < 1:
            continue

        direction  = "Up" if slice_df["is_up"].iloc[0] else "Down"
        start_date = str(slice_df.index[0].date())
        end_date   = str(slice_df.index[-1].date())
        duration   = len(slice_df)
        start_px   = float(slice_df["close"].iloc[0])
        is_current = (end_loc == len(ind))

        weekly = []
        for wk_num, (idx, row) in enumerate(slice_df.iterrows(), start=1):
            px       = float(row["close"])
            perf_pct = (px - start_px) / start_px * 100
            snap = WeekSnap(
                week_num    = wk_num,
                date        = str(idx.date()),
                price       = round(px, 2),
                ema10       = round(float(row["ema10"]), 2),
                ema20       = round(float(row["ema20"]), 2),
                atr_pct     = round(float(row["atr_pct"]), 2),
                strength    = str(row["strength"]),
                compression = str(row["compression"]),
                dist20_pct  = round(float(row["dist20"]), 2),
                perf_pct    = round(perf_pct, 2),
            )
            weekly.append(snap)

        perfs      = [w.perf_pct for w in weekly]
        peak_perf  = max(perfs, key=abs) if direction == "Up" else min(perfs)
        if direction == "Up":
            peak_perf = max(perfs)
            trough    = min(perfs)
        else:
            peak_perf = min(perfs)   # most negative = peak for Down
            trough    = max(perfs)

        peak_week = perfs.index(peak_perf) + 1

        ep = TrendEpisode(
            symbol       = symbol,
            direction    = direction,
            start_date   = start_date,
            end_date     = end_date,
            duration     = duration,
            entry        = weekly[0],
            exit         = weekly[-1],
            weekly       = weekly,
            peak_perf    = round(peak_perf, 2),
            peak_week    = peak_week,
            trough_perf  = round(trough, 2),
            strength_seq = [w.strength for w in weekly],
            comp_seq     = [w.compression for w in weekly],
            is_current   = is_current,
        )
        episodes.append(ep)

    return episodes

# ── Stats helpers ─────────────────────────────────────────────────────────────
def percentile(data, p):
    s = sorted(data)
    if not s: return 0
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 1)

def median(data):
    return percentile(data, 50)

def mean(data):
    return round(sum(data) / len(data), 1) if data else 0

def stdev(data):
    if len(data) < 2: return 0
    m = mean(data)
    return round(math.sqrt(sum((x - m)**2 for x in data) / (len(data) - 1)), 1)

# ── Fingerprint similarity ────────────────────────────────────────────────────
def similarity_score(current_week1: WeekSnap, hist_week1: WeekSnap, direction: str) -> float:
    """
    Score how similar a historical episode's week-1 is to current week-1.
    Lower = more similar.
    """
    # Dist20% difference (most important — where price sits vs MA)
    dist_diff = abs(current_week1.dist20_pct - hist_week1.dist20_pct)
    # ATR% difference (volatility regime)
    atr_diff  = abs(current_week1.atr_pct - hist_week1.atr_pct)
    # Strength match (categorical — 0 if same, 1 if adjacent, 2 if opposite)
    order = {"Strong": 0, "Mixed": 1, "Weak": 2}
    str_diff = abs(order.get(current_week1.strength, 1) - order.get(hist_week1.strength, 1))
    # Compression match
    comp_diff = 0 if current_week1.compression == hist_week1.compression else 1

    return dist_diff * 2 + atr_diff * 0.5 + str_diff * 3 + comp_diff * 2

# ── Display helpers ───────────────────────────────────────────────────────────
STRENGTH_STYLE = {"Strong": "bold green", "Mixed": "yellow", "Weak": "bold red"}
COMP_STYLE     = {"Expanding": "cyan", "Contracting": "dim"}

def strength_cell(s: str) -> Text:
    return Text(s, style=STRENGTH_STYLE.get(s, ""))

def comp_cell(c: str) -> Text:
    return Text(c, style=COMP_STYLE.get(c, ""))

def perf_cell(p: float) -> Text:
    color = "green" if p >= 0 else "red"
    return Text(f"{p:+.1f}%", style=color)

def dist_cell(d: float) -> Text:
    color = "green" if d >= 0 else "red"
    return Text(f"{d:+.1f}%", style=color)

def trend_cell(direction: str) -> Text:
    return Text("▲ Up" if direction == "Up" else "▼ Down",
                style="bold green" if direction == "Up" else "bold red")

# ── Section 1: Duration statistics ───────────────────────────────────────────
def print_duration_stats(episodes: list, direction: str):
    hist = [e for e in episodes if e.direction == direction and not e.is_current]
    if not hist:
        console.print(f"[dim]No completed {direction} episodes found.[/dim]")
        return

    durations = [e.duration for e in hist]
    console.print(Rule(f"[bold]Duration Stats — {direction} Trends  (n={len(hist)})[/bold]"))

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", padding=(0,1))
    for col in ["Min","P25","Median","P75","Max","Mean","StdDev"]:
        t.add_column(col, justify="right")

    t.add_row(
        str(min(durations)),
        str(int(percentile(durations, 25))),
        str(int(median(durations))),
        str(int(percentile(durations, 75))),
        str(max(durations)),
        str(mean(durations)),
        str(stdev(durations)),
    )
    console.print(t)

    # Bucket distribution
    buckets = {"1–4 wks": 0, "5–12 wks": 0, "13–26 wks": 0, "27–52 wks": 0, ">52 wks": 0}
    for d in durations:
        if d <= 4:   buckets["1–4 wks"]   += 1
        elif d <= 12: buckets["5–12 wks"]  += 1
        elif d <= 26: buckets["13–26 wks"] += 1
        elif d <= 52: buckets["27–52 wks"] += 1
        else:         buckets[">52 wks"]   += 1

    b = Table(box=box.SIMPLE_HEAD, header_style="bold dim", padding=(0,1))
    for k in buckets: b.add_column(k, justify="right")
    b.add_row(*[f"{v}  ({v/len(hist)*100:.0f}%)" for v in buckets.values()])
    console.print(b)
    console.print()

# ── Section 2: Week-by-week average progression ───────────────────────────────
def print_weekly_progression(episodes: list, direction: str, max_weeks: int = 20):
    hist = [e for e in episodes if e.direction == direction and not e.is_current
            and e.duration >= 2]
    if not hist:
        return

    console.print(Rule(f"[bold]Average Weekly Progression — {direction} Trends[/bold]"))
    console.print(f"[dim]Shows how metrics evolve across weeks. N = episodes that reached each week.[/dim]\n")

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", padding=(0,1))
    t.add_column("Wk",          justify="right", min_width=3)
    t.add_column("N",           justify="right", min_width=4)
    t.add_column("Perf% avg",   justify="right", min_width=9)
    t.add_column("Perf% p25",   justify="right", min_width=9)
    t.add_column("Perf% p75",   justify="right", min_width=9)
    t.add_column("Dist20 avg",  justify="right", min_width=10)
    t.add_column("Strong%",     justify="right", min_width=8)
    t.add_column("Mixed%",      justify="right", min_width=7)
    t.add_column("Weak%",       justify="right", min_width=6)
    t.add_column("Expand%",     justify="right", min_width=8)

    for wk in range(1, max_weeks + 1):
        snaps = [e.weekly[wk-1] for e in hist if len(e.weekly) >= wk]
        if not snaps:
            break
        n      = len(snaps)
        perfs  = [s.perf_pct  for s in snaps]
        dists  = [s.dist20_pct for s in snaps]
        strs   = [s.strength   for s in snaps]
        comps  = [s.compression for s in snaps]

        p_avg  = mean(perfs)
        p25    = percentile(perfs, 25)
        p75    = percentile(perfs, 75)
        d_avg  = mean(dists)
        strong = strs.count("Strong") / n * 100
        mixed  = strs.count("Mixed")  / n * 100
        weak   = strs.count("Weak")   / n * 100
        expand = comps.count("Expanding") / n * 100

        p_color = "green" if p_avg >= 0 else "red"
        d_color = "green" if d_avg >= 0 else "red"

        t.add_row(
            str(wk),
            str(n),
            Text(f"{p_avg:+.1f}%",  style=p_color),
            Text(f"{p25:+.1f}%",    style="dim"),
            Text(f"{p75:+.1f}%",    style="dim"),
            Text(f"{d_avg:+.1f}%",  style=d_color),
            Text(f"{strong:.0f}%",  style="bold green" if strong > 50 else ""),
            Text(f"{mixed:.0f}%",   style="yellow"     if mixed  > 50 else ""),
            Text(f"{weak:.0f}%",    style="bold red"   if weak   > 50 else ""),
            Text(f"{expand:.0f}%",  style="cyan"       if expand > 50 else ""),
        )
    console.print(t)
    console.print()

# ── Section 3: Episode summary list ──────────────────────────────────────────
def print_episode_list(episodes: list, direction: str, top_n: int = None):
    hist = [e for e in episodes if e.direction == direction and not e.is_current]
    if not hist:
        return

    console.print(Rule(f"[bold]All Completed {direction} Episodes[/bold]"))

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", padding=(0,1))
    t.add_column("Start",       min_width=11)
    t.add_column("End",         min_width=11)
    t.add_column("Dur",         justify="right", min_width=4)
    t.add_column("Entry Str",   justify="right", min_width=10)
    t.add_column("Entry Comp",  justify="right", min_width=12)
    t.add_column("Entry D20%",  justify="right", min_width=10)
    t.add_column("Peak Perf",   justify="right", min_width=9)
    t.add_column("Peak Wk",     justify="right", min_width=7)
    t.add_column("Exit Perf",   justify="right", min_width=9)
    t.add_column("Exit Str",    justify="right", min_width=9)
    t.add_column("Exit Comp",   justify="right", min_width=12)

    for e in sorted(hist, key=lambda x: x.start_date):
        t.add_row(
            e.start_date,
            e.end_date,
            str(e.duration),
            strength_cell(e.entry.strength),
            comp_cell(e.entry.compression),
            dist_cell(e.entry.dist20_pct),
            perf_cell(e.peak_perf),
            str(e.peak_week),
            perf_cell(e.exit.perf_pct),
            strength_cell(e.exit.strength),
            comp_cell(e.exit.compression),
        )
    console.print(t)
    console.print()

# ── Section 4: Fingerprint match ─────────────────────────────────────────────
def print_fingerprint_match(episodes: list, current: TrendEpisode, top_n: int = 8):
    hist = [e for e in episodes
            if e.direction == current.direction
            and not e.is_current
            and e.duration >= 2]
    if not hist:
        console.print("[dim]No historical episodes to match against.[/dim]")
        return

    scored = []
    for e in hist:
        score = similarity_score(current.entry, e.entry, current.direction)
        scored.append((score, e))

    scored.sort(key=lambda x: x[0])
    top = scored[:top_n]

    console.print(Rule(f"[bold]Fingerprint Match — Most Similar Past {current.direction} Starts[/bold]"))
    console.print(f"[dim]Matched on: entry Dist20%, ATR%, Strength, Compression.  "
                  f"Current entry: Str={current.entry.strength}  "
                  f"Comp={current.entry.compression}  "
                  f"Dist20={current.entry.dist20_pct:+.1f}%  "
                  f"ATR={current.entry.atr_pct:.1f}%[/dim]\n")

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", padding=(0,1))
    t.add_column("Score",      justify="right", min_width=6)
    t.add_column("Start",      min_width=11)
    t.add_column("Dur",        justify="right", min_width=4)
    t.add_column("Entry D20%", justify="right", min_width=10)
    t.add_column("Entry Str",  justify="right", min_width=10)
    t.add_column("Entry Comp", justify="right", min_width=12)
    t.add_column("Peak Perf",  justify="right", min_width=9)
    t.add_column("Peak Wk",    justify="right", min_width=7)
    t.add_column("Exit Perf",  justify="right", min_width=9)
    t.add_column("Exit Str",   justify="right", min_width=9)

    for score, e in top:
        t.add_row(
            f"{score:.1f}",
            e.start_date,
            str(e.duration),
            dist_cell(e.entry.dist20_pct),
            strength_cell(e.entry.strength),
            comp_cell(e.entry.compression),
            perf_cell(e.peak_perf),
            str(e.peak_week),
            perf_cell(e.exit.perf_pct),
            strength_cell(e.exit.strength),
        )
    console.print(t)

    # Aggregate what happened next in matched episodes
    matched_eps = [e for _, e in top]
    if matched_eps:
        console.print()
        console.print("[bold dim]What happened in these matched episodes:[/bold dim]")
        durations  = [e.duration  for e in matched_eps]
        peak_perfs = [e.peak_perf for e in matched_eps]
        exit_perfs = [e.exit.perf_pct for e in matched_eps]

        s = Table(box=box.SIMPLE_HEAD, header_style="bold dim", padding=(0,1))
        s.add_column("Metric",    min_width=20)
        s.add_column("Min",       justify="right")
        s.add_column("Median",    justify="right")
        s.add_column("Max",       justify="right")
        s.add_column("Mean",      justify="right")

        def stat_row(label, data, fmt=lambda x: f"{x:+.1f}%"):
            s.add_row(label,
                      fmt(min(data)), fmt(median(data)),
                      fmt(max(data)), fmt(mean(data)))

        stat_row("Duration (weeks)",    durations,  fmt=lambda x: str(int(x)))
        stat_row("Peak Perf%",          peak_perfs)
        stat_row("Exit Perf%",          exit_perfs)
        console.print(s)
    console.print()

# ── Section 5: Detailed drilldown of specific episode ────────────────────────
def print_episode_detail(ep: TrendEpisode, label: str = ""):
    title = f"Episode Detail — {ep.symbol} {ep.direction}  {ep.start_date}"
    if label: title = f"{label}  |  {title}"
    console.print(Rule(f"[bold]{title}[/bold]"))

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", padding=(0,1))
    t.add_column("Wk",     justify="right", min_width=3)
    t.add_column("Date",   min_width=11)
    t.add_column("Price",  justify="right", min_width=8)
    t.add_column("EMA10",  justify="right", min_width=8)
    t.add_column("EMA20",  justify="right", min_width=8)
    t.add_column("ATR%",   justify="right", min_width=6)
    t.add_column("Str",    justify="right", min_width=8)
    t.add_column("Comp",   justify="right", min_width=12)
    t.add_column("Dist20%",justify="right", min_width=9)
    t.add_column("Perf%",  justify="right", min_width=8)

    for w in ep.weekly:
        t.add_row(
            str(w.week_num),
            w.date,
            f"{w.price:,.2f}",
            f"{w.ema10:,.2f}",
            f"{w.ema20:,.2f}",
            f"{w.atr_pct:.1f}%",
            strength_cell(w.strength),
            comp_cell(w.compression),
            dist_cell(w.dist20_pct),
            perf_cell(w.perf_pct),
        )
    console.print(t)
    console.print(
        f"  Peak: [bold]{ep.peak_perf:+.1f}%[/bold] at week {ep.peak_week}   "
        f"Exit: [bold]{ep.exit.perf_pct:+.1f}%[/bold]   "
        f"Duration: [bold]{ep.duration}[/bold] wks\n"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Trend Episode Analyzer")
    parser.add_argument("symbol",           type=str, help="Ticker symbol, e.g. QQQ")
    parser.add_argument("--min-weeks",      type=int, default=1,
                        help="Exclude episodes shorter than N weeks (default 1)")
    parser.add_argument("--direction",      type=str, default=None,
                        choices=["Up","Down"], help="Focus on one direction only")

    parser.add_argument("--progression-weeks", type=int, default=20,
                        help="How many weeks to show in progression table (default 20)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    console.print(f"\n[bold]Trend Episode Analyzer — {symbol}[/bold]\n")
    console.print(f"[dim]Fetching max history weekly data…[/dim]")

    df = yf.download(symbol, period="max", interval="1wk",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < EMA_SLOW + 2:
        console.print(f"[bold red]Error:[/bold red] not enough data for {symbol}")
        sys.exit(1)

    console.print(f"[dim]Loaded {len(df)} weekly bars "
                  f"({str(df.index[0].date())} → {str(df.index[-1].date())})[/dim]\n")

    ind      = build_indicators(df)
    episodes = segment_episodes(ind, symbol)

    # Filter by min-weeks
    episodes = [e for e in episodes if e.duration >= args.min_weeks or e.is_current]

    # Identify current (open) episode
    current = next((e for e in episodes if e.is_current), None)

    directions = [args.direction] if args.direction else (["Up","Down"] if not current else [current.direction, ("Down" if current.direction == "Up" else "Up")])

    # ── Header: current state ────────────────────────────────────────────────
    if current:
        console.print(Panel(
            f"[bold]Current episode:[/bold]  "
            f"{trend_cell(current.direction)}   "
            f"Age [bold]{current.duration}[/bold] wks   "
            f"Perf [bold]{current.exit.perf_pct:+.1f}%[/bold]   "
            f"Strength [bold]{current.entry.strength}[/bold]→[bold]{current.exit.strength}[/bold]   "
            f"Compression [bold]{current.exit.compression}[/bold]   "
            f"Dist20 [bold]{current.exit.dist20_pct:+.1f}%[/bold]",
            title=symbol, expand=False
        ))
        console.print()

    total_hist = len([e for e in episodes if not e.is_current])
    console.print(f"[dim]Total completed episodes (≥{args.min_weeks} wk): {total_hist}[/dim]\n")

    # ── Sections ─────────────────────────────────────────────────────────────
    primary_dir = current.direction if current else (args.direction or "Down")

    # 1. Duration stats for current direction
    print_duration_stats(episodes, primary_dir)

    # 2. Weekly progression for current direction
    print_weekly_progression(episodes, primary_dir, max_weeks=args.progression_weeks)

    # 3. Episode list for current direction
    print_episode_list(episodes, primary_dir)

    # Also show opposite direction summary briefly
    opp = "Up" if primary_dir == "Down" else "Down"
    if not args.direction:
        console.print(Rule(f"[dim]Opposite direction ({opp}) — duration summary[/dim]"))
        print_duration_stats(episodes, opp)

if __name__ == "__main__":
    main()
