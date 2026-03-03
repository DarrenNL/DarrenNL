"""
Alpha Picks portfolio performance calculator.

Model: Seeking Alpha Alpha Picks (S&P Global methodology)
- Each pick receives $1 of new capital, bought at open price on pick date
- Sells use closing price on sell date
- Sell/trim proceeds are reinvested equally into all remaining positions
- Dividends are reinvested (captured via Yahoo Finance adjusted prices)
- Performance = TWR exact (time-weighted return with intraday deposit timing)
"""

import json, datetime, os, sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


def load_config():
    with open("portfolio_config.json") as f:
        return json.load(f)


def fetch_prices(cfg):
    def _ysym(p):
        return p.get("yahoo_symbol", p["symbol"])
    tickers = sorted({_ysym(p) for p in cfg["positions"]} | {"SPY"})
    start = cfg["inception_date"]
    end = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data


def _col(data, level, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        for order in [(level, ticker), (ticker, level)]:
            if order in data.columns:
                return data[order]
    return data.get(level)


def simulate(cfg, data):
    spy_close = _col(data, "Close", "SPY").dropna()
    spy_open = _col(data, "Open", "SPY").dropna()
    trading_days = spy_close.index.sort_values()

    positions = cfg["positions"]
    pos_list = []
    for p in positions:
        ys = p.get("yahoo_symbol", p["symbol"])
        pick = pd.Timestamp(p["pick_date"])
        close_dt = pd.Timestamp(p["close_date"]) if p["close_date"] else None
        trims = sorted(pd.Timestamp(t["date"]) for t in p.get("trims", []))
        cs = _col(data, "Close", ys)
        has_data = cs is not None and not cs.dropna().empty
        pos_list.append({
            "ys": ys, "pick": pick, "close_dt": close_dt,
            "trims": set(trims), "done_trims": set(),
            "has_data": has_data,
            "buy_ovr": p.get("purchase_price_override"),
            "sell_ovr": p.get("sell_price_override"),
        })

    ALLOC = 1.0
    holdings = {}
    lp = {}
    active = set()
    closed = set()
    invested = 0.0

    spy_shares = 0.0
    spy_invested = 0.0

    records = []

    for day in trading_days:
        # Value existing positions at OPEN (for TWR v_open)
        op = {}
        for i in list(active):
            p = pos_list[i]
            if p["has_data"]:
                os_ = _col(data, "Open", p["ys"])
                if os_ is not None and day in os_.index and not pd.isna(os_.loc[day]):
                    op[i] = float(os_.loc[day])
                else:
                    op[i] = lp.get(i, 1.0)
            else:
                op[i] = lp.get(i, 1.0)
        v_open = sum(holdings.get(i, 0) * op.get(i, 1.0) for i in active)
        spy_v_open = spy_shares * float(spy_open.loc[day]) if day in spy_open.index and spy_shares > 0 else 0.0

        # Mark to market at CLOSE for existing positions
        for i in list(active):
            p = pos_list[i]
            if p["has_data"]:
                cs = _col(data, "Close", p["ys"])
                if cs is not None and day in cs.index:
                    v = cs.loc[day]
                    if not pd.isna(v):
                        lp[i] = float(v)

        # Open new positions
        deposit = 0.0
        for i, p in enumerate(pos_list):
            if i in active or i in closed:
                continue
            if day >= p["pick"]:
                buy = p.get("buy_ovr")
                if buy is None:
                    os_ = _col(data, "Open", p["ys"])
                    if os_ is not None and day in os_.index and not pd.isna(os_.loc[day]):
                        buy = float(os_.loc[day])
                if buy is None:
                    cs_ = _col(data, "Close", p["ys"])
                    if cs_ is not None and day in cs_.index and not pd.isna(cs_.loc[day]):
                        buy = float(cs_.loc[day])
                if buy is None:
                    buy = 1.0

                holdings[i] = ALLOC / buy
                lp[i] = buy
                active.add(i)
                deposit += ALLOC
                invested += ALLOC

                # SPY benchmark: buy $1 of SPY on each pick date
                if day in spy_open.index and not pd.isna(spy_open.loc[day]):
                    spy_shares += ALLOC / float(spy_open.loc[day])
                    spy_invested += ALLOC

                # Update to close price
                if p["has_data"]:
                    cs = _col(data, "Close", p["ys"])
                    if cs is not None and day in cs.index:
                        v = cs.loc[day]
                        if not pd.isna(v):
                            lp[i] = float(v)

        # Trims: resize overgrown positions to equal weight, redistribute excess
        n_act = len(active)
        if n_act > 1:
            pv = sum(holdings.get(i, 0) * lp.get(i, 1) for i in active)
            tgt = pv / n_act
            for i in list(active):
                p = pos_list[i]
                for td in list(p["trims"] - p["done_trims"]):
                    if day >= td:
                        cur_val = holdings.get(i, 0) * lp.get(i, 1)
                        if cur_val > tgt:
                            excess = cur_val - tgt
                            holdings[i] -= excess / lp.get(i, 1)
                            others = [j for j in active if j != i]
                            if others:
                                per_o = excess / len(others)
                                for j in others:
                                    pj = lp.get(j, 1.0)
                                    if pj > 0:
                                        holdings[j] += per_o / pj
                        p["done_trims"].add(td)

        # Closes: sell and redistribute proceeds
        for i in list(active):
            p = pos_list[i]
            if p["close_dt"] and day >= p["close_dt"]:
                sp = p["sell_ovr"] if p["sell_ovr"] else lp.get(i, 1)
                proceeds = holdings.get(i, 0) * sp
                active.discard(i)
                closed.add(i)
                holdings[i] = 0
                if active:
                    per_o = proceeds / len(active)
                    for j in active:
                        pj = lp.get(j, 1.0)
                        if pj > 0:
                            holdings[j] += per_o / pj

        v_close = sum(holdings.get(i, 0) * lp.get(i, 1) for i in active)
        spy_v_close = spy_shares * float(spy_close.loc[day]) if day in spy_close.index else 0.0

        records.append({
            "date": day,
            "portfolio_value": v_close,
            "v_open": v_open,
            "deposit": deposit,
            "spy_value": spy_v_close,
            "spy_v_open": spy_v_open,
        })

    df = pd.DataFrame(records).set_index("date")
    return df, invested, spy_invested


def _twr_exact(df, val_col, open_col, deposit_col):
    """TWR exact: split deposit days into overnight + intraday sub-periods."""
    twr = 1.0
    prev_v = 0.0
    for i in range(len(df)):
        row = df.iloc[i]
        v_open = row[open_col]
        deposit = row[deposit_col]
        v_close = row[val_col]

        if deposit > 0 and prev_v > 0:
            overnight_ret = v_open / prev_v
            intraday_ret = v_close / (v_open + deposit) if (v_open + deposit) > 0 else 1.0
            twr *= overnight_ret * intraday_ret
        elif deposit > 0 and prev_v == 0:
            intraday_ret = v_close / deposit if deposit > 0 else 1.0
            twr *= intraday_ret
        else:
            if prev_v > 0:
                twr *= (v_close / prev_v)
        prev_v = v_close
    return twr


def _daily_ret_twr_exact(df, val_col, open_col, deposit_col):
    """Daily return series using TWR exact sub-period logic."""
    prev_v = 0.0
    rets = []
    for i in range(len(df)):
        row = df.iloc[i]
        v_open = row[open_col]
        deposit = row[deposit_col]
        v_close = row[val_col]

        if deposit > 0 and prev_v > 0:
            daily_ret = (v_open / prev_v) * (v_close / (v_open + deposit)) - 1
        elif deposit > 0 and prev_v == 0:
            daily_ret = (v_close / deposit) - 1
        else:
            daily_ret = (v_close / prev_v) - 1 if prev_v > 0 else 0.0
        rets.append(daily_ret)
        prev_v = v_close
    return np.array(rets)


def compute_metrics(df, invested, spy_invested, inception_date):
    v_end = df["portfolio_value"].iloc[-1]
    spy_end = df["spy_value"].iloc[-1]

    # TWR exact for portfolio and SPY benchmark
    twr_port = _twr_exact(df, "portfolio_value", "v_open", "deposit")
    total_performance = round((twr_port - 1) * 100, 2)

    twr_spy = _twr_exact(df, "spy_value", "spy_v_open", "deposit")
    spy_performance = round((twr_spy - 1) * 100, 2)

    inception = pd.Timestamp(inception_date)
    years = (df.index[-1] - inception).days / 365.25
    cagr = round((twr_port ** (1 / years) - 1) * 100, 2)

    # Daily returns using TWR exact sub-period logic
    daily_ret = pd.Series(
        _daily_ret_twr_exact(df, "portfolio_value", "v_open", "deposit"),
        index=df.index,
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Daily SPY returns (TWR exact sub-period)
    spy_daily = pd.Series(
        _daily_ret_twr_exact(df, "spy_value", "spy_v_open", "deposit"),
        index=df.index,
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Alpha and beta via linear regression
    rf_daily = 0.0
    excess_port = daily_ret - rf_daily
    excess_spy = spy_daily - rf_daily
    mask = (excess_spy != 0) & np.isfinite(excess_port) & np.isfinite(excess_spy)
    if mask.sum() > 30:
        slope, intercept, _, _, _ = stats.linregress(excess_spy[mask], excess_port[mask])
        beta = round(slope, 2)
        alpha = round(intercept * 252, 2)
    else:
        beta, alpha = 1.0, 0.0

    # Sharpe ratio (annualized)
    mean_er = excess_port.mean()
    std_er = excess_port.std()
    sharpe = round(mean_er / std_er * np.sqrt(252), 2) if std_er > 0 else 0.0

    # Max drawdown and average drawdown from cumulative TWR
    cum_ret = (1 + daily_ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_drawdown = round(-drawdown.min() * 100, 2)
    avg_drawdown = round(-drawdown[drawdown < 0].mean() * 100, 2) if (drawdown < 0).any() else 0.0

    # Volatility (annualized std of daily returns)
    volatility = round(daily_ret.std() * np.sqrt(252) * 100, 2) if len(daily_ret) > 1 else 0.0

    # VaR and CVaR (95% and 99%)
    var_95 = round(np.percentile(daily_ret, 5) * 100, 2)
    var_99 = round(np.percentile(daily_ret, 1) * 100, 2)
    cvar_95 = round(daily_ret[daily_ret <= np.percentile(daily_ret, 5)].mean() * 100, 2) if (daily_ret <= np.percentile(daily_ret, 5)).any() else var_95
    cvar_99 = round(daily_ret[daily_ret <= np.percentile(daily_ret, 1)].mean() * 100, 2) if (daily_ret <= np.percentile(daily_ret, 1)).any() else var_99

    # Sortino ratio (downside deviation only)
    downside_ret = daily_ret[daily_ret < 0]
    downside_std = np.sqrt((downside_ret ** 2).mean()) if len(downside_ret) > 0 else 1e-10
    sortino = round(mean_er / downside_std * np.sqrt(252), 2) if downside_std > 0 else 0.0

    # Up/down capture vs SPY
    spy_up = spy_daily > 0
    spy_down = spy_daily < 0
    up_capture = round((daily_ret[spy_up].mean() / spy_daily[spy_up].mean() * 100), 2) if spy_up.any() and spy_daily[spy_up].mean() != 0 else 0.0
    down_capture = round((daily_ret[spy_down].mean() / spy_daily[spy_down].mean() * 100), 2) if spy_down.any() and spy_daily[spy_down].mean() != 0 else 0.0

    # Correlation with benchmark
    corr = round(daily_ret.corr(spy_daily), 2) if len(daily_ret) > 1 and spy_daily.std() > 0 else 0.0

    # Win rate
    win_rate = round((daily_ret > 0).mean() * 100, 2)

    # Stability (R-squared of cumulative log returns vs time)
    log_cum = np.log(cum_ret.replace(0, np.nan).dropna())
    if len(log_cum) > 10:
        x = np.arange(len(log_cum))
        _, _, r_value, _, _ = stats.linregress(x, log_cum)
        stability = round(r_value ** 2, 2)
    else:
        stability = 0.0

    # Rolling returns (1Y, 3Y, 5Y in trading days)
    roll_1y = roll_3y = roll_5y = None
    td_1y = min(252, len(daily_ret) - 1)
    td_3y = min(756, len(daily_ret) - 1)
    td_5y = min(1260, len(daily_ret) - 1)
    if td_1y > 0:
        roll_1y = round(((1 + daily_ret.iloc[-td_1y:]).prod() - 1) * 100, 2)
    if td_3y > 0:
        roll_3y = round(((1 + daily_ret.iloc[-td_3y:]).prod() ** (1 / (td_3y / 252)) - 1) * 100, 2)
    if td_5y > 0:
        roll_5y = round(((1 + daily_ret.iloc[-td_5y:]).prod() ** (1 / (td_5y / 252)) - 1) * 100, 2)

    return {
        "total_performance": total_performance,
        "spy_performance": spy_performance,
        "cagr": cagr,
        "alpha": alpha,
        "beta": beta,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "volatility": volatility,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "correlation": corr,
        "win_rate": win_rate,
        "stability": stability,
        "roll_1y": roll_1y,
        "roll_3y": roll_3y,
        "roll_5y": roll_5y,
        "years": round(years, 2),
    }


def write_csv(m):
    with open("performance.csv", "w") as f:
        for k, v in m.items():
            f.write(f"{k},{v}\n")


def update_svgs(m):
    import update
    days = str((datetime.datetime.today() - datetime.datetime(2022, 7, 1)).days)
    metrics = {
        "cagr": f"{m['cagr']}%",
        "days": days,
        "total": f"{m['total_performance']}%",
        "alpha": str(m["alpha"]),
        "beta": str(m["beta"]),
        "sharpe": str(m["sharpe"]),
        "drawdown": f"{m['roll_1y']}%" if m.get('roll_1y') is not None else "N/A",
        "sortino": str(m["sortino"]),
        "var95": f"{m['var_95']}%",
        "cvar95": f"{m['cvar_95']}%",
        "winrate": f"{m['win_rate']}%",
    }
    update.svg_overwrite("dark_ver.svg", metrics)
    update.svg_overwrite("light_ver.svg", metrics)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    prices = fetch_prices(cfg)
    df, invested, spy_invested = simulate(cfg, prices)
    m = compute_metrics(df, invested, spy_invested, cfg["inception_date"])

    print("\n=== Portfolio Metrics ===")
    for k, v in m.items():
        print(f"  {k:>20s}: {v}")

    write_csv(m)
    update_svgs(m)
    print("\nDone.")
