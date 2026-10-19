from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(os.path.expanduser("~/data/Stock_data"))
FILE_EARNINGS = DATA_DIR / "earnings_latest.csv"
FILE_PRICES = DATA_DIR / "stock_prices_latest.csv"
CACHE_PATH = DATA_DIR / "Stock_example_1_sorted_v1.npz"
OBS_LEN = 40
FUTURE_LEN = 5
UP_THRESH = 0.05
DOWN_THRESH = -0.05
TEST_SIZE = 0.20
VAL_SIZE = 0.20
PRICE_TICKER_COL = 0
PRICE_DATE_COL = 1
EARN_TICKER_COL = 0
EARN_DATE_COL = 1
EARN_TIME_COL = 5
DESIRED_CHANNELS = ("open","high","low","close","close_adjusted","volume")
DATASET_NAME = "Stock_example_1"
m = OBS_LEN
k = FUTURE_LEN
window = m + k
stride = 1
def _norm(x):
    return (str(x).lower().replace(" ", "").replace("_", "").replace("-", ""))

def _match_col(columns, target):
    target_norm = _norm(target)
    for column in columns:
        if _norm(column) == target_norm:
            return column
    for column in columns:
        column_norm = _norm(column)
        if (target_norm in column_norm or column_norm in target_norm):
            return column
    return None
def _parse_dates(series, name):
    try:
        parsed = pd.to_datetime(series,format="mixed",errors="raise")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series,errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing/unparseable dates.")
    return parsed
def _split_index(dates, fraction):
    dates = np.asarray(dates,dtype="datetime64[ns]")
    n = len(dates)
    nominal = min(max(int(np.floor(n * fraction)),1),n - 1)
    cutoff = dates[nominal]
    split_idx = int(np.searchsorted(dates,cutoff,side="left"))
    if split_idx == 0:
        split_idx = int(np.searchsorted(dates,cutoff,side="right"))
    if split_idx <= 0 or split_idx >= n:
        raise RuntimeError("Could not create a valid "
            "chronological split.")
    return split_idx
def _build_corrected_cache():
    if not FILE_EARNINGS.exists():
        raise FileNotFoundError(FILE_EARNINGS)
    if not FILE_PRICES.exists():
        raise FileNotFoundError(FILE_PRICES)
    print("Loading raw stock data...")
    earnings = pd.read_csv(FILE_EARNINGS)
    prices = pd.read_csv(FILE_PRICES)
    earn_ticker = earnings.columns[EARN_TICKER_COL]
    earn_date = earnings.columns[EARN_DATE_COL]
    earn_time = earnings.columns[EARN_TIME_COL]
    price_ticker = prices.columns[PRICE_TICKER_COL]
    price_date = prices.columns[PRICE_DATE_COL]
    earnings[earn_date] = _parse_dates(earnings[earn_date],"earnings date")
    prices[price_date] = _parse_dates(prices[price_date],"price date")
    earnings[earn_time] = (earnings[earn_time].astype("string").str.strip().str.lower().fillna(""))
    earnings = (earnings.sort_values([earn_date,earn_ticker],kind="mergesort").reset_index(drop=True))
    prices = (prices.sort_values([price_ticker,price_date],kind="mergesort").reset_index(drop=True))
    if not earnings[earn_date].is_monotonic_increasing:
        raise RuntimeError("Earnings rows are not "
            "chronologically ordered.")
    price_order_ok = (prices.groupby(price_ticker,sort=False)[price_date].apply(lambda s: s.is_monotonic_increasing).all())
    if not bool(price_order_ok):
        raise RuntimeError("At least one ticker's "
            "price history is not "
            "chronologically ordered.")
    print("Confirmed chronological ordering "
        "of earnings and per-ticker prices.")
    price_columns = list(prices.columns)
    channels = []
    for requested in DESIRED_CHANNELS:
        found = _match_col(price_columns,requested)
        if found is None:
            raise RuntimeError("Could not find price "
                f"column {requested!r}.")
        channels.append(found)
    close_col = None
    for candidate in ("close_adjusted","adjclose","adj_close","adjustedclose","close"):
        found = _match_col(price_columns,candidate)
        if found in channels:
            close_col = found
            break
    if close_col is None:
        raise RuntimeError("Could not identify "
            "adjusted-close/close column.")
    for column in channels:
        prices[column] = pd.to_numeric(prices[column],errors="coerce")
    ticker_data = {}
    for ticker, group in (prices.groupby(price_ticker,sort=False)):
        ticker_data[ticker] = (group[price_date].to_numpy(dtype="datetime64[ns]"),
            group[channels].to_numpy(dtype=np.float32))
    close_idx = channels.index(close_col)
    X_obs = []
    X_fut = []
    raw_y = []
    sample_dates = []
    sample_symbols = []
    for row in tqdm(earnings.itertuples(index=False,name=None),total=len(earnings),
        desc=("Processing earnings dates"),):
        ticker = row[EARN_TICKER_COL]
        earnings_date = row[EARN_DATE_COL]
        release_time = row[EARN_TIME_COL]
        if pd.isna(release_time):
            continue
        release_time = (str(release_time).strip().lower())
        if release_time not in ("pre","post",):
            continue
        t0_calendar = (earnings_date
            if release_time == "pre"
            else
            earnings_date + pd.Timedelta(days=1))
        item = ticker_data.get(ticker)
        if item is None:
            continue
        (trading_dates,features) = item
        t0_idx = int(np.searchsorted(trading_dates,np.datetime64(t0_calendar,"ns"),
                side="left"))
        if t0_idx >= len(trading_dates):
            continue
        if (t0_idx - OBS_LEN < 0 or t0_idx + FUTURE_LEN >= len(trading_dates)):
            continue
        observed = features[t0_idx- OBS_LEN:t0_idx]
        future_with_t0 = features[t0_idx:t0_idx + FUTURE_LEN + 1]
        if (np.isnan(observed).any() or np.isnan(future_with_t0).any()):
            continue
        t0_close = (future_with_t0[0,close_idx])
        tF_close = (future_with_t0[FUTURE_LEN,close_idx])
        if t0_close == 0:
            net_return = 0.0
        else:
            net_return = float((tF_close- t0_close)/t0_close)
        if (net_return >= UP_THRESH):
            label = 1
        elif (net_return<= DOWN_THRESH):
            label = -1
        else:
            label = 0
        X_obs.append(observed)
        X_fut.append(future_with_t0[1:])
        raw_y.append(label)
        sample_dates.append(trading_dates[t0_idx])
        sample_symbols.append(str(ticker))
    if not X_obs:
        raise RuntimeError("No valid stock-event "
            "samples were created.")
    X_obs = np.asarray(X_obs,dtype=np.float32)
    X_fut = np.asarray(X_fut,dtype=np.float32)
    raw_y = np.asarray(raw_y,dtype=np.int64)
    sample_dates = np.asarray(sample_dates,dtype="datetime64[ns]")
    sample_symbols = np.asarray(sample_symbols,dtype=str)
    order = np.lexsort((sample_symbols,sample_dates.astype("int64")))
    X_obs = X_obs[order]
    X_fut = X_fut[order]
    raw_y = raw_y[order]
    sample_dates = sample_dates[order]
    sample_symbols = sample_symbols[order]
    if not pd.Series(sample_dates).is_monotonic_increasing:
        raise RuntimeError("Constructed samples are "
            "not chronologically ordered.")
    y = (raw_y+ 1).astype(np.int64)
    print("Corrected sample count:",len(y))
    print("Observed shape:",X_obs.shape)
    print("Future shape:",X_fut.shape)
    print("Date range:",sample_dates[0],"to",sample_dates[-1])
    print("Raw class counts:",{ -1:int((raw_y== -1).sum()),0:int((raw_y == 0).sum()),
            1:int((raw_y== 1).sum())})
    print("Channels:",channels)
    np.savez_compressed(CACHE_PATH,X_obs=X_obs,X_fut=X_fut,y=y,raw_y=raw_y,
        dates=sample_dates,symbols=sample_symbols,channels=np.asarray(channels,dtype=str))
    print("Saved corrected cache:",CACHE_PATH)
    return (X_obs,X_fut,y,raw_y,sample_dates,sample_symbols,np.asarray(channels,dtype=str))
def _load_or_build():
    force = (os.environ.get("STOCK1_FORCE_REBUILD","0")== "1")
    if (CACHE_PATH.exists() and not force):
        print("Loading corrected cache:",CACHE_PATH)
        with np.load(CACHE_PATH,allow_pickle=False) as z:
            return (z["X_obs"].astype(np.float32),z["X_fut"].astype(np.float32),
                z["y"].astype(int),z["raw_y"].astype(int),z["dates"].astype("datetime64[ns]"),
                z["symbols"].astype(str),z["channels"].astype(str))
    return (_build_corrected_cache())

(X_obs_all,X_fore_all,y_all,raw_y_all,dates_all,symbols_all,resolved_channels,) = _load_or_build()
split_idx = _split_index(dates_all,1.0 - TEST_SIZE)
X_observed_train = (X_obs_all[:split_idx])
X_forecast_train = (X_fore_all[:split_idx])
y_train = (y_all[:split_idx])
train_dates = (dates_all[:split_idx])
X_observed_test = (X_obs_all[split_idx:])
X_forecast_test = (X_fore_all[split_idx:])
y_test = (y_all[split_idx:])
test_dates = (dates_all[split_idx:])
test_symbols = (symbols_all[split_idx:])
if (train_dates.max() >= test_dates.min()):
    raise RuntimeError("Train/test temporal "
        "separation failed.")
fit_idx = _split_index(train_dates,1.0 - VAL_SIZE)
X_observed_fit = (X_observed_train[:fit_idx])
X_forecast_fit = (X_forecast_train[:fit_idx])
y_fit = (y_train[:fit_idx])
fit_dates = (train_dates[:fit_idx])
X_observed_val = (X_observed_train[fit_idx:])
X_forecast_val = (X_forecast_train[fit_idx:])
y_val = (y_train[fit_idx:])
val_dates = (train_dates[fit_idx:])
if (fit_dates.max() >= val_dates.min()):
    raise RuntimeError("Fit/validation temporal "
        "separation failed.")
def _counts(y):
    values, counts = np.unique(y,return_counts=True)
    return {int(v):int(c) for v, c in zip(values,counts)}
print("Dataset:",DATASET_NAME)
print("Train:",X_observed_train.shape,X_forecast_train.shape,y_train.shape)
print("Fit:",X_observed_fit.shape,X_forecast_fit.shape,y_fit.shape)
print("Validation:",X_observed_val.shape,X_forecast_val.shape,y_val.shape)

print("Test:",X_observed_test.shape,X_forecast_test.shape,y_test.shape)
print("Train dates:",train_dates[0],"to",train_dates[-1])
print("Test dates:",test_dates[0],"to",test_dates[-1])
print("Train counts "
    "(0=down,1=flat,2=up):",_counts(y_train))
print("Test counts "
    "(0=down,1=flat,2=up):",_counts(y_test))
def get_data():
    return {"dataset_name":DATASET_NAME,
        "X_observed_train":X_observed_train,
        "X_forecast_train":X_forecast_train,
        "y_train":y_train,
        "X_observed_test":X_observed_test,
        "X_forecast_test":X_forecast_test,
        "y_test":y_test,
        "X_observed_fit":X_observed_fit,
        "X_forecast_fit":X_forecast_fit,
        "y_fit":y_fit,
        "X_observed_val":X_observed_val,
        "X_forecast_val":X_forecast_val,
        "y_val":y_val,
        "test_starts":test_dates,
        "test_symbols":test_symbols,
        "m":m,"k":k,"window":window,"stride":stride,
        "up_threshold":UP_THRESH,
        "down_threshold":DOWN_THRESH,
        "resolved_channels":resolved_channels}
if __name__ == "__main__":
    print("Standalone validation complete.")