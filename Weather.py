import os
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
DATA_DIR = os.path.expanduser("~/data/Weather_data")
CITY = os.environ.get("WEATHER_CITY", "Shanghai")
FIRST_YEAR = {"Shanghai": 1984,"NYC": 1984,"Tokyo": 1981}
if CITY not in FIRST_YEAR:
    raise ValueError(f"Unknown weather city: {CITY}")
TEST_YEAR = int(os.environ.get("WEATHER_TEST_YEAR", "2021"))
csv_files = [os.path.join(DATA_DIR, f"{CITY}_{year}.csv") for year in range(FIRST_YEAR[CITY], 2022)]
FEATURES = ["WS10M","RH2M","T2M_MAX","T2M_MIN","PRECTOTCORR"]
lookback = 30
forecast_horizon = 7
m = lookback
k = forecast_horizon
window = m + k
stride = 1
df_list = []
for file in csv_files:
    filename = os.path.basename(file)
    match = re.search(rf"{re.escape(CITY)}_(\d{{4}})\.csv$", filename)
    if match is None:
        raise ValueError(f"Could not extract year from {filename}")
    year_file = int(match.group(1))
    with open(file,"r",encoding="utf-8",errors="replace",) as f:
        lines = f.readlines()
    start_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if line.lstrip().startswith("YEAR")
        ), None)
    if start_idx is None:
        raise ValueError(f"Could not find YEAR header in {file}")
    df = pd.read_csv(file,skiprows=start_idx)
    df["year_file"] = year_file
    df_list.append(df)
combined_df = pd.concat(df_list,ignore_index=True)
required_columns = (["YEAR"] + FEATURES)
missing_columns = [column for column in required_columns if column not in combined_df.columns]
if missing_columns:
    raise ValueError(f"{CITY} data are missing required columns: "
        f"{missing_columns}")
available_years = sorted(combined_df["YEAR"].astype(int).unique())
if TEST_YEAR not in available_years:
    raise ValueError(f"TEST_YEAR={TEST_YEAR} is not present. "
        f"Available years: {available_years}")
if not any(year < TEST_YEAR for year in available_years):
    raise ValueError(f"No training years occur before {TEST_YEAR}.")
data = (combined_df[FEATURES].to_numpy(dtype=np.float32))

hot_threshold = float(np.percentile(combined_df["T2M_MAX"],95))
cold_threshold = float(np.percentile(combined_df["T2M_MIN"],5))
print(f"{CITY} test year: {TEST_YEAR}")
print(f"Available years: "
    f"{available_years}")
print(f"Hot threshold "
    f"(95th percentile T2M_MAX): "
    f"{hot_threshold:.4f}")
print(f"Cold threshold "
    f"(5th percentile T2M_MIN): "
    f"{cold_threshold:.4f}")
X_observed = []
X_forecast = []
y = []
n_samples = (len(data) - lookback- forecast_horizon)
sample_starts = np.arange(n_samples,dtype=int)
for i in sample_starts:
    observed_window = data[i : i + lookback]
    future_window = data[i + lookback :i + lookback + forecast_horizon]
    future_tmax = (future_window[:,FEATURES.index("T2M_MAX"),])
    future_tmin = (future_window[:,FEATURES.index("T2M_MIN"),])
    label = int((np.max(future_tmax)> hot_threshold) or (np.min(future_tmin)< cold_threshold))
    X_observed.append(observed_window)
    X_forecast.append(future_window)
    y.append(label)
X_observed = np.asarray(X_observed,dtype=np.float32)
X_forecast = np.asarray(X_forecast,dtype=np.float32)
y = np.asarray(y,dtype=np.int64)
forecast_end_years = (combined_df["YEAR"].to_numpy()[lookback + forecast_horizon :len(data)])
if (len(forecast_end_years) != len(X_observed)):
    raise RuntimeError("Year vector and window arrays have different lengths.")
train_mask = (forecast_end_years<= (TEST_YEAR - 1))
test_mask = (forecast_end_years == TEST_YEAR)
X_observed_train = (X_observed[train_mask])
X_forecast_train = (X_forecast[train_mask])
y_train = ( y[train_mask])
X_observed_test = (X_observed[test_mask])
X_forecast_test = (X_forecast[test_mask])
y_test = (y[test_mask])
test_starts = (sample_starts[test_mask])
if len(y_train) == 0:
    raise RuntimeError(f"No training examples for "
        f"test year {TEST_YEAR}.")
if len(y_test) == 0:
    raise RuntimeError(f"No test examples for "
        f"test year {TEST_YEAR}.")
print("\nTraining shapes:",X_observed_train.shape, X_forecast_train.shape,
    y_train.shape)
print("Test shapes:",X_observed_test.shape,X_forecast_test.shape,y_test.shape)
print("Training class counts:",Counter(y_train.tolist()))
print("Test class counts:",Counter(y_test.tolist()))
idx = np.arange(len(X_observed_train))
idx_fit, idx_val = (train_test_split(idx,test_size=0.20,random_state=42,stratify=y_train))
X_observed_fit = (X_observed_train[idx_fit].astype(np.float32))
X_forecast_fit = (X_forecast_train[idx_fit].astype(np.float32))
y_fit = (y_train[idx_fit].astype(int))
X_observed_val = (X_observed_train[idx_val].astype(np.float32))
X_forecast_val = (X_forecast_train[idx_val].astype(np.float32))
y_val = (y_train[idx_val].astype(int))
DATASET_NAME = f"{CITY}_{TEST_YEAR}"
def get_data():
    return { "dataset_name":DATASET_NAME,
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
        "y_val": y_val,
        "test_starts":test_starts,
        "m": m,"k": k,"window":window,
        "stride":stride,"test_year": TEST_YEAR,
        "hot_threshold":hot_threshold,
        "cold_threshold":cold_threshold,
        "features":FEATURES,}
if __name__ == "__main__":
    dataset = get_data()
    print("\nDataset:",dataset["dataset_name"])
    print("Train:",dataset["X_observed_train"].shape,dataset["X_forecast_train"].shape,
        dataset["y_train"].shape)
    print("Fit:",dataset["X_observed_fit"].shape,dataset["X_forecast_fit"].shape,
        dataset["y_fit"].shape)
    print("Validation:",dataset["X_observed_val"].shape,dataset["X_forecast_val"].shape,
        dataset["y_val"].shape)
    print("Test:",dataset["X_observed_test"].shape,dataset["X_forecast_test"].shape,
        dataset["y_test"].shape)