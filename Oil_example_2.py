import pathlib
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
CSV_PATH = "~/data/ETDataset/ETT-small/ETTm2.csv"
m = 4
k = 48
tau = 30
train_frac = 0.8
stride = 1
COLS = ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]
df = (pd.read_csv(pathlib.Path(CSV_PATH).expanduser(),parse_dates=["date"],).sort_values("date"))
X = df[COLS].to_numpy(dtype=np.float32)
T = len(X)
window = m + k
cutoff_idx = int(np.floor(train_frac * T))
train_starts = np.arange(0,cutoff_idx - window + 1,stride)
test_starts = np.arange(cutoff_idx,T - window + 1,stride)
assert np.all(train_starts + window <= cutoff_idx)
assert np.all(test_starts >= cutoff_idx)
print(f"Total windows          : "
    f"{T - window + 1}")
print(f"Train windows (overlap): "
    f"{len(train_starts)}")
print(f"Test  windows (overlap): "
    f"{len(test_starts)}")
def early_warning_label(win,tau=tau,):
    ot_future = win[m:, 6]
    return int(ot_future.max() > tau)
def build_arrays(starts,):
    n = len(starts)
    x_obs = np.empty((n,m,len(COLS)),dtype=np.float32)
    x_fut = np.empty((n,k,len(COLS)),dtype=np.float32)
    y = np.empty((n,),dtype=np.int64)
    for idx, i in enumerate(starts):
        w = X[i : i + window]
        x_obs[idx] = w[:m]
        x_fut[idx] = w[m:]
        y[idx] = (early_warning_label(w))
    return (x_obs,x_fut,y)
x_obs_train, x_fut_train, y_train = (build_arrays(train_starts))
x_obs_test, x_fut_test, y_test = (build_arrays(test_starts))
print("\nShapes:")
print(" x_obs_train:",x_obs_train.shape)
print(" x_fut_train:",x_fut_train.shape)
print(" y_train    :",y_train.shape)
print(" x_obs_test :",x_obs_test.shape)
print(" x_fut_test :",x_fut_test.shape)
print(" y_test     :",y_test.shape)
unique_train, counts_train = (np.unique(y_train,return_counts=True))
for cls, cnt in zip(unique_train,counts_train,):
    pct = (cnt / len(y_train) * 100)
    print(f"Train class {cls}: "
        f"{cnt} examples "
        f"({pct:.1f}%)")
unique_test, counts_test = (np.unique(y_test,return_counts=True))
for cls, cnt in zip(unique_test,counts_test):
    pct = (cnt / len(y_test) * 100)
    print(f" Test class {cls}: "
        f"{cnt} examples "
        f"({pct:.1f}%)")
def balance_downsample(x_obs,x_fut,y,random_state=None):
    rng = (np.random.default_rng(random_state))
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n_min = min(len(idx0),len(idx1))
    sel0 = rng.choice(idx0,size=n_min,replace=False)
    sel1 = rng.choice(idx1,size=n_min,replace=False)
    sel = np.concatenate([sel0,sel1])
    rng.shuffle(sel)
    return (x_obs[sel],x_fut[sel],y[sel])
x_obs_train, x_fut_train, y_train = (balance_downsample(x_obs_train,x_fut_train,
        y_train,random_state=42))
print("Balanced training counts:",Counter(y_train))
X_observed_train = (x_obs_train)
X_forecast_train = (x_fut_train)
X_observed_test = (x_obs_test)
X_forecast_test = (x_fut_test)
idx = np.arange(len(X_observed_train))
idx_fit, idx_val = (train_test_split(idx,test_size=0.20,random_state=42,stratify=y_train))
X_observed_fit = (X_observed_train[idx_fit].astype(np.float32))
X_forecast_fit = (X_forecast_train[idx_fit].astype(np.float32))
y_fit = (y_train[idx_fit].astype(int))
X_observed_val = (X_observed_train[idx_val].astype(np.float32))
X_forecast_val = (X_forecast_train[idx_val].astype(np.float32))
y_val = (y_train[idx_val].astype(int))
DATASET_NAME = ("Oil_example_2")
def get_data():
    return {"dataset_name":DATASET_NAME,
        "X_observed_train":X_observed_train,
        "X_forecast_train":X_forecast_train,
        "y_train": y_train,
        "X_observed_test":X_observed_test,
        "X_forecast_test":X_forecast_test,
        "y_test":y_test,
        "X_observed_fit":X_observed_fit,
        "X_forecast_fit":X_forecast_fit,
        "y_fit":y_fit,
        "X_observed_val":X_observed_val,
        "X_forecast_val":X_forecast_val,
        "y_val":y_val,
        "test_starts":test_starts,
        "m": m,"k":k,"window":window,
        "stride":stride,"tau":tau,
        "train_frac":train_frac}
if __name__ == "__main__":
    data = get_data()
    print("\nDataset:",data["dataset_name"])
    print("Train:",data["X_observed_train"].shape,data["X_forecast_train"].shape,data["y_train"].shape)
    print("Test:",data["X_observed_test"].shape,data["X_forecast_test"].shape,data["y_test"].shape)
    print("Training class counts:",np.unique(data["y_train"],return_counts=True))
    print("Test class counts:",np.unique( data["y_test"],return_counts=True))