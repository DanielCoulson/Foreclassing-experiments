import os, random, numpy as np, torch, sys, time, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from tqdm import tqdm
TQDM = True 
import importlib.util
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from sklearn.linear_model import RidgeClassifierCV
import math


MODERNTCN_ROOT = os.path.expanduser("~/ModernTCN") 
sys.path.append(os.path.join(MODERNTCN_ROOT, "ModernTCN-classification")) 
from models.ModernTCN import Model as ModernTCNClassModel  

class _ModernTCNClassConfig: 
    def __init__(self, seq_len, enc_in, num_class): 
        self.task_name = 'classification' 
        self.seq_len = seq_len 
        self.enc_in = enc_in 
        self.num_class = num_class 
        self.pred_len = 1 
        self.stem_ratio = 2 
        self.downsample_ratio = 1 
        self.ffn_ratio = 2 
        self.patch_size = seq_len 
        self.patch_stride = seq_len 
        self.num_blocks = [1, 1] 
        self.large_size = [3, 3] 
        self.small_size = [3, 3]
        self.dims = [64, 64]
        self.dw_dims = [64, 64]
        self.small_kernel_merged = True
        self.use_multi_scale = True
        self.dropout = 0.0
        self.head_dropout = 0.0
        self.class_dropout = 0.0
        self.revin = True
        self.affine = False
        self.subtract_last = False
        self.decomposition = False
        self.kernel_size = 3
        self.individual = False
        self.freq = 'h'
class ModernTCNClassifier:
    def __init__(self, max_epochs=50, batch_size=256, lr=1e-3, weight_decay=1e-4, device=None, verbose=True):
        if device is not None:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.model_ = None
        self.config_ = None
        self.classes_ = None
        self.is_fitted_ = False
        if verbose:
            print(f"[ModernTCN] Using device: {self.device}")
            
    def _build(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        _, L, D = X.shape
        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.config_ = _ModernTCNClassConfig(seq_len=L, enc_in=D, num_class=len(classes))
        self.model_ = ModernTCNClassModel(self.config_).to(self.device)
        return y_idx
    
    def _loader(self, X, y_idx=None, shuffle=False):
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if y_idx is None:
            ds = TensorDataset(X_t)
        else:
            ds = TensorDataset(X_t, torch.from_numpy(np.asarray(y_idx, dtype=np.int64)))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X, y):
        y_idx = self._build(X, y)
        loader = self._loader(X, y_idx=y_idx, shuffle=True)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        self.model_.train()
        for epoch in range(self.max_epochs):
            total, n = 0.0, 0
            it = tqdm(loader, desc=f"[ModernTCN] Epoch {epoch+1}/{self.max_epochs}", leave=True) if TQDM else loader
            for batch in it:
                xb, yb = batch
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model_(xb, None, None, None)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                bs = xb.size(0)
                total += loss.item() * bs
                n += bs
            if self.verbose and not TQDM:
                print(f"[ModernTCN] Epoch {epoch+1}/{self.max_epochs} loss={total/max(1,n):.4f}")
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted.")
        loader = self._loader(X, shuffle=False)
        self.model_.eval()
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model_(xb, None, None, None)
                outs.append(F.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
MODERNTCN_ROOT = os.path.expanduser("~/ModernTCN")
sys.path.append(os.path.join(MODERNTCN_ROOT, "ModernTCN-classification"))
from models.ModernTCN import Model as ModernTCNClassModel  # noqa: E402
import importlib.util
st_model_path = os.path.join(
    MODERNTCN_ROOT,
    "ModernTCN-short-term",
    "models",
    "ModernTCN.py",
)
spec = importlib.util.spec_from_file_location("moderntcn_st", st_model_path)
moderntcn_st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moderntcn_st)
ModernTCNForecastModel = moderntcn_st.Model  

class _ModernTCNClassConfig:
    def __init__(self, seq_len, enc_in, num_class):
        self.task_name = 'classification'
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_class = num_class
        self.pred_len = 1
        self.stem_ratio = 2
        self.downsample_ratio = 1
        self.ffn_ratio = 2
        self.patch_size = seq_len
        self.patch_stride = seq_len
        self.num_blocks = [1, 1]
        self.large_size = [3, 3]
        self.small_size = [3, 3]
        self.dims = [64, 64]
        self.dw_dims = [64, 64]
        self.small_kernel_merged = True
        self.use_multi_scale = True
        self.dropout = 0.0
        self.head_dropout = 0.0
        self.class_dropout = 0.0
        self.revin = True
        self.affine = False
        self.subtract_last = False
        self.decomposition = False
        self.kernel_size = 3
        self.individual = False
        self.freq = 'h'

class _ModernTCNForecastConfig:
    def __init__(self, seq_len, pred_len, enc_in):
        self.task_name = 'short_term_forecast'
        self.seq_len = seq_len
        self.label_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = enc_in

        self.stem_ratio = 6
        self.downsample_ratio = 2
        self.ffn_ratio = 2

        self.patch_size = min(16, seq_len)
        self.patch_stride = min(8, self.patch_size)

        self.num_blocks = [1, 1]          
        self.large_size = [13, 13]
        self.small_size = [5, 5]
        self.dims = [64, 64]
        self.dw_dims = [64, 64]

        self.small_kernel_merged = False
        self.use_multi_scale = False

        self.dropout = 0.0
        self.head_dropout = 0.0
        self.class_dropout = 0.0

        self.revin = True
        self.affine = False
        self.subtract_last = False

        self.decomposition = False
        self.kernel_size = 25
        self.individual = False
        self.freq = 'h'

class ModernTCNClassifier:
    def __init__(self, max_epochs=50, batch_size=256, lr=1e-3, weight_decay=1e-4, device=None, verbose=True):
        if device is not None:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.model_ = None
        self.config_ = None
        self.classes_ = None
        self.is_fitted_ = False
        if verbose:
            print(f"[ModernTCN] Using device: {self.device}")

    def _build(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        _, L, D = X.shape
        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.config_ = _ModernTCNClassConfig(seq_len=L, enc_in=D, num_class=len(classes))
        self.model_ = ModernTCNClassModel(self.config_).to(self.device)
        return y_idx

    def _loader(self, X, y_idx=None, shuffle=False):
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if y_idx is None:
            ds = TensorDataset(X_t)
        else:
            ds = TensorDataset(X_t, torch.from_numpy(np.asarray(y_idx, dtype=np.int64)))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X, y):
        y_idx = self._build(X, y)
        loader = self._loader(X, y_idx=y_idx, shuffle=True)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        self.model_.train()
        for epoch in range(self.max_epochs):
            total, n = 0.0, 0
            it = tqdm(loader, desc=f"[ModernTCN] Epoch {epoch+1}/{self.max_epochs}", leave=True) if TQDM else loader
            for batch in it:
                xb, yb = batch
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model_(xb, None, None, None)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                bs = xb.size(0)
                total += loss.item() * bs
                n += bs
            if self.verbose and not TQDM:
                print(f"[ModernTCN] Epoch {epoch+1}/{self.max_epochs} loss={total/max(1,n):.4f}")
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted.")
        loader = self._loader(X, shuffle=False)
        self.model_.eval()
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model_(xb, None, None, None)
                outs.append(F.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class ModernTCNForecastThenClassifyOfficial:
    def __init__(
        self,
        X_forecast_train,          
        max_epochs_forecast=20,
        max_epochs_class=50,
        batch_size=256,
        lr_forecast=1e-3,
        lr_class=1e-3,
        weight_decay=1e-4,
        device=None,
        verbose=True,
    ):
        X_forecast_train = np.asarray(X_forecast_train, dtype=np.float32)
        if X_forecast_train.ndim != 3:
            raise ValueError(f"X_forecast_train must be (N, L_fut, D), got {X_forecast_train.shape}")
        self.X_fut_train = X_forecast_train
        self.L_fut = X_forecast_train.shape[1]
        self.D = X_forecast_train.shape[2]

        if device is not None:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self.max_epochs_forecast = max_epochs_forecast
        self.max_epochs_class = max_epochs_class
        self.batch_size = batch_size
        self.lr_forecast = lr_forecast
        self.lr_class = lr_class
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.forecaster_ = None
        self.classifier_ = None
        self.classes_ = None
        self.is_fitted_ = False

        if verbose:
            print(f"[TwoStage-ModernTCN] Using device: {self.device}")

    def _loader_forecast(self, X_obs, X_fut, shuffle):
        ds = TensorDataset(torch.from_numpy(np.asarray(X_obs, dtype=np.float32)),
                           torch.from_numpy(np.asarray(X_fut, dtype=np.float32)))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _loader_plain(self, X, shuffle=False):
        ds = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _build_forecaster(self, L_obs, D):
        cfg = _ModernTCNForecastConfig(seq_len=L_obs, pred_len=self.L_fut, enc_in=D)
        return ModernTCNForecastModel(cfg).to(self.device)

    def _train_forecaster(self, X_obs_train):
        X_obs_train = np.asarray(X_obs_train, dtype=np.float32)
        N, L_obs, D = X_obs_train.shape
        if D != self.D:
            raise ValueError(f"Feature dim mismatch: obs D={D}, fut D={self.D}")
        self.forecaster_ = self._build_forecaster(L_obs, D)
        loader = self._loader_forecast(X_obs_train, self.X_fut_train, shuffle=True)

        opt = torch.optim.Adam(self.forecaster_.parameters(), lr=self.lr_forecast, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        self.forecaster_.train()
        for epoch in range(self.max_epochs_forecast):
            total, n = 0.0, 0
            it = tqdm(loader, desc=f"[ModernTCN-forecast] Epoch {epoch+1}/{self.max_epochs_forecast}", leave=True) if TQDM else loader
            for xb, yb in it:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                yhat = self.forecaster_(xb, None, None, None)   
                loss = loss_fn(yhat, yb)
                loss.backward()
                opt.step()
                bs = xb.size(0)
                total += loss.item() * bs
                n += bs
            if self.verbose and not TQDM:
                print(f"[ModernTCN-forecast] Epoch {epoch+1}/{self.max_epochs_forecast} loss={total/max(1,n):.4f}")

    @torch.no_grad()
    def _forecast(self, X_obs):
        if self.forecaster_ is None:
            raise RuntimeError("Forecaster not trained. Call fit() first.")
        loader = self._loader_plain(X_obs, shuffle=False)
        self.forecaster_.eval()
        outs = []
        for (xb,) in loader:
            xb = xb.to(self.device)
            yhat = self.forecaster_(xb, None, None, None)
            outs.append(yhat.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def fit(self, X_obs_train, y_train):
        X_obs_train = np.asarray(X_obs_train, dtype=np.float32)
        y_train = np.asarray(y_train)
        self._train_forecaster(X_obs_train)
        fut_pred_train = self._forecast(X_obs_train)      
        X_concat_train = np.concatenate([X_obs_train, fut_pred_train], axis=1)
        self.classifier_ = ModernTCNClassifier(
            max_epochs=self.max_epochs_class,
            batch_size=self.batch_size,
            lr=self.lr_class,
            weight_decay=self.weight_decay,
            device=self.device,
            verbose=self.verbose,
        )
        self.classifier_.fit(X_concat_train, y_train)
        self.classes_ = self.classifier_.classes_
        self.is_fitted_ = True
        return self

    def predict_proba(self, X_obs):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        fut_pred = self._forecast(X_obs)
        X_concat = np.concatenate([np.asarray(X_obs, dtype=np.float32), fut_pred], axis=1)
        return self.classifier_.predict_proba(X_concat)

    def predict(self, X_obs):
        proba = self.predict_proba(X_obs)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


MODERNTCN_ROOT = os.path.expanduser("~/ModernTCN")
sys.path.append(os.path.join(MODERNTCN_ROOT, "ModernTCN-classification"))
from models.ModernTCN import Model as ModernTCNClassModel  
import importlib.util
st_model_path = os.path.join(
    MODERNTCN_ROOT,
    "ModernTCN-short-term",
    "models",
    "ModernTCN.py",
)
spec = importlib.util.spec_from_file_location("moderntcn_st", st_model_path)
moderntcn_st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moderntcn_st)
ModernTCNForecastModel = moderntcn_st.Model  
def to_int(y):
    y = np.asarray(y)
    return np.argmax(y, axis=1) if (y.ndim == 2 and y.shape[1] > 1) else y.astype(int)

class _ModernTCNClassConfigBayes:
    def __init__(self, seq_len, enc_in, num_class):
        self.task_name = 'classification'
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_class = num_class
        self.pred_len = 1
        self.stem_ratio = 2
        self.downsample_ratio = 1
        self.ffn_ratio = 2
        self.patch_size = seq_len
        self.patch_stride = seq_len
        self.num_blocks = [1, 1]
        self.large_size = [3, 3]
        self.small_size = [3, 3]
        self.dims = [64, 64]
        self.dw_dims = [64, 64]
        self.small_kernel_merged = True
        self.use_multi_scale = True
        self.dropout = 0.1
        self.head_dropout = 0.0
        self.class_dropout = 0.1
        self.revin = True
        self.affine = False
        self.subtract_last = False
        self.decomposition = False
        self.kernel_size = 3
        self.individual = False
        self.freq = 'h'

class _ModernTCNForecastConfigBayes:
    def __init__(self, seq_len, pred_len, enc_in):
        self.task_name = 'short_term_forecast'
        self.seq_len = seq_len
        self.label_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = enc_in
        self.stem_ratio = 6
        self.downsample_ratio = 2
        self.ffn_ratio = 2
        self.patch_size = min(16, seq_len)
        self.patch_stride = min(8, self.patch_size)
        self.num_blocks = [1, 1]         
        self.large_size = [13, 13]
        self.small_size = [5, 5]
        self.dims = [64, 64]
        self.dw_dims = [64, 64]
        self.small_kernel_merged = False
        self.use_multi_scale = False
        self.dropout = 0.1
        self.head_dropout = 0.0
        self.class_dropout = 0.0
        self.revin = True
        self.affine = False
        self.subtract_last = False
        self.decomposition = False
        self.kernel_size = 25
        self.individual = False
        self.freq = 'h'

class _MCDropout(nn.Module):
    def __init__(self, p: float = 0.1, is_mc_dropout: bool = True):
        super().__init__()
        self.p = float(p)
        self.is_mc_dropout = is_mc_dropout
    def forward(self, x):
        if self.training or self.is_mc_dropout:
            return F.dropout(x, p=self.p, training=True)
        return x

class _MCDropoutWrappedConv1d(nn.Module):
    def __init__(self, conv: nn.Conv1d, p: float = 0.1, is_mc_dropout: bool = True):
        super().__init__()
        self.conv = conv
        self.mc_dropout = _MCDropout(p=p, is_mc_dropout=is_mc_dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.mc_dropout(x)
        return x

def concrete_reg_loss(module: nn.Module):
    for p in module.parameters():
        return torch.zeros((), device=p.device)
    return torch.tensor(0.0)

class _MCDropoutWrappedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, p: float = 0.1, is_mc_dropout: bool = True):
        super().__init__()
        self.linear = linear
        self.mc_dropout = _MCDropout(p=p, is_mc_dropout=is_mc_dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.mc_dropout(x)
        return x

def _infer_total_padding_1d(conv: nn.Conv1d, dummy_L: int = 256) -> int:
    device_ = conv.weight.device
    dtype_ = conv.weight.dtype
    with torch.no_grad():
        x = torch.zeros(1, conv.in_channels, dummy_L, device=device_, dtype=dtype_)
        y = conv(x)
    L_out = int(y.size(-1))
    K = conv.kernel_size[0]
    S = conv.stride[0]
    D_ = conv.dilation[0]
    L_in = dummy_L
    P_total = (L_out - 1) * S - L_in + D_ * (K - 1) + 1
    return int(P_total)

class BoltzmannConv1d(nn.Module):
    def __init__(
        self,
        base_conv: nn.Conv1d,
        kernel_sizes,
        temperature: float = 1.0,
    ):
        super().__init__()
        if isinstance(base_conv, BoltzmannConv1d):
            raise ValueError("base_conv is already a BoltzmannConv1d")
        kernel_sizes = list(kernel_sizes)
        assert len(kernel_sizes) >= 1, "kernel_sizes must be non-empty"
        self.in_channels = base_conv.in_channels
        self.out_channels = base_conv.out_channels
        self.stride = base_conv.stride[0]
        self.dilation = base_conv.dilation[0]
        self.groups = base_conv.groups
        self.bias_flag = base_conv.bias is not None
        self.temperature = float(temperature)
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.orig_kernel_size = base_conv.kernel_size[0]
        self.orig_total_pad = _infer_total_padding_1d(base_conv)
        logits = torch.full((self.num_kernels,), -2.0)  
        if self.orig_kernel_size in self.kernel_sizes:
            idx = self.kernel_sizes.index(self.orig_kernel_size)
            logits[idx] = 0.0   
        self.logits = nn.Parameter(logits)
        self.convs = nn.ModuleList()
        with torch.no_grad():
            w_src = base_conv.weight.data.clone()
            b_src = base_conv.bias.data.clone() if self.bias_flag else None
        for k in self.kernel_sizes:
            conv_k = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=k,
                stride=self.stride,
                padding=0,  
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias_flag,
            )
            with torch.no_grad():
                w_dst = conv_k.weight.data
                w_dst.zero_()
                K0 = w_src.size(-1)
                if k >= K0:
                    start = k - K0
                    w_dst[..., start:start + K0] = w_src
                else:
                    w_dst.copy_(w_src[..., K0 - k:])
                if self.bias_flag and b_src is not None:
                    conv_k.bias.data.copy_(b_src)
            self.convs.append(conv_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pi = F.softmax(self.logits / self.temperature, dim=0)  
        N, C, L_in = x.shape
        K0 = self.orig_kernel_size
        S = self.stride
        D_ = self.dilation
        P0 = self.orig_total_pad
        L_out_target = (L_in + P0 - D_ * (K0 - 1) - 1) // S + 1
        if L_out_target <= 0:
            return self.convs[0](x)
        outs = []
        for pi_j, k, conv_k in zip(pi, self.kernel_sizes, self.convs):
            P_left_j = (L_out_target - 1) * S - L_in + D_ * (k - 1) + 1
            P_left_j = max(0, int(P_left_j))
            if P_left_j > 0:
                x_pad = F.pad(x, (P_left_j, 0))   
            else:
                x_pad = x
            y_j = conv_k(x_pad)                  
            Lj = y_j.size(-1)
            if Lj > L_out_target:
                y_j = y_j[..., Lj - L_out_target:]
            elif Lj < L_out_target:
                diff = L_out_target - Lj
                y_j = F.pad(y_j, (diff, 0))
            outs.append(pi_j * y_j)
        return torch.stack(outs, dim=0).sum(dim=0)
    
def make_moderntcn_bayesian(model: nn.Module,
                            N_train: int,
                            l: float = 1e-2,
                            tau: float = 1.0,
                            is_classification: bool = True,
                            is_mc_dropout: bool = True,
                            p_conv: float = 0.1,
                            p_linear: float = 0.1):
    def _wrap(m: nn.Module):
        for name, child in list(m.named_children()):
            _wrap(child)
            if isinstance(child, (nn.Conv1d, BoltzmannConv1d)):
                setattr(
                    m,
                    name,
                    _MCDropoutWrappedConv1d(child, p=p_conv,
                                            is_mc_dropout=is_mc_dropout),
                )
            elif isinstance(child, nn.Linear):
                setattr(
                    m,
                    name,
                    _MCDropoutWrappedLinear(child, p=p_linear,
                                            is_mc_dropout=is_mc_dropout),
                )

    _wrap(model)
    return model
def set_mc_dropout(module: nn.Module, enabled: bool = True):
    for m in module.modules():
        if isinstance(m, _MCDropout):
            m.is_mc_dropout = enabled
def make_moderntcn_boltzmann(
    model: nn.Module,
    kernel_size_map=None,
    temperature: float = 1.0,
):
    if kernel_size_map is None:
        kernel_size_map = {
            3:  [1, 3, 5],
            5:  [3, 5, 7],
            13: [11, 13, 17],
            31: [29, 31, 37],
        }
    def _wrap(m: nn.Module):
        for name, child in list(m.named_children()):
            _wrap(child)
            if isinstance(child, nn.Conv1d):
                k = int(child.kernel_size[0])
                if k > 1 and child.groups == child.in_channels:
                    if k in kernel_size_map:
                        ks = kernel_size_map[k]
                    else:
                        ks = [k]  
                    bc = BoltzmannConv1d(
                        base_conv=child,
                        kernel_sizes=ks,
                        temperature=temperature,
                    )
                    setattr(m, name, bc)
    _wrap(model)
    return model
class BayesianModernTCNClassifier:
    def __init__(
        self,
        max_epochs=50,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        device=None,
        verbose=True,
        l=1e-2,
        tau=1.0,
    ):
        if device is not None:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        if verbose:
            print(f"[BayesianModernTCN] Using device: {self.device}")
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.l = l
        self.tau = tau
        self.model_ = None
        self.config_ = None
        self.classes_ = None
        self.is_fitted_ = False

    def _build_model_and_labels(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (N, L, D), got {X.shape}")
        N, L, D = X.shape
        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes_ = classes
        cfg = _ModernTCNClassConfigBayes(seq_len=L, enc_in=D, num_class=len(classes))
        self.config_ = cfg
        base_model = ModernTCNClassModel(cfg)
        make_moderntcn_boltzmann(base_model)
        make_moderntcn_bayesian(
            base_model,
            N_train=N,
            l=self.l,
            tau=self.tau,
            is_classification=True,
            is_mc_dropout=True,
        )
        self.model_ = base_model.to(self.device)
        return self.model_, y_idx

    def _make_loader(self, X, y_idx=None, shuffle=False):
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X)
        if y_idx is not None:
            y_t = torch.from_numpy(np.asarray(y_idx, dtype=np.int64))
            ds = TensorDataset(X_t, y_t)
        else:
            ds = TensorDataset(X_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _check_is_fitted(self):
        if not self.is_fitted_ or self.model_ is None or self.classes_ is None:
            raise RuntimeError("BayesianModernTCNClassifier is not fitted yet.")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.model_, y_idx = self._build_model_and_labels(X, y)
        loader = self._make_loader(X, y_idx=y_idx, shuffle=True)
        optim = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        self.model_.train()
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            n_samples = 0
            it = tqdm(loader, desc=f"[BayesianModernTCN] Epoch {epoch+1}/{self.max_epochs}", leave=True) if TQDM else loader
            for xb, yb in it:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optim.zero_grad()
                logits = self.model_(xb, None, None, None)
                loss = criterion(logits, yb)
                loss = loss + concrete_reg_loss(self.model_)  
                loss.backward()
                optim.step()
                bs = xb.size(0)
                total_loss += loss.item() * bs
                n_samples += bs
            if self.verbose and not TQDM:
                avg_loss = total_loss / max(1, n_samples)
                print(f"[BayesianModernTCN] Epoch {epoch+1}/{self.max_epochs} "
                      f"loss={avg_loss:.4f}")
        self.is_fitted_ = True
        return self
    def predict_proba(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (N, L, D), got {X.shape}")
        self.model_.eval()
        set_mc_dropout(self.model_, False)  
        loader = self._make_loader(X, shuffle=False)
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model_(xb, None, None, None)
                p = F.softmax(logits, dim=-1)
                probs.append(p.cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict_proba_bayesian(self, X, n_samples=100):
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (N, L, D), got {X.shape}")
        base_loader = self._make_loader(X, shuffle=False)
        self.model_.eval()
        set_mc_dropout(self.model_, True)
        all_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                probs = []
                for (xb,) in base_loader:
                    xb = xb.to(self.device)
                    logits = self.model_(xb, None, None, None)
                    p = F.softmax(logits, dim=-1)
                    probs.append(p.cpu().numpy())
                probs = np.concatenate(probs, axis=0)
                all_samples.append(probs)
        set_mc_dropout(self.model_, False)
        all_samples = np.stack(all_samples, axis=0)  
        mean = all_samples.mean(axis=0)
        std = all_samples.std(axis=0)
        return mean, std

    def predict(self, X, bayesian=False, n_samples=100):
        if not bayesian:
            proba = self.predict_proba(X)
            idx = np.argmax(proba, axis=1)
            return self.classes_[idx]
        mean, _ = self.predict_proba_bayesian(X, n_samples=n_samples)
        idx = np.argmax(mean, axis=1)
        return self.classes_[idx]

class BayesianModernTCNForecastThenClassifyJoint:
    def __init__(
        self,
        X_forecast_train,          
        max_epochs_forecast=20,    
        max_epochs_class=50,
        batch_size=256,
        lr_forecast=1e-3,
        lr_class=1e-3,
        weight_decay=1e-4,
        device=None,
        verbose=True,
        l=1e-2,
        tau=1.0,
        lambda_forecast=1.0,
        lambda_class=1.0,
    ):
        X_forecast_train = np.asarray(X_forecast_train, dtype=np.float32)
        if X_forecast_train.ndim != 3:
            raise ValueError(
                f"X_forecast_train must be (N, L_fut, D), got {X_forecast_train.shape}"
            )

        self.X_fut_train = X_forecast_train
        self.L_fut = X_forecast_train.shape[1]
        self.D = X_forecast_train.shape[2]
        if device is not None:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        if verbose:
            print(f"[BayesianTwoStage-ModernTCN-Joint] Using device: {self.device}")

        self.max_epochs_forecast = max_epochs_forecast
        self.max_epochs_class = max_epochs_class
        self.batch_size = batch_size
        self.lr_forecast = lr_forecast
        self.lr_class = lr_class
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.l = l
        self.tau = tau
        self.lambda_forecast = float(lambda_forecast)
        self.lambda_class = float(lambda_class)
        self.forecaster_ = None
        self.classifier_ = None
        self.classes_ = None
        self.is_fitted_ = False

    def _make_loader_joint(self, X_obs, X_fut, y_idx, shuffle):
        X_obs = np.asarray(X_obs, dtype=np.float32)
        X_fut = np.asarray(X_fut, dtype=np.float32)
        y_idx = np.asarray(y_idx, dtype=np.int64)
        if not (len(X_obs) == len(X_fut) == len(y_idx)):
            raise ValueError("X_obs, X_fut, y_idx must have the same length.")
        ds = TensorDataset(
            torch.from_numpy(X_obs),
            torch.from_numpy(X_fut),
            torch.from_numpy(y_idx),
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _make_loader_no_y(self, X, shuffle=False):
        X = np.asarray(X, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _build_forecaster(self, L_obs, D):
        cfg_f = _ModernTCNForecastConfigBayes(
            seq_len=L_obs,
            pred_len=self.L_fut,
            enc_in=D,
        )
        model_f = ModernTCNForecastModel(cfg_f)
        make_moderntcn_boltzmann(model_f)
        make_moderntcn_bayesian(
            model_f,
            N_train=self.X_fut_train.shape[0],
            l=self.l,
            tau=self.tau,
            is_classification=False,
            is_mc_dropout=False,  
        )
        return model_f.to(self.device)
    
    def _build_classifier(self, L_obs_plus_fut, D, num_class):
        cfg_c = _ModernTCNClassConfigBayes(
            seq_len=L_obs_plus_fut,
            enc_in=D,
            num_class=num_class,
        )
        model_c = ModernTCNClassModel(cfg_c)
        make_moderntcn_boltzmann(model_c)
        make_moderntcn_bayesian(
            model_c,
            N_train=self.X_fut_train.shape[0],
            l=self.l,
            tau=self.tau,
            is_classification=True,
            is_mc_dropout=True,
        )
        return model_c.to(self.device), cfg_c

    def _check(self):
        if not self.is_fitted_ or self.classifier_ is None or self.forecaster_ is None:
            raise RuntimeError("Model not fitted yet.")

    def _forecast(self, X_obs):
        if self.forecaster_ is None:
            raise RuntimeError("Forecaster not trained. Call fit() first.")
        X_obs = np.asarray(X_obs, dtype=np.float32)
        if X_obs.ndim != 3:
            raise ValueError(f"Expected X_obs (N, L_obs, D), got {X_obs.shape}")
        self.forecaster_.eval()
        set_mc_dropout(self.forecaster_, False)
        loader = self._make_loader_no_y(X_obs, shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                yhat = self.forecaster_(xb, None, None, None)
                preds.append(yhat.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def fit(self, X_obs_train, y_train):
        X_obs_train = np.asarray(X_obs_train, dtype=np.float32)
        y_train_arr = to_int(y_train)  
        y_train_arr = np.asarray(y_train_arr, dtype=np.int64)
        if X_obs_train.ndim != 3:
            raise ValueError(
                f"Expected X_obs_train (N, L_obs, D), got {X_obs_train.shape}"
            )
        N, L_obs, D = X_obs_train.shape
        if D != self.D:
            raise ValueError(f"Feature dim mismatch: obs D={D}, fut D={self.D}")
        if len(y_train_arr) != N:
            raise ValueError("X_obs_train and y_train must have the same length.")
        classes, y_idx = np.unique(y_train_arr, return_inverse=True)
        self.classes_ = classes
        self.forecaster_ = self._build_forecaster(L_obs, D)
        L_concat = L_obs + self.L_fut
        self.classifier_, self.config_ = self._build_classifier(
            L_concat, D, num_class=len(classes)
        )
        loader = self._make_loader_joint(
            X_obs_train, self.X_fut_train, y_idx, shuffle=True
        )
        optim_f = torch.optim.Adam(
            self.forecaster_.parameters(),
            lr=self.lr_forecast,
            weight_decay=self.weight_decay,
        )
        optim_c = torch.optim.Adam(
            self.classifier_.parameters(),
            lr=self.lr_class,
            weight_decay=self.weight_decay,
        )
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        self.forecaster_.train()
        self.classifier_.train()
        num_epochs = self.max_epochs_class  
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_loss_f = 0.0
            total_loss_c = 0.0
            n_samples = 0
            it = tqdm(
                loader,
                desc=f"[BayesianModernTCN-Joint] Epoch {epoch+1}/{num_epochs}",
                leave=True,
            ) if TQDM else loader
            for xb_obs, xb_fut, yb in it:
                xb_obs = xb_obs.to(self.device)
                xb_fut = xb_fut.to(self.device)
                yb = yb.to(self.device)
                optim_f.zero_grad()
                optim_c.zero_grad()
                fut_pred = self.forecaster_(xb_obs, None, None, None)  
                x_concat = torch.cat([xb_obs, fut_pred], dim=1)        
                logits = self.classifier_(x_concat, None, None, None)  
                lf = mse_loss(fut_pred, xb_fut)
                lc = ce_loss(logits, yb)
                reg_f = concrete_reg_loss(self.forecaster_)
                reg_c = concrete_reg_loss(self.classifier_)
                loss = (
                    self.lambda_forecast * lf
                    + self.lambda_class * lc
                    + reg_f
                    + reg_c
                )
                loss.backward()
                optim_f.step()
                optim_c.step()
                bs = xb_obs.size(0)
                total_loss += loss.item() * bs
                total_loss_f += lf.item() * bs
                total_loss_c += lc.item() * bs
                n_samples += bs
            if self.verbose and not TQDM:
                avg_loss = total_loss / max(1, n_samples)
                avg_lf = total_loss_f / max(1, n_samples)
                avg_lc = total_loss_c / max(1, n_samples)
                print(
                    f"[BayesianModernTCN-Joint] Epoch {epoch+1}/{num_epochs} "
                    f"loss={avg_loss:.4f} lf={avg_lf:.44f} lc={avg_lc:.4f}"
                )
        self.is_fitted_ = True
        return self

    def predict_proba(self, X_obs):
        self._check()
        X_obs = np.asarray(X_obs, dtype=np.float32)
        if X_obs.ndim != 3:
            raise ValueError(f"Expected X_obs (N, L_obs, D), got {X_obs.shape}")
        fut_pred = self._forecast(X_obs)
        X_concat = np.concatenate([X_obs, fut_pred], axis=1)
        self.forecaster_.eval()
        self.classifier_.eval()
        set_mc_dropout(self.forecaster_, False)
        set_mc_dropout(self.classifier_, False)
        X_concat_t = torch.from_numpy(X_concat.astype(np.float32))
        loader = DataLoader(
            TensorDataset(X_concat_t),
            batch_size=self.batch_size,
            shuffle=False,
        )
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.classifier_(xb, None, None, None)
                p = F.softmax(logits, dim=-1)
                probs.append(p.cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict_proba_bayesian(self, X_obs, n_samples=100):
        self._check()
        X_obs = np.asarray(X_obs, dtype=np.float32)
        if X_obs.ndim != 3:
            raise ValueError(f"Expected X_obs (N, L_obs, D), got {X_obs.shape}")
        loader_obs = self._make_loader_no_y(X_obs, shuffle=False)
        self.forecaster_.eval()
        self.classifier_.eval()
        set_mc_dropout(self.forecaster_, True)
        set_mc_dropout(self.classifier_, True)
        all_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                fut_chunks = []
                for (xb,) in loader_obs:
                    xb = xb.to(self.device)
                    yhat = self.forecaster_(xb, None, None, None)
                    fut_chunks.append(yhat.cpu().numpy())
                fut_pred = np.concatenate(fut_chunks, axis=0)
                X_concat = np.concatenate([X_obs, fut_pred], axis=1)
                X_concat_t = torch.from_numpy(X_concat.astype(np.float32))
                loader_concat = DataLoader(
                    TensorDataset(X_concat_t),
                    batch_size=self.batch_size,
                    shuffle=False,
                )
                probs = []
                for (xb,) in loader_concat:
                    xb = xb.to(self.device)
                    logits = self.classifier_(xb, None, None, None)
                    p = F.softmax(logits, dim=-1)
                    probs.append(p.cpu().numpy())
                probs = np.concatenate(probs, axis=0)
                all_samples.append(probs)
        set_mc_dropout(self.forecaster_, False)
        set_mc_dropout(self.classifier_, False)
        all_samples = np.stack(all_samples, axis=0)  
        mean = all_samples.mean(axis=0)
        std = all_samples.std(axis=0)
        return mean, std

    def predict(self, X_obs, bayesian=True, n_samples=100):
        if not bayesian:
            proba = self.predict_proba(X_obs)
            idx = np.argmax(proba, axis=1)
            return self.classes_[idx]
        mean, _ = self.predict_proba_bayesian(X_obs, n_samples=n_samples)
        idx = np.argmax(mean, axis=1)
        return self.classes_[idx]
    
import math
class RocketMultivariateLengths:
    def __init__(self, n_kernels=10000, lengths=(2, 3, 4), random_state=None):
        self.n_kernels = int(n_kernels)
        self.lengths = tuple(int(l) for l in lengths)
        self.random_state = random_state
    def _check_X(self, X):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_samples, series_length, n_channels), "
                f"got shape {X.shape}"
            )
        return X
    def fit(self, X, y=None):
        X = self._check_X(X)
        n_samples, series_length, n_channels = X.shape
        self.series_length_ = series_length
        self.n_channels_ = n_channels
        rng = np.random.RandomState(self.random_state)
        self.kernel_lengths_ = np.empty(self.n_kernels, dtype=np.int32)
        self.kernel_dilations_ = np.empty(self.n_kernels, dtype=np.int32)
        self.kernel_paddings_ = np.empty(self.n_kernels, dtype=np.int32)
        self.kernel_biases_ = np.empty(self.n_kernels, dtype=np.float32)
        self.kernel_channels_ = []
        self.kernel_weights_ = []
        lengths = np.array(self.lengths, dtype=np.int32)
        if np.any(lengths < 2):
            raise ValueError("Kernel lengths must be >= 2")
        if np.any(lengths > series_length):
            raise ValueError("All kernel lengths must be <= series length")
        for k in range(self.n_kernels):
            L = rng.choice(lengths)
            self.kernel_lengths_[k] = L
            n_chosen = rng.randint(1, n_channels + 1)
            channels = rng.choice(n_channels, size=n_chosen, replace=False)
            self.kernel_channels_.append(channels.astype(np.int32))
            W = rng.normal(loc=0.0, scale=1.0, size=(n_chosen, L)).astype(np.float32)
            W -= W.mean()
            self.kernel_weights_.append(W)
            b = rng.uniform(-1.0, 1.0)
            self.kernel_biases_[k] = b
            if L == 1:
                d = 1
            else:
                A = math.log2((series_length - 1) / (L - 1))
                if A <= 0:
                    d = 1
                else:
                    x = rng.uniform(0.0, A)
                    d = int(2 ** x)
                    if d < 1:
                        d = 1
            self.kernel_dilations_[k] = d
            use_padding = rng.rand() < 0.5
            if use_padding:
                pad = int(((L - 1) * d) / 2)
            else:
                pad = 0
            self.kernel_paddings_[k] = pad
        return self

    def transform(self, X):
        X = self._check_X(X)
        n_samples, series_length, n_channels = X.shape
        if series_length != self.series_length_ or n_channels != self.n_channels_:
            raise ValueError("X has different shape from fit data")
        features = np.empty((n_samples, 2 * self.n_kernels), dtype=np.float32)
        for k in range(self.n_kernels):
            L = int(self.kernel_lengths_[k])
            d = int(self.kernel_dilations_[k])
            pad = int(self.kernel_paddings_[k])
            b = float(self.kernel_biases_[k])
            channels = self.kernel_channels_[k]
            W = self.kernel_weights_[k]  
            Xc = X[:, :, channels]  
            if pad > 0:
                Xp = np.pad(Xc, ((0, 0), (pad, pad), (0, 0)), mode="constant")
            else:
                Xp = Xc
            T_pad = Xp.shape[1]
            effective_len = (L - 1) * d + 1
            n_positions = T_pad - effective_len + 1
            if n_positions <= 0:
                max_vals = np.zeros(n_samples, dtype=np.float32)
                ppv = np.zeros(n_samples, dtype=np.float32)
            else:
                max_vals = np.full(n_samples, -np.inf, dtype=np.float32)
                pos_counts = np.zeros(n_samples, dtype=np.int32)
                w_flat = W.reshape(-1)
                for s in range(n_positions):
                    segment = Xp[:, s : s + effective_len : d, :]  
                    seg_flat = segment.reshape(n_samples, -1)
                    conv = seg_flat @ w_flat + b 
                    max_vals = np.maximum(max_vals, conv)
                    pos_counts += (conv > 0)
                ppv = pos_counts.astype(np.float32) / float(n_positions)
            features[:, 2 * k] = max_vals
            features[:, 2 * k + 1] = ppv
        return features