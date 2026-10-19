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
from models.ModernTCN import Model as ModernTCNClassModel  
import importlib.util
st_model_path = os.path.join( MODERNTCN_ROOT,"ModernTCN-short-term",
    "models","ModernTCN.py")
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
    def __init__(self,X_forecast_train, max_epochs_forecast=20,max_epochs_class=50,
        batch_size=256,lr_forecast=1e-3,lr_class=1e-3,weight_decay=1e-4,device=None,verbose=True):
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
        self.classifier_ = ModernTCNClassifier(max_epochs=self.max_epochs_class,
            batch_size=self.batch_size,lr=self.lr_class,weight_decay=self.weight_decay,
            device=self.device,verbose=self.verbose)
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
st_model_path = os.path.join(MODERNTCN_ROOT,"ModernTCN-short-term","models","ModernTCN.py")
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
    def __init__(self,base_conv: nn.Conv1d,kernel_sizes,temperature: float = 1.0):
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
            conv_k = nn.Conv1d(in_channels=self.in_channels,
                out_channels=self.out_channels,kernel_size=k,
                stride=self.stride,padding=0,dilation=self.dilation,
                groups=self.groups,bias=self.bias_flag)
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
class DynamicConv1d(nn.Module):
    def __init__(self,base_conv: nn.Conv1d,kernel_sizes,temperature: float = 1.0,
        reduction: int = 4,min_hidden: int = 8):
        super().__init__()
        if isinstance(base_conv,(BoltzmannConv1d, DynamicConv1d)):
            raise ValueError("base_conv is already a mixture convolution.")
        kernel_sizes = list(kernel_sizes)
        assert len(kernel_sizes) >= 1, ("kernel_sizes must be non-empty")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if reduction < 1:
            raise ValueError("reduction must be >= 1.")
        self.in_channels = base_conv.in_channels
        self.out_channels = base_conv.out_channels
        self.stride = base_conv.stride[0]
        self.dilation = base_conv.dilation[0]
        self.groups = base_conv.groups
        self.bias_flag = (base_conv.bias is not None)
        self.temperature = float(temperature)
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.orig_kernel_size = (base_conv.kernel_size[0])
        self.orig_total_pad = (_infer_total_padding_1d(base_conv))
        self.convs = nn.ModuleList()
        with torch.no_grad():
            w_src = (base_conv.weight.data.clone())
            b_src = (base_conv.bias.data.clone()
                if self.bias_flag
                else None)
        for k in self.kernel_sizes:
            conv_k = nn.Conv1d(in_channels=self.in_channels,out_channels=self.out_channels,
                kernel_size=k,stride=self.stride,padding=0,dilation=self.dilation,
                groups=self.groups,bias=self.bias_flag)
            with torch.no_grad():
                w_dst = conv_k.weight.data
                w_dst.zero_()
                K0 = w_src.size(-1)
                if k >= K0:
                    start = k - K0
                    w_dst[...,start:start + K0] = w_src
                else:
                    w_dst.copy_(w_src[...,K0 - k:])
                if (self.bias_flag and b_src is not None):
                    conv_k.bias.data.copy_(b_src)
            self.convs.append(conv_k)
        hidden_dim = max(int(min_hidden),self.in_channels // int(reduction))
        self.attention_pool = (nn.AdaptiveAvgPool1d(1))
        self.attention_fc1 = nn.Linear(self.in_channels,hidden_dim)
        self.attention_fc2 = nn.Linear(hidden_dim,self.num_kernels)
        with torch.no_grad():
            self.attention_fc2.weight.zero_()
            initial_logits = torch.full((self.num_kernels,),-2.0)
            if (self.orig_kernel_size in self.kernel_sizes):
                idx = self.kernel_sizes.index(self.orig_kernel_size)
                initial_logits[idx] = 0.0
            self.attention_fc2.bias.copy_(initial_logits)
        self.last_attention = None
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.attention_pool(x)
        pooled = pooled.squeeze(-1)
        hidden = F.relu(self.attention_fc1(pooled))
        logits = self.attention_fc2(hidden)
        pi = F.softmax(logits / self.temperature,dim=1)
        return pi
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        pi = self.attention_weights(x)
        self.last_attention = (pi.detach())
        N, C, L_in = x.shape
        K0 = self.orig_kernel_size
        S = self.stride
        D_ = self.dilation
        P0 = self.orig_total_pad
        L_out_target = (L_in+ P0- D_ * (K0 - 1)- 1) // S + 1
        if L_out_target <= 0:
            return self.convs[0](x)
        outs = []
        for k, conv_k in zip(self.kernel_sizes,self.convs):
            P_left_j = ((L_out_target - 1) * S- L_in+ D_ * (k - 1)+ 1)
            P_left_j = max(0,int(P_left_j))
            if P_left_j > 0:
                x_pad = F.pad(x,(P_left_j, 0))
            else:
                x_pad = x
            y_j = conv_k(x_pad)
            Lj = y_j.size(-1)
            if Lj > L_out_target:
                y_j = y_j[...,Lj - L_out_target:]
            elif Lj < L_out_target:
                diff = (L_out_target- Lj)
                y_j = F.pad(y_j,(diff, 0))
            outs.append(y_j)
        expert_outputs = torch.stack(outs,dim=1)
        pi = pi[:,:,None,None]
        output = (expert_outputs* pi).sum(dim=1)
        return output

def make_moderntcn_bayesian(model: nn.Module,N_train: int,l: float = 1e-2,
                            tau: float = 1.0,is_classification: bool = True,
                            is_mc_dropout: bool = True,p_conv: float = 0.1,
                            p_linear: float = 0.1):
    def _wrap(m: nn.Module):
        for name, child in list(m.named_children()):
            _wrap(child)
            if isinstance(child, (nn.Conv1d, BoltzmannConv1d, DynamicConv1d )):
                setattr(m,name, _MCDropoutWrappedConv1d(child, p=p_conv,is_mc_dropout=is_mc_dropout))
            elif isinstance(child, nn.Linear):
                setattr(m,name, _MCDropoutWrappedLinear(child, p=p_linear,is_mc_dropout=is_mc_dropout))
    _wrap(model)
    return model
def set_mc_dropout(module: nn.Module, enabled: bool = True):
    for m in module.modules():
        if isinstance(m, _MCDropout):
            m.is_mc_dropout = enabled
def make_moderntcn_boltzmann(model: nn.Module,kernel_size_map=None,temperature: float = 1.0):
    if kernel_size_map is None:
        kernel_size_map = {3:  [1, 3, 5],
            5:  [3, 5, 7],
            13: [11, 13, 17],
            31: [29, 31, 37]}
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
                    bc = BoltzmannConv1d(base_conv=child,kernel_sizes=ks,temperature=temperature)
                    setattr(m, name, bc)
    _wrap(model)
    return model
def make_moderntcn_dynamic(model: nn.Module,kernel_size_map=None,temperature: float = 1.0,
    reduction: int = 4,min_hidden: int = 8):
    if kernel_size_map is None:
        kernel_size_map = {3:  [1, 3, 5],
            5:  [3, 5, 7],
            13: [11, 13, 17],
            31: [29, 31, 37]}
    def _wrap(m: nn.Module):
        for name, child in list(
            m.named_children()):
            _wrap(child)
            if isinstance(child,nn.Conv1d):
                k = int(child.kernel_size[0])
                if (k > 1 and child.groups== child.in_channels):
                    if k in kernel_size_map:
                        ks = kernel_size_map[k]
                    else:
                        ks = [k]
                    dc = DynamicConv1d(base_conv=child,kernel_sizes=ks,
                        temperature=temperature,reduction=reduction,min_hidden=min_hidden)
                    setattr(m,name,dc)
    _wrap(model)
    return model

class BayesianModernTCNClassifier:
    def __init__(self,max_epochs=50,batch_size=256,lr=1e-3,weight_decay=1e-4,
        device=None,verbose=True,l=1e-2,tau=1.0):
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
        make_moderntcn_bayesian(base_model,N_train=N,l=self.l,tau=self.tau,
            is_classification=True,is_mc_dropout=True)
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
        optim = torch.optim.Adam(self.model_.parameters(),lr=self.lr,weight_decay=self.weight_decay)
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
    def __init__(self,X_forecast_train,max_epochs_forecast=20,max_epochs_class=50,
        batch_size=256,lr_forecast=1e-3,lr_class=1e-3,weight_decay=1e-4,device=None,
        verbose=True,l=1e-2,tau=1.0,lambda_forecast=1.0,lambda_class=1.0):
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
        ds = TensorDataset(torch.from_numpy(X_obs),torch.from_numpy(X_fut),torch.from_numpy(y_idx))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)
    def _make_loader_no_y(self, X, shuffle=False):
        X = np.asarray(X, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)
    def _build_forecaster(self, L_obs, D):
        cfg_f = _ModernTCNForecastConfigBayes(seq_len=L_obs,pred_len=self.L_fut,enc_in=D)
        model_f = ModernTCNForecastModel(cfg_f)
        make_moderntcn_boltzmann(model_f)
        make_moderntcn_bayesian(model_f,N_train=self.X_fut_train.shape[0],l=self.l,tau=self.tau,
            is_classification=False,is_mc_dropout=False)
        return model_f.to(self.device)
    def _build_classifier(self, L_obs_plus_fut, D, num_class):
        cfg_c = _ModernTCNClassConfigBayes(seq_len=L_obs_plus_fut,enc_in=D,num_class=num_class)
        model_c = ModernTCNClassModel(cfg_c)
        make_moderntcn_boltzmann(model_c)
        make_moderntcn_bayesian(model_c,N_train=self.X_fut_train.shape[0],l=self.l,
            tau=self.tau,is_classification=True,is_mc_dropout=True)
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
            raise ValueError(f"Expected X_obs_train (N, L_obs, D), got {X_obs_train.shape}")
        N, L_obs, D = X_obs_train.shape
        if D != self.D:
            raise ValueError(f"Feature dim mismatch: obs D={D}, fut D={self.D}")
        if len(y_train_arr) != N:
            raise ValueError("X_obs_train and y_train must have the same length.")
        classes, y_idx = np.unique(y_train_arr, return_inverse=True)
        self.classes_ = classes
        self.forecaster_ = self._build_forecaster(L_obs, D)
        L_concat = L_obs + self.L_fut
        self.classifier_, self.config_ = self._build_classifier(L_concat, D, num_class=len(classes))
        loader = self._make_loader_joint(X_obs_train, self.X_fut_train, y_idx, shuffle=True)
        optim_f = torch.optim.Adam(self.forecaster_.parameters(),lr=self.lr_forecast,weight_decay=self.weight_decay)
        optim_c = torch.optim.Adam(self.classifier_.parameters(),lr=self.lr_class, weight_decay=self.weight_decay)
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
            it = tqdm(loader,desc=f"[BayesianModernTCN-Joint] Epoch {epoch+1}/{num_epochs}",leave=True) if TQDM else loader
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
                loss = (self.lambda_forecast * lf + self.lambda_class * lc + reg_f + reg_c)
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
                    f"loss={avg_loss:.4f} lf={avg_lf:.44f} lc={avg_lc:.4f}")
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
        loader = DataLoader(TensorDataset(X_concat_t),batch_size=self.batch_size,shuffle=False)
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
                loader_concat = DataLoader(TensorDataset(X_concat_t),batch_size=self.batch_size,shuffle=False)
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
class ForeClassNetDynamic(BayesianModernTCNForecastThenClassifyJoint):
    def __init__(self,*args,dynamic_temperature=1.0,dynamic_reduction=4,
        dynamic_min_hidden=8,dynamic_kernel_size_map=None,**kwargs,):
        super().__init__(*args,**kwargs,)
        self.dynamic_temperature = float(dynamic_temperature)
        self.dynamic_reduction = int(dynamic_reduction)
        self.dynamic_min_hidden = int(dynamic_min_hidden)
        self.dynamic_kernel_size_map = (dynamic_kernel_size_map)
        if self.verbose:
            print("[ForeClassNetDynamic] "
                "Using input-dependent "
                "multi-scale dynamic convolutions.")
            print("[ForeClassNetDynamic] "
                f"temperature="
                f"{self.dynamic_temperature}, "
                f"reduction="
                f"{self.dynamic_reduction}")
    def _build_forecaster(self,L_obs,D,):
        cfg_f = _ModernTCNForecastConfigBayes(seq_len=L_obs,pred_len=self.L_fut,
            enc_in=D)
        model_f = ModernTCNForecastModel(cfg_f)
        make_moderntcn_dynamic(model_f,kernel_size_map=(self.dynamic_kernel_size_map),
            temperature=(self.dynamic_temperature),reduction=(self.dynamic_reduction),
            min_hidden=(self.dynamic_min_hidden))
        make_moderntcn_bayesian(model_f,N_train=(self.X_fut_train.shape[0]),
            l=self.l,tau=self.tau,is_classification=False,is_mc_dropout=False)
        return model_f.to(self.device)
    def _build_classifier(self,L_obs_plus_fut,D,num_class,):
        cfg_c = _ModernTCNClassConfigBayes(seq_len=L_obs_plus_fut,enc_in=D,num_class=num_class)
        model_c = ModernTCNClassModel(cfg_c)
        make_moderntcn_dynamic(model_c,kernel_size_map=(self.dynamic_kernel_size_map),
            temperature=(self.dynamic_temperature),reduction=(self.dynamic_reduction),
            min_hidden=(self.dynamic_min_hidden))
        make_moderntcn_bayesian(model_c,N_train=(self.X_fut_train.shape[0]),l=self.l,
            tau=self.tau,is_classification=True,is_mc_dropout=True)
        return (model_c.to(self.device),cfg_c)
def get_dynamic_layers(model: nn.Module):
    return [module
        for module in model.modules()
        if isinstance(module,DynamicConv1d)]    
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
                f"got shape {X.shape}")
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

class ForeClassNetConstrained(BayesianModernTCNForecastThenClassifyJoint):
    def __init__(self,X_forecast_train,forecast_loss="mse", mc_samples_train=8,
        mc_samples_tau=32,bootstrap_B=2000,bootstrap_level=0.95,dual_lr=1e-2,
        lambda_init=0.0,max_epochs_forecast=20,max_epochs_class=50,batch_size=256,
        lr_forecast=1e-3,lr_class=1e-3,weight_decay=1e-4,device=None,verbose=True,
        l=1e-2,tau=1.0, random_state=0):
        super().__init__(X_forecast_train=X_forecast_train,
            max_epochs_forecast=max_epochs_forecast,max_epochs_class=max_epochs_class,
            batch_size=batch_size,lr_forecast=lr_forecast,lr_class=lr_class,
            weight_decay=weight_decay,device=device,verbose=verbose,l=l,
            tau=tau,lambda_forecast=1.0,lambda_class=1.0)
        forecast_loss = forecast_loss.lower()
        if forecast_loss not in {"mse", "energy"}:
            raise ValueError("forecast_loss must be either 'mse' or 'energy'.")
        if forecast_loss == "energy":
            if mc_samples_train < 2:
                raise ValueError("Energy score needs mc_samples_train >= 2.")
            if mc_samples_tau < 2:
                raise ValueError("Energy score needs mc_samples_tau >= 2.")
        self.forecast_loss_name = forecast_loss
        self.mc_samples_train = int(mc_samples_train)
        self.mc_samples_tau = int(mc_samples_tau)
        self.bootstrap_B = int(bootstrap_B)
        self.bootstrap_level = float(bootstrap_level)
        self.dual_lr = float(dual_lr)
        self.lambda_init = float(lambda_init)
        self.random_state = int(random_state)
        self.forecast_tau_ = None
        self.lambda_dual_ = None
        self.forecast_reference_losses_ = None
        self.training_history_ = []
    def _make_loader_forecast_pair(self,X_obs,X_fut,shuffle=False):
        X_obs = np.asarray(X_obs, dtype=np.float32)
        X_fut = np.asarray(X_fut, dtype=np.float32)
        if len(X_obs) != len(X_fut):
            raise ValueError("X_obs and X_fut must have the same length.")
        ds = TensorDataset(torch.from_numpy(X_obs),torch.from_numpy(X_fut))
        return DataLoader(ds,batch_size=self.batch_size,shuffle=shuffle)
    def _sample_forecasts(self, xb_obs, S):
        samples = []
        for _ in range(S):
            yhat = self.forecaster_(xb_obs,None,None,None)
            samples.append(yhat)
        return torch.stack(samples, dim=0)
    def _forecast_loss_per_example(self,forecast_samples,target):
        S = forecast_samples.size(0)
        if self.forecast_loss_name == "mse":
            sq = (forecast_samples- target.unsqueeze(0)).pow(2)
            per_mc_sample = (sq.flatten(start_dim=2).mean(dim=2))
            return per_mc_sample.mean(dim=0)
        flat_samples = forecast_samples.flatten(start_dim=2)
        flat_target = target.flatten(start_dim=1)
        term1 = torch.linalg.vector_norm(flat_samples - flat_target.unsqueeze(0),
            ord=2,dim=2,).mean(dim=0)
        pair_sum = torch.zeros_like(term1)
        n_pairs = 0
        for s in range(S):
            for t in range(s + 1, S):
                pair_dist = torch.linalg.vector_norm(flat_samples[s] - flat_samples[t],
                    ord=2,dim=1)
                pair_sum = pair_sum + pair_dist
                n_pairs += 1
        if n_pairs == 0:
            raise RuntimeError("Energy score requires at least two MC samples.")
        mean_pair_distance = pair_sum / n_pairs
        term2 = 0.5 * mean_pair_distance
        return term1 - term2
    def _marginal_classification_loss(self,xb_obs,yb,S):
        forecast_samples = []
        log_probs = []
        for _ in range(S):
            fut_pred = self.forecaster_(xb_obs,None,None,None)
            x_concat = torch.cat([xb_obs, fut_pred],dim=1)
            logits = self.classifier_(x_concat,None,None,None)
            forecast_samples.append(fut_pred)
            log_probs.append(F.log_softmax(logits,dim=-1))
        forecast_samples = torch.stack(forecast_samples,dim=0)
        log_probs = torch.stack(log_probs,dim=0)
        log_mean_probs = (torch.logsumexp(log_probs,dim=0)- math.log(S))
        decision_loss = F.nll_loss(log_mean_probs,yb)
        return decision_loss, forecast_samples
    def _pretrain_forecaster(self,X_obs_train):
        loader = self._make_loader_forecast_pair(X_obs_train,self.X_fut_train,
            shuffle=True)
        optim_f = torch.optim.Adam(self.forecaster_.parameters(),lr=self.lr_forecast,
            weight_decay=self.weight_decay)
        self.forecaster_.train()
        set_mc_dropout(self.forecaster_,True)
        for epoch in range(self.max_epochs_forecast):
            total = 0.0
            n = 0
            it = tqdm(loader,desc=(
                    f"[ForeClassNet-"
                    f"{self.forecast_loss_name}-pretrain] "
                    f"Epoch {epoch+1}/"
                    f"{self.max_epochs_forecast}"),
                leave=True) if TQDM else loader
            for xb_obs, xb_fut in it:
                xb_obs = xb_obs.to(self.device)
                xb_fut = xb_fut.to(self.device)
                optim_f.zero_grad()
                samples = self._sample_forecasts(xb_obs,self.mc_samples_train)
                per_example = (self._forecast_loss_per_example(samples,xb_fut))
                loss_f = per_example.mean()
                loss = (loss_f + concrete_reg_loss(self.forecaster_))
                loss.backward()
                optim_f.step()
                bs = xb_obs.size(0)
                total += (loss_f.item() * bs)
                n += bs
            if self.verbose and not TQDM:
                print(f"[ForeClassNet-"
                    f"{self.forecast_loss_name}-pretrain] "
                    f"epoch={epoch+1} "
                    f"forecast_loss="
                    f"{total/max(1,n):.6f}")
        set_mc_dropout(self.forecaster_,False)
    @torch.no_grad()
    def _forecast_losses_on_dataset(self,X_obs,X_fut,S):
        X_obs = np.asarray(X_obs,dtype=np.float32)
        X_fut = np.asarray(X_fut,dtype=np.float32)
        loader = self._make_loader_forecast_pair(X_obs,X_fut,shuffle=False)
        self.forecaster_.eval()
        set_mc_dropout(self.forecaster_,True)
        losses = []
        for xb_obs, xb_fut in loader:
            xb_obs = xb_obs.to(self.device)
            xb_fut = xb_fut.to(self.device)
            samples = self._sample_forecasts(xb_obs,S)
            per_example = (self._forecast_loss_per_example(samples,xb_fut))
            losses.append(per_example.cpu().numpy())
        set_mc_dropout(self.forecaster_,False)
        return np.concatenate(losses)
    def _bootstrap_upper_bound(self,losses):
        losses = np.asarray(losses,dtype=np.float64)
        if losses.ndim != 1:
            raise ValueError("Expected one loss per validation time series.")
        if len(losses) < 2:
            raise ValueError("Need at least two independent validation series.")
        rng = np.random.default_rng(self.random_state)
        n = len(losses)
        boot_means = np.empty(self.bootstrap_B,dtype=np.float64)
        for b in range(self.bootstrap_B):
            indices = rng.integers(low=0,high=n,size=n)
            boot_means[b] = (losses[indices].mean())
        upper_bound = np.quantile(boot_means,self.bootstrap_level)
        return float(upper_bound)
    def fit(self,X_obs_train,y_train,X_obs_val,X_fut_val):
        X_obs_train = np.asarray(X_obs_train,dtype=np.float32)
        y_train_arr = np.asarray(to_int(y_train),dtype=np.int64)
        X_obs_val = np.asarray(X_obs_val,dtype=np.float32)
        X_fut_val = np.asarray(X_fut_val,dtype=np.float32)
        if X_obs_train.ndim != 3:
            raise ValueError("Expected X_obs_train "
                "(N, L_obs, D), got "
                f"{X_obs_train.shape}")
        N, L_obs, D = X_obs_train.shape
        if D != self.D:
            raise ValueError(f"Feature dimension mismatch: "
                f"obs D={D}, future D={self.D}")
        if len(y_train_arr) != N:
            raise ValueError("X_obs_train and y_train "
                "must have the same length.")
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.forecaster_ = (self._build_forecaster(L_obs,D))
        self._pretrain_forecaster(X_obs_train)
        ref_losses = (self._forecast_losses_on_dataset(X_obs_val,X_fut_val,S=self.mc_samples_tau))
        self.forecast_reference_losses_ = (ref_losses)
        self.forecast_tau_ = (self._bootstrap_upper_bound(ref_losses))
        if (not np.isfinite(self.forecast_tau_) or self.forecast_tau_ <= 0):
            raise RuntimeError("Invalid forecast threshold "
                               f"tau={self.forecast_tau_}")
        if self.verbose:
            print("\n"
                f"[ForeClassNet-"
                f"{self.forecast_loss_name}] "
                f"forecast-only validation mean = "
                f"{ref_losses.mean():.6f}")
            print(f"[ForeClassNet-"
                f"{self.forecast_loss_name}] "
                f"{100*self.bootstrap_level:.1f}% "
                f"bootstrap UCB forecast_tau = "
                f"{self.forecast_tau_:.6f}\n")
        classes, y_idx = np.unique(y_train_arr,return_inverse=True)
        self.classes_ = classes
        L_concat = (L_obs + self.L_fut)
        (self.classifier_,self.config_) = self._build_classifier(L_concat,D,num_class=len(classes))
        loader = self._make_loader_joint(X_obs_train,self.X_fut_train,y_idx,shuffle=True)
        optim_f = torch.optim.Adam(self.forecaster_.parameters(),lr=self.lr_forecast,weight_decay=self.weight_decay)
        optim_c = torch.optim.Adam(self.classifier_.parameters(),lr=self.lr_class,weight_decay=self.weight_decay)
        self.lambda_dual_ = max(0.0,self.lambda_init)
        self.forecaster_.train()
        self.classifier_.train()
        set_mc_dropout(self.forecaster_,True)
        set_mc_dropout(self.classifier_,True)
        self.training_history_ = []
        for epoch in range(self.max_epochs_class):
            total_D = 0.0
            total_F = 0.0
            total_constraint = 0.0
            n = 0
            it = tqdm(loader,desc=(f"[ForeClassNet-"
                    f"{self.forecast_loss_name}] "
                    f"Epoch {epoch+1}/"
                    f"{self.max_epochs_class}"),leave=True,) if TQDM else loader
            for xb_obs, xb_fut, yb in it:
                xb_obs = xb_obs.to(self.device)
                xb_fut = xb_fut.to(self.device)
                yb = yb.to(self.device)
                optim_f.zero_grad()
                optim_c.zero_grad()
                (loss_D,forecast_samples) = (self._marginal_classification_loss(xb_obs,yb,S=self.mc_samples_train))
                per_example_F = (self._forecast_loss_per_example(forecast_samples,xb_fut))
                loss_F = (per_example_F.mean())
                constraint = (loss_F / self.forecast_tau_- 1.0)
                loss_primal = (loss_D + self.lambda_dual_ * constraint + concrete_reg_loss(self.forecaster_)+ concrete_reg_loss(self.classifier_))
                loss_primal.backward()
                optim_f.step()
                optim_c.step()
                self.lambda_dual_ = max(0.0,self.lambda_dual_ + self.dual_lr * float(constraint.detach().cpu()))
                bs = xb_obs.size(0)
                total_D += (loss_D.item()* bs)
                total_F += (loss_F.item()* bs)
                total_constraint += (constraint.item()* bs)
                n += bs
            row = {"epoch": epoch + 1,
                "decision_loss":total_D/ max(1, n),
                "forecast_loss":total_F/ max(1, n),
                "constraint":total_constraint / max(1, n),
                "lambda":self.lambda_dual_,
                "tau":self.forecast_tau_,}
            self.training_history_.append(row)
            if self.verbose and not TQDM:
                print(f"[ForeClassNet-"
                    f"{self.forecast_loss_name}] "
                    f"epoch={row['epoch']} "
                    f"L_D={row['decision_loss']:.6f} "
                    f"L_F={row['forecast_loss']:.6f} "
                    f"g={row['constraint']:.6f} "
                    f"lambda={row['lambda']:.6f}")
        set_mc_dropout(self.forecaster_,False)
        set_mc_dropout(self.classifier_,False)
        self.is_fitted_ = True
        return self
class ForeClassNetMSE(ForeClassNetConstrained):
    def __init__(self,X_forecast_train,**kwargs):
        super().__init__(X_forecast_train=X_forecast_train,forecast_loss="mse",**kwargs)
class ForeClassNetEnergy(ForeClassNetConstrained):
    def __init__(self,X_forecast_train,**kwargs):
        super().__init__( X_forecast_train=X_forecast_train,forecast_loss="energy",**kwargs)
def _transformer_device(device=None):
    if device is not None:
        return device
    if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
def _transformer_revin_normalize(x,eps=1e-5):
    mean = x.mean(dim=1,keepdim=True)
    variance = x.var(dim=1,keepdim=True,unbiased=False)
    std = torch.sqrt(variance + eps)
    x_norm = (x - mean) / std
    return (x_norm,mean,std)
class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=10000,):
        super().__init__()
        position = torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(10000.0)/ d_model))
        pe = torch.zeros(max_len,d_model,dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position* div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe",pe.unsqueeze(0),persistent=False)
    def forward(self,x):
        length = x.size(1)
        if length > self.pe.size(1):
            raise ValueError(f"Sequence length {length} exceeds "
                f"positional-encoding maximum "
                f"{self.pe.size(1)}.")
        return (x+self.pe[:,:length].to(dtype=x.dtype))
class _TransformerClassifierNet(nn.Module):
    def __init__(self,input_dim,num_classes,d_model=64,nhead=4,num_layers=2,
        dim_feedforward=128,dropout=0.1,max_len=10000,use_revin=True):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.use_revin = bool(use_revin)
        self.input_projection = nn.Linear(input_dim,d_model)
        self.position = (_SinusoidalPositionalEncoding(d_model=d_model,max_len=max_len))
        encoder_layer = (nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=(
                    dim_feedforward),dropout=dropout,activation="gelu",batch_first=True,norm_first=False))
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(d_model,num_classes))
    def forward(self,x,):
        if self.use_revin:
            x, _, _ = (_transformer_revin_normalize(x))
        h = self.input_projection(x)
        h = self.position(h)
        h = self.encoder(h)
        h = self.final_norm(h)
        pooled = h.mean(dim=1)
        logits = self.head(pooled)
        return logits

class _TransformerForecasterNet(nn.Module):
    def __init__(self,input_dim,pred_len,d_model=64,nhead=4,num_encoder_layers=2,
        num_decoder_layers=2,dim_feedforward=128,dropout=0.1,max_len=10000,use_revin=True):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.pred_len = int(pred_len)
        self.use_revin = bool(use_revin)
        self.input_projection = nn.Linear(input_dim,d_model)
        self.source_position = (_SinusoidalPositionalEncoding(d_model=d_model,max_len=max_len))
        encoder_layer = (nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
                dim_feedforward=(dim_feedforward),dropout=dropout,activation="gelu",
                batch_first=True,norm_first=False))
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_encoder_layers)
        decoder_layer = (nn.TransformerDecoderLayer(d_model=d_model,nhead=nhead,
                dim_feedforward=(dim_feedforward),dropout=dropout,activation="gelu",
                batch_first=True,norm_first=False))
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layers=num_decoder_layers)
        self.future_queries = nn.Parameter(torch.empty(1,self.pred_len,d_model))
        nn.init.normal_(self.future_queries,mean=0.0,std=0.02)
        self.target_position = (_SinusoidalPositionalEncoding(d_model=d_model,max_len=max_len))
        self.output_projection = nn.Linear(d_model,input_dim)
    def forward(self,x):
        if self.use_revin:
            (x_norm,mean,std) = (_transformer_revin_normalize(x))
        else:
            x_norm = x
            mean = None
            std = None
        source = self.input_projection(x_norm)
        source = self.source_position(source)
        memory = self.encoder(source)
        queries = (self.future_queries.expand(x.size(0),-1,-1))
        queries = self.target_position(queries)
        decoded = self.decoder(tgt=queries,memory=memory)
        forecast_norm = (self.output_projection(decoded))
        if self.use_revin:
            forecast = (forecast_norm* std + mean)
        else:
            forecast = (forecast_norm)
        return forecast
    
class TransformerClassifier:
    def __init__(self,max_epochs=50,batch_size=512,lr=1e-3,weight_decay=1e-4,
        device=None,verbose=True,d_model=64,nhead=4,num_layers=2,dim_feedforward=128,
        dropout=0.1,use_revin=True):
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = (_transformer_device(device))
        self.verbose = bool(verbose)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.use_revin = bool(use_revin)
        self.model_ = None
        self.classes_ = None
        self.is_fitted_ = False
        if self.verbose:
            print("[TransformerClassifier] "f"Using device: {self.device}")
    def _loader(self,X,y_idx=None,shuffle=False):
        X_tensor = torch.from_numpy(np.asarray(X,dtype=np.float32))
        if y_idx is None:
            dataset = TensorDataset(X_tensor)
        else:
            y_tensor = torch.from_numpy(np.asarray(y_idx,dtype=np.int64))
            dataset = TensorDataset(X_tensor,y_tensor)
        return DataLoader(dataset,batch_size=self.batch_size,shuffle=shuffle)
    def fit(self,X,y):
        X = np.asarray(X,dtype=np.float32)
        y = to_int(y)
        if X.ndim != 3:
            raise ValueError("X must have shape " 
                             "(N, L, D).")
        N, L, D = X.shape
        (classes,y_idx) = np.unique(y,return_inverse=True)
        self.classes_ = classes
        self.model_ = (_TransformerClassifierNet(input_dim=D,num_classes=len(classes),d_model=self.d_model,
                nhead=self.nhead,num_layers=(self.num_layers),dim_feedforward=(self.dim_feedforward),
                dropout=self.dropout,max_len=max(64,L + 8),use_revin=(self.use_revin)).to(self.device))
        loader = self._loader(X,y_idx=y_idx,shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(),lr=self.lr,weight_decay=(self.weight_decay))
        criterion = (nn.CrossEntropyLoss())
        for epoch in range(self.max_epochs):
            self.model_.train()
            total_loss = 0.0
            n = 0
            iterator = (tqdm(loader,desc=("[TransformerClassifier] "
                        f"Epoch "
                        f"{epoch + 1}/"
                        f"{self.max_epochs}"),leave=True)if TQDM
                else loader)
            for xb, yb in iterator:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits,yb)
                loss.backward()
                optimizer.step()
                bs = xb.size(0)
                total_loss += (loss.item() * bs)
                n += bs
            if (self.verbose and not TQDM):
                print("[TransformerClassifier] "
                    f"Epoch "
                    f"{epoch + 1}/"
                    f"{self.max_epochs} "
                    f"loss="
                    f"{total_loss / max(1, n):.6f}")
        self.is_fitted_ = True
        return self
    def predict_proba(self,X,):
        if not self.is_fitted_:
            raise RuntimeError("TransformerClassifier "
                "is not fitted.")
        loader = self._loader(X,shuffle=False)
        self.model_.eval()
        probabilities = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model_(xb)
                p = F.softmax(logits,dim=-1)
                probabilities.append(p.cpu().numpy())
        return np.concatenate(probabilities,axis=0)
    def predict(self,X):
        probabilities = (self.predict_proba(X))
        indices = np.argmax(probabilities,axis=1)
        return self.classes_[indices]

class TransformerForecastThenClassify:
    def __init__(self,X_forecast_train,max_epochs_forecast=20,max_epochs_class=50,
        batch_size=512,lr_forecast=1e-3,lr_class=1e-3,weight_decay=1e-4,
        device=None,verbose=True,d_model=64,nhead=4,num_encoder_layers=2,num_decoder_layers=2,
        num_classifier_layers=2,dim_feedforward=128,dropout=0.1,use_revin=True):
        X_forecast_train = np.asarray(X_forecast_train,dtype=np.float32)
        if X_forecast_train.ndim != 3:
            raise ValueError("X_forecast_train must have "
                "shape (N, L_future, D).")
        self.X_fut_train = (X_forecast_train)
        self.L_fut = (X_forecast_train.shape[1])
        self.D = (X_forecast_train.shape[2])
        self.max_epochs_forecast = int(max_epochs_forecast)
        self.max_epochs_class = int(max_epochs_class)
        self.batch_size = int(batch_size)
        self.lr_forecast = float(lr_forecast)
        self.lr_class = float(lr_class)
        self.weight_decay = float(weight_decay)
        self.device = (_transformer_device(device))
        self.verbose = bool(verbose)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_classifier_layers = int(num_classifier_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.use_revin = bool(use_revin)
        self.forecaster_ = None
        self.classifier_ = None
        self.classes_ = None
        self.is_fitted_ = False
        if self.verbose:
            print("[TransformerForecastThenClassify] "
                f"Using device: {self.device}")
    def _forecast_loader(self,X_obs,X_fut,shuffle):
        dataset = TensorDataset(torch.from_numpy(np.asarray(X_obs,dtype=np.float32)),
            torch.from_numpy(np.asarray(X_fut,dtype=np.float32)))
        return DataLoader(dataset,batch_size=self.batch_size,shuffle=shuffle)
    def _plain_loader(self,X,shuffle=False):
        dataset = TensorDataset(torch.from_numpy(np.asarray(X,dtype=np.float32)))
        return DataLoader(dataset,batch_size=self.batch_size,shuffle=shuffle)
    def _build_forecaster(self,L_obs,D):
        max_len = max(64, L_obs + self.L_fut + 8)
        return (_TransformerForecasterNet(input_dim=D,pred_len=self.L_fut,d_model=self.d_model,
                nhead=self.nhead,num_encoder_layers=(self.num_encoder_layers),
                num_decoder_layers=(self.num_decoder_layers),dim_feedforward=(self.dim_feedforward),
                dropout=self.dropout,max_len=max_len,use_revin=(self.use_revin),).to(self.device))
    def _train_forecaster(self,X_obs):
        X_obs = np.asarray(X_obs,dtype=np.float32)
        N, L_obs, D = (X_obs.shape)
        if len(self.X_fut_train) != N:
            raise ValueError("Observed and future training "
                "arrays must have equal length.")
        if D != self.D:
            raise ValueError("Observed and future feature "
                "dimensions differ.")
        self.forecaster_ = (self._build_forecaster(L_obs,D))
        loader = (self._forecast_loader(X_obs,self.X_fut_train,shuffle=True))
        optimizer = torch.optim.Adam(self.forecaster_.parameters(),lr=self.lr_forecast,
            weight_decay=(self.weight_decay))
        criterion = nn.MSELoss()
        for epoch in range(self.max_epochs_forecast):
            self.forecaster_.train()
            total_loss = 0.0
            n = 0
            iterator = (tqdm(loader,desc=("[Transformer-forecast] "
                        f"Epoch "
                        f"{epoch + 1}/"
                        f"{self.max_epochs_forecast}"),leave=True,) if TQDM
                else loader)
            for xb, yb in iterator:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                forecast = (self.forecaster_(xb))
                loss = criterion(forecast,yb)
                loss.backward()
                optimizer.step()
                bs = xb.size(0)
                total_loss += (loss.item()* bs)
                n += bs
            if (self.verbose and not TQDM):
                print("[Transformer-forecast] "
                    f"Epoch "
                    f"{epoch + 1}/"
                    f"{self.max_epochs_forecast} "
                    f"loss="
                    f"{total_loss / max(1, n):.6f}")
    def _forecast(self,X_obs):
        if self.forecaster_ is None:
            raise RuntimeError("Forecaster is not fitted.")
        loader = self._plain_loader(X_obs,shuffle=False)
        self.forecaster_.eval()
        outputs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                prediction = (self.forecaster_(xb))
                outputs.append(prediction.cpu().numpy())
        return np.concatenate(outputs,axis=0)
    def fit(self,X_obs_train,y_train):
        X_obs_train = np.asarray(X_obs_train,dtype=np.float32)
        self._train_forecaster(X_obs_train)
        forecast_train = (self._forecast(X_obs_train))
        X_concat_train = np.concatenate([X_obs_train,forecast_train,],axis=1,)
        self.classifier_ = (TransformerClassifier(max_epochs=(self.max_epochs_class),
                batch_size=(self.batch_size),lr=self.lr_class,weight_decay=(self.weight_decay),
                device=self.device,verbose=self.verbose,d_model=self.d_model,nhead=self.nhead,
                num_layers=(self.num_classifier_layers),dim_feedforward=(self.dim_feedforward),
                dropout=self.dropout,use_revin=(self.use_revin),))
        self.classifier_.fit(X_concat_train,y_train)
        self.classes_ = (self.classifier_.classes_)
        self.is_fitted_ = True
        return self
    def predict_proba(self,X_obs,):
        if not self.is_fitted_:
            raise RuntimeError("TransformerForecastThenClassify "
                "is not fitted.")
        X_obs = np.asarray(X_obs,dtype=np.float32)
        forecast = self._forecast(X_obs)
        X_concat = np.concatenate([X_obs,forecast,],axis=1)
        return (self.classifier_.predict_proba(X_concat))
    def predict(self,X_obs):
        probabilities = (self.predict_proba(X_obs))
        indices = np.argmax(probabilities,axis=1)
        return self.classes_[indices]       

from concretedropout.pytorch import (
    ConcreteDropout1D as _RepositoryConcrete1D,
    ConcreteLinearDropout as _RepositoryConcreteLinear,
    get_weight_regularizer as _cd_weight_regularizer,
    get_dropout_regularizer as _cd_dropout_regularizer,
)

class _ForeClassSafeConcreteMixin:
    def forward(self, x, layer):
        eps = max(1e-6, torch.finfo(x.dtype).eps)
        self.p = torch.sigmoid(self.p_logit).clamp(eps, 1.0 - eps)
        self.regularization = self.get_regularization(x, layer)
        if self.training or self.is_mc_dropout:
            x = self.spatial_concrete_dropout(x)
        return layer(x)
    def get_regularization(self, x, layer):
        p = self.p
        input_dim = layer.in_features if isinstance(layer, nn.Linear) else x.shape[1]
        weight_term = self.weight_regularizer * layer.weight.square().sum() / (1.0 - p)
        entropy_term = p * torch.log(p) + (1.0 - p) * torch.log1p(-p)
        return (weight_term + self.dropout_regularizer * input_dim * entropy_term).sum()
    def spatial_concrete_dropout(self, x):
        shape = self._get_noise_shape(x)
        noise = torch.rand(shape, device=x.device, dtype=x.dtype)
        if self.dropout_mode == "bernoulli":
            keep = (noise >= self.p).to(x.dtype)
        else:
            eps = max(1e-6, torch.finfo(x.dtype).eps)
            noise = noise.clamp(eps, 1.0 - eps)
            relaxed_logit = (torch.log(self.p) - torch.log1p(-self.p) + torch.log(noise) - torch.log1p(-noise))
            keep = 1.0 - torch.sigmoid(relaxed_logit / self.temperature)
        return x * keep / (1.0 - self.p)
class _ForeClassConcrete1D(_ForeClassSafeConcreteMixin, _RepositoryConcrete1D):
    pass
class _ForeClassConcreteLinear(_ForeClassSafeConcreteMixin, _RepositoryConcreteLinear):
    pass
class _ForeClassConcreteLayer(nn.Module):
    def __init__(self, layer, *, weight_regularizer, dropout_regularizer,
                 init_p, temperature, dropout_mode, is_mc_dropout):
        super().__init__()
        self.layer = layer
        cd_type = _ForeClassConcreteLinear if isinstance(layer, nn.Linear) else _ForeClassConcrete1D
        self.concrete = cd_type(weight_regularizer=weight_regularizer,
            dropout_regularizer=dropout_regularizer,init_min=init_p,
            init_max=init_p,temperature=temperature,is_mc_dropout=is_mc_dropout)
        self.concrete.dropout_mode = dropout_mode
        self.concrete.p_logit.requires_grad_(dropout_mode == "concrete")
    def forward(self, x):
        return self.concrete(x, self.layer)
def _make_foreclass_concrete(model, *, N_train, l, tau, reg_scale,is_classification, init_p, temperature_conv,
                            temperature_linear, dropout_mode, is_mc_dropout):
    wr = reg_scale * _cd_weight_regularizer(N_train, l=l, tau=tau)
    dr = reg_scale * _cd_dropout_regularizer(N_train, tau=tau, cross_entropy_loss=is_classification)
    def wrap(module):
        for name, child in list(module.named_children()):
            if isinstance(child, _ForeClassConcreteLayer):
                continue
            if isinstance(child, (nn.Conv1d, nn.Linear)):
                setattr(module, name, _ForeClassConcreteLayer(child, weight_regularizer=wr, dropout_regularizer=dr,
                    init_p=init_p,temperature=(temperature_linear if isinstance(child, nn.Linear)
                                 else temperature_conv),dropout_mode=dropout_mode, is_mc_dropout=is_mc_dropout))
            else:
                wrap(child)
    wrap(model)
    return model
def concrete_reg_loss(module: nn.Module):
    parameter = next(module.parameters(), None)
    total = parameter.new_zeros(()) if parameter is not None else torch.zeros(())
    for child in module.modules():
        if isinstance(child, _ForeClassConcreteLayer):
            total = total + child.concrete.regularization
    return total
def set_mc_dropout(module: nn.Module, enabled: bool = True):
    for child in module.modules():
        if isinstance(child, _MCDropout):
            child.is_mc_dropout = bool(enabled)
        elif isinstance(child, _ForeClassConcreteLayer):
            child.concrete.is_mc_dropout = bool(enabled)
class ForeClassNetConcrete(BayesianModernTCNForecastThenClassifyJoint):
    def __init__(self, X_forecast_train, *, init_p=0.1,
                 temperature_conv=2.0 / 3.0, temperature_linear=0.1,
                 reg_scale_forecast=1.0, reg_scale_class=1.0,
                 dropout_mode="concrete", weight_decay=0.0, **kwargs):
        if not np.isfinite(init_p) or not 0.0 < init_p < 1.0:
            raise ValueError("init_p must be strictly between zero and one.")
        for name, value in (("temperature_conv", temperature_conv),("temperature_linear", temperature_linear)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (("reg_scale_forecast", reg_scale_forecast),("reg_scale_class", reg_scale_class),("weight_decay", weight_decay)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if dropout_mode not in ("concrete", "bernoulli"):
            raise ValueError("dropout_mode must be 'concrete' or 'bernoulli'.")
        super().__init__(X_forecast_train=X_forecast_train,weight_decay=weight_decay, **kwargs)
        if self.X_fut_train.shape[0] == 0:
            raise ValueError("Training data must not be empty.")
        if not np.isfinite(self.l) or self.l <= 0:
            raise ValueError("l must be finite and positive.")
        if not np.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be finite and positive.")
        self.init_p = float(init_p)
        self.temperature_conv = float(temperature_conv)
        self.temperature_linear = float(temperature_linear)
        self.reg_scale_forecast = float(reg_scale_forecast)
        self.reg_scale_class = float(reg_scale_class)
        self.dropout_mode = dropout_mode
    def _wrap_concrete(self, model, *, is_classification, is_mc_dropout):
        return _make_foreclass_concrete(model, N_train=len(self.X_fut_train), l=self.l, tau=self.tau,
            reg_scale=(self.reg_scale_class if is_classification else self.reg_scale_forecast),
            is_classification=is_classification, init_p=self.init_p,temperature_conv=self.temperature_conv,
            temperature_linear=self.temperature_linear,dropout_mode=self.dropout_mode, is_mc_dropout=is_mc_dropout).to(self.device)
    def _build_forecaster(self, L_obs, D):
        cfg = _ModernTCNForecastConfigBayes(seq_len=L_obs, pred_len=self.L_fut, enc_in=D)
        model = ModernTCNForecastModel(cfg)
        make_moderntcn_boltzmann(model)
        return self._wrap_concrete(model, is_classification=False, is_mc_dropout=False)
    def _build_classifier(self, L_obs_plus_fut, D, num_class):
        cfg = _ModernTCNClassConfigBayes(seq_len=L_obs_plus_fut, enc_in=D, num_class=num_class)
        model = ModernTCNClassModel(cfg)
        make_moderntcn_boltzmann(model)
        return self._wrap_concrete(model, is_classification=True, is_mc_dropout=True), cfg
    def fit(self, X_obs_train, y_train):
        if len(X_obs_train) != len(self.X_fut_train):
            raise ValueError("Observed and future training arrays must have equal length.")
        return super().fit(X_obs_train, y_train)
    def predict_proba_bayesian(self, X_obs, n_samples=100):
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        try:
            return super().predict_proba_bayesian(X_obs, n_samples=n_samples)
        finally:
            for model in (self.forecaster_, self.classifier_):
                if model is not None:
                    set_mc_dropout(model, False)
    def get_dropout_probabilities(self):
        """Return a JSON-serializable dictionary of final per-layer probabilities."""
        self._check()
        probabilities = {}
        for stage, model in (("forecaster", self.forecaster_),("classifier", self.classifier_)):
            for name, child in model.named_modules():
                if isinstance(child, _ForeClassConcreteLayer):
                    p = torch.sigmoid(child.concrete.p_logit.detach()).clamp(1e-6, 1.0 - 1e-6)
                    probabilities[f"{stage}.{name}"] = float(p.cpu().item())
        return probabilities