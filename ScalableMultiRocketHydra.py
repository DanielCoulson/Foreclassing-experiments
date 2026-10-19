# MultiRocketHydra implementation
# This script contains the implementation used for the financial experiments.
# For the Shanghai, New York City, and Tokyo weather experiments, we used
# the standard MultiRocketHydraClassifier provided by the aeon library. 
import gc
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from aeon.classification.convolution_based import (MultiRocketHydraClassifier as _AeonMultiRocketHydraClassifier)
from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import (HydraTransformer)
from aeon.utils.validation import check_n_jobs
class MultiRocketHydraClassifier(_AeonMultiRocketHydraClassifier):
    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)
        print(f"[MRH] fitting Hydra transform on {len(X)} cases",flush=True)
        self._transform_hydra = HydraTransformer(n_kernels=self.n_kernels,n_groups=self.n_groups,n_jobs=self._n_jobs,
            random_state=self.random_state)
        Xt_hydra = self._transform_hydra.fit_transform(X)
        self._scale_hydra = _SparseScaler()
        Xt_hydra = self._scale_hydra.fit_transform(Xt_hydra)
        if hasattr(Xt_hydra, "detach"):
            Xt_hydra = Xt_hydra.detach().cpu().numpy()
        Xt_hydra = np.asarray(Xt_hydra, dtype=np.float32)
        print(f"[MRH] Hydra features: "
            f"{Xt_hydra.shape}, {Xt_hydra.dtype}",flush=True)
        print("[MRH] fitting MultiRocket transform", flush=True)
        self._transform_multirocket = MultiRocket(n_jobs=self._n_jobs,random_state=self.random_state)
        Xt_multirocket = self._transform_multirocket.fit_transform(X)
        Xt_multirocket = np.asarray(Xt_multirocket,dtype=np.float32)
        self._scale_multirocket = StandardScaler()
        Xt_multirocket = self._scale_multirocket.fit_transform(Xt_multirocket).astype(np.float32, copy=False)
        print(f"[MRH] MultiRocket features: "
            f"{Xt_multirocket.shape}, {Xt_multirocket.dtype}",flush=True)
        print("[MRH] concatenating feature matrices", flush=True)
        Xt = np.concatenate((Xt_hydra, Xt_multirocket),axis=1).astype(np.float32, copy=False)
        del Xt_hydra
        del Xt_multirocket
        gc.collect()
        gb = Xt.nbytes / (1024 ** 3)
        print(f"[MRH] combined features: {Xt.shape}, "
            f"{Xt.dtype}, {gb:.2f} GiB",flush=True)
        n = len(y)
        split = int(np.floor(0.8 * n))
        X_fit = Xt[:split]
        y_fit = y[:split]
        X_val = Xt[split:]
        y_val = y[split:]
        alphas = np.logspace(-3, 3, 10)
        print(f"[MRH] selecting alpha using "
            f"{len(y_fit)} fit / {len(y_val)} validation cases",
            flush=True)
        best_alpha = None
        best_acc = -np.inf
        for alpha in alphas:
            clf = RidgeClassifier(alpha=float(alpha),solver="lsqr",class_weight=self.class_weight)
            clf.fit(X_fit, y_fit)
            pred = clf.predict(X_val)
            acc = accuracy_score(y_val, pred)
            print(f"[MRH] alpha={alpha:.6g}, "
                f"validation accuracy={acc:.6f}",flush=True)
            if acc > best_acc:
                best_acc = acc
                best_alpha = float(alpha)
            del clf
            gc.collect()
        self.best_alpha_ = best_alpha
        self.validation_accuracy_ = best_acc
        print(f"[MRH] selected alpha={best_alpha:.6g} "
            f"(validation accuracy={best_acc:.6f})",flush=True)

        print("[MRH] fitting final LSQR RidgeClassifier "
            "on full training set",flush=True)
        self.classifier = RidgeClassifier(alpha=best_alpha,solver="lsqr",class_weight=self.class_weight)
        self.classifier.fit(Xt, y)
        print("[MRH] final classifier fitted", flush=True)
        del Xt
        gc.collect()
        return self