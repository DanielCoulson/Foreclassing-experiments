import argparse
import importlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import re
from scipy.stats import spearmanr
from torch.utils.data import (DataLoader,TensorDataset,)
from sklearn.linear_model import RidgeClassifierCV
from MyModels import (ModernTCNClassifier,ModernTCNForecastThenClassifyOfficial,
    BayesianModernTCNForecastThenClassifyJoint,ForeClassNetDynamic,ForeClassNetMSE,
    ForeClassNetEnergy,TransformerClassifier,TransformerForecastThenClassify,RocketMultivariateLengths,DynamicConv1d,set_mc_dropout)
SEEDS = range(5)
MC_SAMPLES = 100
SELECTED_MODEL = "all"
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
def sync_accelerator():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif (hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize")):
        torch.mps.synchronize()
def timed_call(function):
    sync_accelerator()
    start = time.perf_counter()
    result = function()
    sync_accelerator()
    elapsed = (time.perf_counter()- start)
    return result, elapsed
def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"
def hardware_information():
    if torch.cuda.is_available():
        gpu_name = (torch.cuda.get_device_name(0))
    elif (hasattr(torch.backends, "mps")and torch.backends.mps.is_available()):
        gpu_name = "Apple MPS"
    else:
        gpu_name = "CPU"
    information = {"hostname":socket.gethostname(),"platform":platform.platform(),
        "pytorch_version":torch.__version__,"cuda_version":torch.version.cuda,
        "device":choose_device(),"gpu_name":gpu_name,"slurm_job_id":os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list":os.environ.get("SLURM_JOB_NODELIST")}
    try:
        information["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi","--query-gpu=""name,driver_version,""memory.total",
                "--format=csv,noheader",],text=True,).strip()
    except Exception:
        information["nvidia_smi"] = None
    return information
def save_probability_output(output_dir,model_name,seed,y_true,proba,training_time,
    inference_time,proba_std=None,proba_mc_off=None,mc_off_inference_time=None,
    extra=None,classes = None):
    output_dir.mkdir(parents=True,exist_ok=True)
    proba = np.asarray(proba)
    y_true = np.asarray(y_true).astype(int)
    if classes is None:
        classes = np.arange(proba.shape[1])
    classes = np.asarray(classes)
    y_pred_default = classes[np.argmax(proba, axis=1)]
    arrays = { "y_true":y_true,"y_pred_default":y_pred_default,
        "proba":proba,"training_time_seconds":np.asarray(training_time),
        "inference_time_seconds":np.asarray(inference_time, ), "classes": classes}
    if proba_std is not None:
        arrays[ "proba_std"] = np.asarray(proba_std)
    if proba_mc_off is not None:
        arrays["proba_mc_off"] = np.asarray(proba_mc_off)
    if mc_off_inference_time is not None:
        arrays["mc_off_inference_time_seconds"] = np.asarray(mc_off_inference_time)
    if extra is not None:
        for key, value in extra.items():
            if (value is not None and np.isscalar(value)):
                arrays[key] = (np.asarray(value))
    np.savez_compressed(output_dir/f"{model_name}_seed_{seed}.npz",**arrays)
def run_probability_model(model_name,model_factory,fit_function,X_test,y_test,output_dir,timing_rows,bayesian=False,after_inference = None):
    if SELECTED_MODEL not in ("all", model_name):
        return
    for seed in SEEDS:
        print("\n"+ "=" * 90)
        print(f"{model_name} "f"— seed {seed}")
        print("=" * 90)
        set_seed(seed)
        clf = model_factory(seed)
        _, training_time = timed_call(lambda:fit_function(clf))
        if bayesian:
            (prediction_result,inference_time,) = timed_call(lambda:clf.predict_proba_bayesian(X_test.astype(np.float32),
                        n_samples=(MC_SAMPLES),))
            (proba,proba_std,) = prediction_result
            (proba_mc_off,mc_off_inference_time,) = timed_call(lambda:clf.predict_proba(X_test.astype(np.float32)))
        else:
            (proba,inference_time) = timed_call(lambda:clf.predict_proba(X_test.astype(np.float32)))
            proba_std = None
            proba_mc_off = None
            mc_off_inference_time = None
        extra = {"forecast_tau":getattr(clf,"forecast_tau_",None),"final_lambda":getattr(clf,"lambda_dual_",None)}
        history = getattr(clf,"training_history_", None)
        if history:
            extra["final_constraint"] = history[-1].get("constraint")
            pd.DataFrame(history).to_csv(output_dir /f"{model_name}_"f"seed_{seed}_"f"training_history.csv",index=False)
        save_probability_output(output_dir=output_dir, model_name=model_name,
            seed=seed, y_true=y_test,proba=proba,proba_std=proba_std,proba_mc_off=(proba_mc_off),
            training_time=(training_time),inference_time=(inference_time),mc_off_inference_time=(mc_off_inference_time),
            extra=extra, classes = getattr(clf,"classes_",None))
        if after_inference is not None: 
            after_inference(clf=clf,seed=seed,X_test=X_test,y_test=y_test,output_dir=output_dir)
        timing_rows.append({"model":model_name,"seed":seed,"training_time_seconds":training_time,
            "inference_time_seconds":inference_time,"mc_samples":(MC_SAMPLES if bayesian else 1),
            "mc_off_inference_time_seconds":mc_off_inference_time,}) 
def _safe_column_name(name):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(name),)
def collect_dynamic_attention(clf,X_obs):
    X_obs = np.asarray(X_obs,dtype=np.float32)
    if X_obs.ndim != 3:
        raise ValueError("X_obs must have shape "
            "(N, L, D).")
    clf.forecaster_.eval()
    clf.classifier_.eval()
    set_mc_dropout(clf.forecaster_,False)
    set_mc_dropout(clf.classifier_,False)
    captured = {}
    metadata = {}
    hooks = []
    def register_dynamic_layers(component_name,model):
        for layer_name, module in (model.named_modules()):
            if not isinstance(module,DynamicConv1d):
                continue
            if len(module.kernel_sizes) <= 1:
                continue
            key = (f"{component_name}::"
                f"{layer_name}")
            captured[key] = []
            metadata[key] = {"component":component_name,"layer":layer_name,
                "kernel_sizes":list(module.kernel_sizes)}
            def make_hook(storage_key):
                def hook(mod,inputs,output):
                    if (mod.last_attention is not None):
                        captured[storage_key].append(mod.last_attention.detach().cpu().numpy().copy())
                return hook
            hooks.append(module.register_forward_hook(make_hook(key)))
    register_dynamic_layers("forecaster",clf.forecaster_)
    register_dynamic_layers("classifier",clf.classifier_)
    if len(captured) == 0:
        raise RuntimeError("No multi-kernel DynamicConv1d layers "
            "were found in this model.")
    try:
        obs_loader = (clf._make_loader_no_y(X_obs,shuffle=False))
        future_chunks = []
        with torch.no_grad():
            for (xb,) in obs_loader:
                xb = xb.to(clf.device)
                yhat = clf.forecaster_(xb,None,None, None)
                future_chunks.append(yhat.cpu().numpy())
        future_pred = np.concatenate(future_chunks,axis=0)
        X_concat = np.concatenate([X_obs,future_pred],axis=1).astype(np.float32)
        classifier_loader = DataLoader(TensorDataset(torch.from_numpy(X_concat)),batch_size=clf.batch_size,shuffle=False)
        with torch.no_grad():
            for (xb,) in classifier_loader:
                xb = xb.to(clf.device)
                _ = clf.classifier_(xb,None,None,None)
    finally:
        for hook in hooks:
            hook.remove()
    attention_data = {}
    for key in captured:
        if len(captured[key]) == 0:
            raise RuntimeError(f"No weights captured for "
                f"{key}.")
        W = np.concatenate(captured[key],axis=0)
        if len(W) != len(X_obs):
            raise RuntimeError(f"{key}: expected "
                f"{len(X_obs)} examples, "
                f"but captured {len(W)}.")
        attention_data[key] = {**metadata[key],"weights": W}
    return attention_data
def compute_time_series_pattern_descriptors(X):
    X = np.asarray(X,dtype=np.float64)
    N, L, D = X.shape
    temporal_sd = (np.std(X,axis=1).mean(axis=1))
    if L > 1:
        mean_abs_change = (np.abs(np.diff(X,axis=1)).mean(axis=(1, 2)))
    else:
        mean_abs_change = np.zeros(N)
    temporal_range = ((X.max(axis=1)-X.min(axis=1)).mean(axis=1))
    t = np.arange(L,dtype=np.float64)
    t_centered = (t - t.mean())
    denominator = np.sum(t_centered ** 2)
    if denominator > 0:
        slopes = np.einsum("t,ntd->nd",t_centered, X) / denominator
        mean_signed_slope = (slopes.mean(axis=1))
        mean_abs_slope = (np.abs(slopes).mean(axis=1))
    else:
        mean_signed_slope = (np.zeros(N))
        mean_abs_slope = (np.zeros(N))
    mean_level = X.mean(axis=(1, 2))
    return pd.DataFrame({"example_index":np.arange(N),"mean_level":mean_level,
        "temporal_sd":temporal_sd,"mean_abs_change":mean_abs_change,"temporal_range":temporal_range,
        "mean_signed_slope":mean_signed_slope,"mean_abs_slope":mean_abs_slope})
def summarise_dynamic_attention(attention_data,seed):
    kernel_rows = []
    layer_rows = []
    for key, item in (attention_data.items()):
        W = np.asarray(item["weights"],dtype=float)
        kernel_sizes = np.asarray(item["kernel_sizes"])
        K = W.shape[1]
        for j, kernel_size in enumerate(kernel_sizes):
            w = W[:, j]
            kernel_rows.append({"seed":seed,"component":item["component"],
                "layer":item["layer"],"kernel_size":int(kernel_size),
                "mean_weight":np.mean(w),"std_weight_across_examples":np.std(w,ddof=1,),
                "min_weight":np.min(w),"q05_weight":np.quantile(w,0.05,),"median_weight":np.median(w),
                "q95_weight":np.quantile(w,0.95,),"max_weight":np.max(w)})
        mean_mixture = W.mean(axis=0)
        kernel_sds = W.std(axis=0,ddof=1)
        l1_from_mean = np.abs(W - mean_mixture).sum(axis=1)
        dominant_index = np.argmax(W,axis=1)
        dominant_kernel = (kernel_sizes[dominant_index])
        _, dominant_counts = (np.unique(dominant_kernel,return_counts=True))
        dominant_switch_rate = (1.0 - dominant_counts.max() / len(dominant_kernel))
        eps = 1e-12
        entropy = -np.sum(W*np.log(W + eps),axis=1)
        normalized_entropy = (entropy /np.log(K))
        layer_rows.append({"seed":seed,"component":item["component"],"layer":item["layer"],
            "n_examples":len(W),"n_candidate_kernels":K,"kernel_sizes":str(list(map(int,kernel_sizes))),
            "mean_kernel_weight_sd":np.mean(kernel_sds),"max_kernel_weight_sd":np.max(kernel_sds),
            "mean_l1_from_layer_mean":np.mean(l1_from_mean),"median_l1_from_layer_mean":np.median(l1_from_mean),
            "n_dominant_kernels_used":len(np.unique(dominant_kernel)),"dominant_kernel_switch_rate":dominant_switch_rate,
            "mean_normalized_entropy":np.mean(normalized_entropy),"std_normalized_entropy":np.std(normalized_entropy,ddof=1)})
    return (pd.DataFrame(kernel_rows),pd.DataFrame(layer_rows))
def dynamic_pattern_correlations(attention_data,descriptors,seed):
    descriptor_names = ["mean_level","temporal_sd","mean_abs_change","temporal_range",
        "mean_signed_slope","mean_abs_slope"]
    rows = []
    for key, item in (attention_data.items()):
        W = np.asarray(item["weights"],dtype=float)
        kernel_sizes = (item["kernel_sizes"])
        for j, kernel_size in enumerate(kernel_sizes):
            weight = W[:, j]
            for descriptor_name in (descriptor_names):
                x = (descriptors[descriptor_name].to_numpy(dtype=float))
                valid = (np.isfinite(x) & np.isfinite(weight))
                if (valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(weight[valid]) == 0):
                    rho = np.nan
                    p_value = np.nan
                else:
                    rho, p_value = (spearmanr(x[valid],weight[valid]))
                rows.append({"seed":seed,"component":item["component"],"layer":item["layer"],
                    "kernel_size":int(kernel_size),"pattern_descriptor":descriptor_name,
                    "spearman_rho":rho,"abs_spearman_rho":(abs(rho)if np.isfinite(rho)else np.nan),
                    "spearman_p":p_value})
    return pd.DataFrame(rows)
def make_dynamic_attention_example_table(attention_data,descriptors,y_true,seed):
    output = (descriptors.copy())
    output.insert(0,"seed",seed)
    output["y_true"] = np.asarray(y_true).astype(int)
    for key, item in (attention_data.items()):
        W = np.asarray(item["weights"])
        prefix = _safe_column_name(key)
        kernel_sizes = np.asarray(item["kernel_sizes"])
        for j, kernel_size in enumerate(kernel_sizes):
            output[f"{prefix}__weight_k"
                f"{int(kernel_size)}"] = W[:, j]
        dominant_idx = np.argmax(W,axis=1)
        output[f"{prefix}__dominant_kernel"] = kernel_sizes[dominant_idx]
        eps = 1e-12
        entropy = -np.sum(W*np.log(W + eps),axis=1)
        output[f"{prefix}__normalized_entropy"] = (entropy /np.log(W.shape[1]))
    return output
def save_dynamic_diagnostics(clf,seed,X_test,y_test,output_dir,):
    print("\nInspecting input-dependent dynamic weights "
        f"for seed {seed}...")
    dynamic_output_dir = (output_dir.parent/ "dynamic_convolution_analysis")
    dynamic_output_dir.mkdir(parents=True,exist_ok=True,)
    attention_data = collect_dynamic_attention(clf,X_test.astype(np.float32))
    pattern_descriptors = (compute_time_series_pattern_descriptors(X_test))
    (kernel_summary,layer_summary) = summarise_dynamic_attention(attention_data,seed=seed)
    pattern_correlations = (dynamic_pattern_correlations(attention_data,pattern_descriptors,seed=seed))
    example_table = (make_dynamic_attention_example_table(attention_data,pattern_descriptors,y_true=y_test,seed=seed))
    example_table.to_csv(dynamic_output_dir / ("ForeClassNetDynamic_"
            f"attention_examples_seed_{seed}.csv.gz"),index=False,compression="gzip")
    kernel_summary.to_csv(dynamic_output_dir / ("ForeClassNetDynamic_"
            f"kernel_summary_seed_{seed}.csv"),index=False)
    layer_summary.to_csv(dynamic_output_dir / ("ForeClassNetDynamic_"
                                               f"layer_summary_seed_{seed}.csv"),index=False)
    pattern_correlations.to_csv(dynamic_output_dir / ("ForeClassNetDynamic_"
            f"pattern_correlations_seed_{seed}.csv"),index=False)
    print("\nDynamic adaptation summary:")
    print(layer_summary.round(4).to_string(index=False))
    print("\nDynamic-convolution diagnostics saved to:")
    print(dynamic_output_dir.resolve())
def main():
    global SEEDS, SELECTED_MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="Oil_example_1")
    parser.add_argument("--results-root",default="results")
    parser.add_argument("--model",default="all",choices=["all","ModernTCN",
            "ModernTCNForecastThenClassify","ForeClassNet",
            "TransformerClassifier","TransformerForecastThenClassify","ForeClassNetDynamic",
            "ForeClassNetMSE","ForeClassNetEnergy","ROCKET"])
    parser.add_argument("--seed",type=int,default=None)
    args = parser.parse_args()
    if (args.seed is not None and args.seed not in range(5)):
        parser.error("--seed must be one of 0, 1, 2, 3, 4")
    SELECTED_MODEL = args.model
    SEEDS = (range(5)if args.seed is None else [args.seed])
    run_tag = (f"{args.model}_seed_{args.seed}" if args.seed is not None else f"{args.model}_all_seeds")
    dataset_module = (importlib.import_module(args.dataset))
    data = (dataset_module.get_data())
    dataset_name = (data["dataset_name"] )
    output_dir = (Path(args.results_root)/dataset_name/"raw_outputs")
    output_dir.mkdir(parents=True,exist_ok=True)
    device = choose_device()
    print("Dataset:",dataset_name)
    print("Device:",device)
    with open(output_dir.parent/f"hardware_{run_tag}.json","w") as f:
        json.dump(hardware_information(),f,indent=2)
    np.savez_compressed(output_dir.parent/f"dataset_metadata_{run_tag}.npz",y_test=np.asarray(data["y_test"]),
    test_starts=np.asarray(data["test_starts"]),m=np.asarray(data["m"]),k=np.asarray(data["k"]),
    window=np.asarray(data["window"]))
    X_train = (data["X_observed_train"].astype(np.float32))
    X_future_train = (data["X_forecast_train"].astype(np.float32))
    y_train = (data["y_train"].astype(int))
    X_test = (data["X_observed_test"].astype(np.float32))
    y_test = (data["y_test"].astype(int))
    timing_rows = []
    run_probability_model(model_name="ModernTCN",model_factory=lambda seed:ModernTCNClassifier(
            max_epochs=50,batch_size=512,lr=1e-3,weight_decay=1e-4,device=device,verbose=True),
    fit_function=lambda clf:clf.fit(X_train,y_train,),X_test=X_test,y_test=y_test,output_dir=output_dir,
    timing_rows=timing_rows,bayesian=False)
    run_probability_model(model_name=("ModernTCNForecastThenClassify"),model_factory=lambda seed:
        ModernTCNForecastThenClassifyOfficial(X_forecast_train=(X_future_train),
            max_epochs_forecast=20,max_epochs_class=50,batch_size=512,lr_forecast=1e-3,
            lr_class=1e-3,weight_decay=1e-4,device=device,verbose=True),
    fit_function=lambda clf:clf.fit(X_train,y_train,),X_test=X_test,y_test=y_test,
    output_dir=output_dir,timing_rows=timing_rows,bayesian=False)
    run_probability_model(model_name="ForeClassNet",model_factory=lambda seed:BayesianModernTCNForecastThenClassifyJoint(
                X_forecast_train=(X_future_train),max_epochs_forecast=20,max_epochs_class=50,batch_size=512,
                lr_forecast=1e-3,lr_class=1e-3,weight_decay=1e-4,device=device,verbose=True,l=1e-2,tau=1.0,),
        fit_function=lambda clf:clf.fit(X_train,y_train,),X_test=X_test,y_test=y_test,output_dir=output_dir,
        timing_rows=timing_rows,bayesian=True)     
    run_probability_model(model_name="TransformerClassifier",model_factory=lambda seed:TransformerClassifier(
            max_epochs=50,batch_size=512,lr=1e-3, weight_decay=1e-4,device=device,verbose=True,
            d_model=64,nhead=4,num_layers=2,dim_feedforward=128,dropout=0.1, use_revin=True,
        ),fit_function=lambda clf:clf.fit(X_train,y_train,),X_test=X_test,y_test=y_test,
    output_dir=output_dir,timing_rows=timing_rows,bayesian=False)       
    run_probability_model(model_name=("TransformerForecastThenClassify"),model_factory=lambda seed:TransformerForecastThenClassify(
            X_forecast_train=(X_future_train),max_epochs_forecast=20,max_epochs_class=50,batch_size=512,lr_forecast=1e-3,lr_class=1e-3,
            weight_decay=1e-4,device=device,verbose=True,d_model=64,nhead=4,num_encoder_layers=2,num_decoder_layers=2, num_classifier_layers=2,
            dim_feedforward=128,dropout=0.1,use_revin=True,),fit_function=lambda clf:clf.fit(X_train,y_train,),X_test=X_test,y_test=y_test,
    output_dir=output_dir,timing_rows=timing_rows,bayesian=False)    
    run_probability_model(model_name="ForeClassNetDynamic",model_factory=lambda seed:ForeClassNetDynamic(X_forecast_train=X_future_train,
            max_epochs_forecast=20,max_epochs_class=50,batch_size=512,lr_forecast=1e-3,
            lr_class=1e-3,weight_decay=1e-4,device=device,verbose=True,l=1e-2,tau=1.0,
            dynamic_temperature=1.0,dynamic_reduction=4,dynamic_min_hidden=8,),
            fit_function=lambda clf:clf.fit(X_train,y_train,),X_test=X_test, 
            y_test=y_test,output_dir=output_dir,timing_rows=timing_rows,bayesian=True,
            after_inference=save_dynamic_diagnostics)
    X_fit = (data["X_observed_fit"].astype(np.float32))
    X_future_fit = (data["X_forecast_fit"].astype(np.float32))
    y_fit = (data["y_fit"].astype(int))
    X_val = (data["X_observed_val"].astype(np.float32))
    X_future_val = (data["X_forecast_val"].astype(np.float32))
    y_val = (data["y_val"].astype(int)) 
    run_probability_model(model_name="ForeClassNetMSE",model_factory=lambda seed:ForeClassNetMSE(X_forecast_train=(X_future_fit),
            max_epochs_forecast=20,max_epochs_class=50,batch_size=512,lr_forecast=1e-3,lr_class=1e-3,weight_decay=1e-4,mc_samples_train=8,
            mc_samples_tau=32,bootstrap_B=2000,bootstrap_level=0.95,dual_lr=1e-2,lambda_init=0.0,l=1e-2,tau=1.0,random_state=seed,device=device,
            verbose=True,),fit_function=lambda clf:clf.fit(X_fit,y_fit,X_val,X_future_val), 
                          X_test=X_test,y_test=y_test,output_dir=output_dir,timing_rows=timing_rows,bayesian=True) 
    run_probability_model(model_name="ForeClassNetEnergy",model_factory=lambda seed:ForeClassNetEnergy(X_forecast_train=(X_future_fit),
            max_epochs_forecast=20,max_epochs_class=50,batch_size=512,lr_forecast=1e-3,lr_class=1e-3,weight_decay=1e-4,mc_samples_train=8,
            mc_samples_tau=32,bootstrap_B=2000,bootstrap_level=0.95,dual_lr=1e-2,lambda_init=0.0,l=1e-2,tau=1.0,random_state=seed,device=device,verbose=True), 
                          fit_function=lambda clf:clf.fit(X_fit,y_fit,X_val,X_future_val,),X_test=X_test,y_test=y_test,output_dir=output_dir,timing_rows=timing_rows, 
                          bayesian=True)  
    for seed in (SEEDS if args.model in ("all", "ROCKET") else []): 
        print("\n"+ "=" * 90) 
        print(f"ROCKET + RidgeClassifierCV "f"— seed {seed}") 
        print("=" * 90) 
        set_seed(seed) 
        y_train_labels = np.asarray(y_train).astype(int) 
        y_test_labels = np.asarray(y_test).astype(int) 
        rocket = (RocketMultivariateLengths(n_kernels=10000,lengths=(2,3,4,),random_state=seed)) 
        def train_rocket():
            rocket.fit(X_train.astype(np.float32))
            X_train_feat = (rocket.transform(X_train.astype(np.float32)))
            ridge = RidgeClassifierCV(alphas=np.logspace(-3,3,10,))
            ridge.fit(X_train_feat,y_train_labels)
            return ridge
        (ridge,training_time,) = timed_call(train_rocket)
        def infer_rocket():
            X_test_feat = (rocket.transform(X_test.astype(np.float32)))
            decision_scores = (ridge.decision_function(X_test_feat))
            return (X_test_feat,decision_scores)
        (rocket_inference_result,inference_time,) = timed_call(infer_rocket)
        (X_test_feat,decision_scores,) = rocket_inference_result
        y_pred = ridge.predict(X_test_feat)
        np.savez_compressed(output_dir / f"ROCKET_seed_{seed}.npz",y_true=y_test_labels, 
                            y_pred_default=y_pred,decision_score=decision_scores, 
                            classes=np.asarray(ridge.classes_), 
                            training_time_seconds=np.asarray(training_time), 
                            inference_time_seconds=np.asarray(inference_time))
        timing_rows.append({"model":"ROCKET", "seed":seed,"training_time_seconds":training_time,
            "inference_time_seconds":inference_time,"mc_samples":1,"mc_off_inference_time_seconds": None}) 
    timings_df = pd.DataFrame(timing_rows) 
    timing_path = (output_dir.parent/f"timing_{run_tag}.csv")
    timings_df.to_csv(timing_path,index=False)
    print("\n"+ "=" * 90)
    print("ALL MODEL TRAINING FINISHED")
    print("=" * 90)
    print("\nRaw model outputs saved to:")
    print(output_dir.resolve())
    print("\nTiming information saved to:")
    print(timing_path.resolve())  
if __name__ == "__main__":
    main()