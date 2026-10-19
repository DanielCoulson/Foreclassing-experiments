import argparse
import json
import os
import platform
import random
import socket
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ScalableMultiRocketHydra import MultiRocketHydraClassifier
import Stock_example_1
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--results-root",default="results")
    args = parser.parse_args()
    seed = args.seed
    if seed not in range(5):
        parser.error("--seed must be 0,1,2,3,4")
    set_seed(seed)
    data = Stock_example_1.get_data()
    dataset_name = data["dataset_name"]
    X_train = np.asarray(data["X_observed_train"],dtype=np.float32)
    X_test = np.asarray(data["X_observed_test"],dtype=np.float32)
    y_train = np.asarray(data["y_train"],dtype=int)
    y_test = np.asarray(data["y_test"],dtype=int)
    X_train = X_train.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)
    output_root = (Path(args.results_root) / dataset_name)
    output_dir = (output_root / "raw_outputs")
    output_dir.mkdir(parents=True,exist_ok=True)
    print("=" * 90)
    print("Dataset:", dataset_name)
    print("Model: MultiRocketHydraClassifier")
    print("Seed:", seed)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    print("Node:", socket.gethostname())
    print("=" * 90)
    clf = MultiRocketHydraClassifier(random_state=seed)
    start = time.perf_counter()
    clf.fit(X_train,y_train)
    training_time = (time.perf_counter()- start)
    start = time.perf_counter()
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    inference_time = (time.perf_counter()- start)
    classes = np.asarray(getattr(clf,"classes_",np.unique(y_train)))
    np.savez_compressed(output_dir / f"MRHydra_seed_{seed}.npz",y_true=y_test,
        y_pred_default=np.asarray(y_pred,dtype=int),proba=np.asarray(proba),
        classes=classes,training_time_seconds=np.asarray(training_time),
        inference_time_seconds=np.asarray(inference_time))
    pd.DataFrame([{"model": "MRHydra","seed": seed,"training_time_seconds":training_time,
            "inference_time_seconds":inference_time,"mc_samples": 1,"mc_off_inference_time_seconds":None}]).to_csv(
        output_root/ f"timing_MRHydra_seed_{seed}.csv",index=False)
    with open(output_root / f"hardware_MRHydra_seed_{seed}.json","w") as f:
        json.dump({"hostname":socket.gethostname(),"platform":platform.platform(),
                "device":"cpu","classifier":"MultiRocketHydraClassifier","slurm_job_id":os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id":os.environ.get("SLURM_ARRAY_TASK_ID")},f,indent=2)
    np.savez_compressed(output_root / f"dataset_metadata_MRHydra_seed_{seed}.npz",y_test=y_test,test_starts=np.asarray(
            data["test_starts"]),m=np.asarray(data["m"]),k=np.asarray(data["k"]),window=np.asarray(data["window"]))
    print("MRHydra finished successfully.")
if __name__ == "__main__":
    main()