import pathlib

import numpy as np
import pandas as pd

from core import dataloader
from core import helper
import sklearn
from sklearn.metrics import classification_report
import argparse
import time
timer = time.perf_counter

# parser = argparse.ArgumentParser(description='time series classifiers as stopper.')
# parser.add_argument(
#     'window_size', type=int, help='window size for classifers'
# )
# parser.add_argument(
#     'step_size', type=int, help='step size for classifers'
# )
# args = parser.parse_args()
# print(args)

# TRAIN_DATA_PATH = pathlib.Path("./data/training/dataset_exp4")
# TRAIN_DATA_PATH = pathlib.Path("./double_descent_data")
TRAIN_DATA_PATH = pathlib.Path("./data/testing/real_world_data")
training_set = dataloader.TrainingLogDataset(TRAIN_DATA_PATH)
training_set.loadDataset()

# OUT_PATH = pathlib.Path("./out/cmp_early_stop")
# OUT_PATH = pathlib.Path("./out/dd_cmp_early_stop")
OUT_PATH = pathlib.Path("./out/test_whole_history")
OUT_PATH.mkdir(exist_ok=True)
print(training_set)

models_path = pathlib.Path("./out")
# for cls_name in ["tsf", "tsbf", "bossvs", "saxvsm", "knndtw"]:
for cls_name in ["tsf", "tsbf", "bossvs", "hmmgmm", "saxvsm", "knndtw", "autocorr", "pearson", "spearman"]:
# for cls_name in ["autocorr", "pearson", "spearman"]:
    print("="*9, cls_name, "="*9)
    # models = models_path.glob("*.plk")
    model_path = list(models_path.glob(f"{cls_name}_*.pkl"))[0]
    model = helper.readPkl(model_path)
    # model
    # model_path

    # classifier_window = args.window_size
    # step = classifier_window // 10
    # step = args.step_size

    def addInfo(classifier_stop_res):
        dst_len = len(classifier_stop_res["total_time"])
        classifier_stop_res["label"] = training_set.labels[:dst_len]
        classifier_stop_res["name"] = training_set.names[:dst_len]
        classifier_stop_res["window_size"] = [10] * dst_len
        classifier_stop_res["step"] = [10] * dst_len
        return classifier_stop_res

    classifier_stop_res = {
        # "label": training_set.labels,
        # "name": training_set.names,
        # "is_stopped": [],
        "is_overfit": [],
        # "stop_epoch": [],
        # "best_epoch": [],
        # "best_loss": [],
        "total_time": [],
        "timer_count": [],
        # "window_size": [classifier_window] * len(training_set.names),
        # "step": [step] * len(training_set.names),
    }
    for idx, name in enumerate(training_set.names):
        idx = training_set.names.index(name)
        cur_data = training_set.data[idx]
        total_time = 0
        timer_count = 0
        is_overfit = []
        for i in range(len(cur_data["monitor_metric"])):
            if i < 10:
                is_overfit.append(0)
                continue
            end_epoch = i
            window_data = {n: d[:end_epoch] for n, d in cur_data.items()}
            if hasattr(model, "preprocessor"):
                processed_data = model.preprocessor.process([window_data])
            else:
                processed_data = [window_data]
            t1 = timer()
            res = model.predict(processed_data)
            t2 = timer()
            total_time += t2 - t1
            timer_count += 1
            is_overfit.append(int(res[0]))
        classifier_stop_res["is_overfit"].append(is_overfit)
        classifier_stop_res["total_time"].append(total_time)
        classifier_stop_res["timer_count"].append(timer_count)
        # break
        if idx % 50 == 0:
            print(f"{idx}/{len(training_set.names)}")
            classifier_stop_res = addInfo(classifier_stop_res)
            tmp = pd.DataFrame.from_dict(classifier_stop_res)
            tmp.to_csv(OUT_PATH / f"{model_path.stem}.csv", index=False)
    classifier_stop_res = addInfo(classifier_stop_res)
    classifier_stop_res = pd.DataFrame.from_dict(classifier_stop_res)
    classifier_stop_res.to_csv(OUT_PATH / f"{model_path.stem}.csv", index=False)
