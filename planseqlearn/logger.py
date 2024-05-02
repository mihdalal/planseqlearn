# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import datetime
from collections import defaultdict, OrderedDict

import torch
import wandb
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from rlkit.core.eval_util import create_stats_ordered_dict
import numpy as np
import time

COMMON_TRAIN_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("buffer_size", "BS", "int"),
    ("fps", "FPS", "float"),
    ("total_time", "T", "time"),
]

COMMON_EVAL_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


def _format(key, value, ty):
    if ty == "int":
        value = int(value)
        return f"{key}: {value}"
    elif ty == "float":
        return f"{key}: {value:.04f}"
    elif ty == "time":
        value = str(datetime.timedelta(seconds=int(value)))
        return f"{key}: {value}"
    else:
        raise f"invalid format type: {ty}"


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = {}
        for key, meter in self._meters.items():
            if key.startswith("train"):
                key = key[len("train") + 1 :]
            else:
                key = key[len("eval") + 1 :]
            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["episode"]) >= data["episode"]:
                    break
                rows.append(row)
        with self._csv_file_name.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(data.keys()), restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open("a")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=sorted(data.keys()), restval=0.0
            )
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(_format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["frame"] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb, use_wandb):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(
            log_dir / "train.csv", formating=COMMON_TRAIN_FORMAT
        )
        self._eval_mg = MetersGroup(log_dir / "eval.csv", formating=COMMON_EVAL_FORMAT)
        if use_tb:
            self._sw = SummaryWriter(str(log_dir / "tb"))
        else:
            self._sw = None

        self.use_wandb = use_wandb

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_wandb_log(self, key, value, step):
        if self.use_wandb:
            wandb.log({key: value}, step=step)

    def log(self, key, value, step):
        assert key.startswith("train") or key.startswith("eval")
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._try_sw_log(key, value, step)
        self._try_wandb_log(key, value, step)
        mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f"{ty}/{key}", value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == "eval":
            self._eval_mg.dump(step, "eval")
        if ty is None or ty == "train":
            self._train_mg.dump(step, "train")

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f"{self._ty}/{key}", value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)


# computing path infos for rlkit dump
import rlkit.pythonplusplus as ppp


def compute_path_info(infos):
    all_env_infos = [ppp.list_of_dicts__to__dict_of_lists(ep_info) for ep_info in infos]
    statistics = OrderedDict()
    stat_prefix = ""
    for k in all_env_infos[0].keys():
        final_ks = np.array([info[k][-1] for info in all_env_infos])
        first_ks = np.array([info[k][0] for info in all_env_infos])
        all_ks = np.concatenate([info[k] for info in all_env_infos])
        statistics.update(
            create_stats_ordered_dict(
                stat_prefix + k,
                final_ks,
                stat_prefix="{}/final/".format("env_infos"),
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                stat_prefix + k,
                first_ks,
                stat_prefix="{}/initial/".format("env_infos"),
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                stat_prefix + k,
                all_ks,
                stat_prefix="{}/".format("env_infos"),
            )
        )
    return statistics


# add env infos into everything here
def compute_path_info(infos):
    all_env_infos = [ppp.list_of_dicts__to__dict_of_lists(ep_info) for ep_info in infos]
    statistics = OrderedDict()
    for k in all_env_infos[0].keys():
        means = [np.mean(info[k]) for info in all_env_infos]
        maxes = [np.max(info[k]) for info in all_env_infos]
        mins = [np.min(info[k]) for info in all_env_infos]
        initials = [info[k][0] for info in all_env_infos]
        finals = [info[k][-1] for info in all_env_infos]
        # create stats ordered dict for each value
        statistics.update(
            create_stats_ordered_dict(
                f"env_infos/{k + ' Mean'}",
                means,
                exclude_max_min=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                f"env_infos/{k + ' Max'}",
                maxes,
                exclude_max_min=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                f"env_infos/{k + ' Min'}",
                mins,
                exclude_max_min=True,
            )
        )
    return statistics
