import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils import instantiate
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.utils import timeit_context, gpu_mem_context
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_sub


def wudi_merging(
    task_vectors: List[torch.Tensor],
    accelerator="cuda",
    iter_num: int = 300,
    exclude_keys: List[str] = None,
):
    exclude_keys = [] if exclude_keys is None else exclude_keys

    with gpu_mem_context("Testing GPU memory usage") as mem_tracker:
        with timeit_context("WUDI Merging"):
            new_vector = {}
            for key in tqdm(task_vectors[0], desc="WUDI Merging"):
                # tqdm.write(f"key: {key}")
                original_device = task_vectors[0][key].device
                tvs = torch.stack(
                    [
                        task_vector[key].to(device=accelerator, non_blocking=True)
                        for task_vector in task_vectors
                    ]
                )
                num_tvs = len(tvs)
                new_vector[key] = torch.nn.Parameter(torch.sum(tvs, dim=0))

                if len(task_vectors[0][key].shape) == 2 and key not in exclude_keys:
                    optimizer = torch.optim.Adam([new_vector[key]], lr=1e-5, weight_decay=0)
                    l2_norms = torch.square(
                        torch.norm(tvs.reshape(tvs.shape[0], -1), p=2, dim=-1)
                    )
                    for i in tqdm(
                        range(iter_num), leave=False
                    ):
                        # disturbing_vectors = new_vector[key].unsqueeze(0) - tvs
                        # product = torch.matmul(disturbing_vectors, tvs.transpose(1, 2))
                        # loss = torch.sum(
                        #     torch.square(product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
                        # )

                        disturbing_vectors = new_vector[key].unsqueeze(0) - tvs  # Shape: (N, C, D)
                        gram_u = torch.matmul(disturbing_vectors.transpose(1, 2), disturbing_vectors)  # Shape: (N, D, D)
                        gram_v = torch.matmul(tvs.transpose(1, 2), tvs)  # Shape: (N, D, D), can be precomputed outside the loop if tvs is fixed
                        sum_squares = torch.sum(gram_u * gram_v, dim=(1, 2))  # Shape: (N,)
                        loss = torch.sum(sum_squares / l2_norms)  # Scalar

                        optimizer.zero_grad()
                        loss.backward()
                        mem_tracker.update_peak_memory()
                        optimizer.step()
                else:
                    new_vector[key] = new_vector[key] / num_tvs
                new_vector[key] = new_vector[key].to(
                    device=original_device, non_blocking=True
                )
    return new_vector


class WUDIMerging(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Whoever Started the Interference Should End It:  Guiding Data-Free Model Merging via Task Vectors
    """

    def __init__(
        self,
        iter_num: int,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        exclude_keys: List[str] = None,
        **kwargs,
    ):
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.iter_num = iter_num
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self.exclude_keys = exclude_keys
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        self.taskpool = self._program.taskpool  # generic, not casting

        # log the model names
        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")

        # get the average model
        pretrained_model = modelpool.load_pretrained_model()
        merged_model = modelpool.load_model(model_names[0])

        if self.evaluate_on_every_step:
            self.taskpool._is_setup = False
            seen_task_names = [
                model_names[0].replace("glue-", "")
            ]
            self.taskpool._all_task_names = seen_task_names
            report = self.taskpool.evaluate(deepcopy(merged_model))
            save_to_json(report, Path(self.log_dir) / f"report_0.json")


        for model_idx, model_name in tqdm(
            enumerate(model_names[1:]), desc="Processing models"
        ):
            model_idx += 1
            task_model = modelpool.load_model(model_name)

            task_vectors = []
            task_vectors.append(
                state_dict_sub(
                    merged_model.state_dict(), pretrained_model.state_dict()
                )
            )
            task_vectors.append(
                state_dict_sub(
                    task_model.state_dict(), pretrained_model.state_dict()
                )
            )


            merged_tv = wudi_merging(
                task_vectors,
                accelerator=self.fabric.device,
                iter_num=self.iter_num,
                exclude_keys=self.exclude_keys,
            )

            merged_model.load_state_dict(
                state_dict_add(pretrained_model.state_dict(), merged_tv)
            )

            if self.save_on_every_step:
                self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step or model_idx == len(model_names) - 1:
                self.taskpool._is_setup = False
                seen_task_names = [
                    mn.replace("glue-", "")
                    for mn in model_names[:model_idx + 1]
                ]
                self.taskpool._all_task_names = seen_task_names
                report = self.taskpool.evaluate(deepcopy(merged_model))
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

        return merged_model




        pretrained_model.load_state_dict(
            state_dict_add(pretrained_model.state_dict(), merged_tv)
        )

        return pretrained_model
