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
from transformers import CLIPVisionModel
from fusion_bench.utils.data import InfiniteDataLoader
from torch.utils.data import DataLoader
from fusion_bench.utils import timeit_context, gpu_mem_context

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.models.hf_clip import HFCLIPClassifier
from .utils import is_leaf_module, svd
from fusion_bench.models.filter_lora import FilterLoRA
import logging
from torch.utils.data import Subset


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class NUFILT(BaseAlgorithm, LightningFabricMixin):
    def __init__(
        self,
        lora_r: int,
        null_r: int,
        grad_r: int,
        null_space: bool = True,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        **kwargs,
    ):
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self.lora_r = lora_r
        self.null_r = null_r
        self.null_space = null_space
        self.grad_r = grad_r
        self._config = DictConfig(kwargs)
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        accelerator = self.fabric.device
        self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
        self._test_datasets = deepcopy(self.taskpool._test_datasets)
        pretrained_model = modelpool.load_pretrained_model().to(accelerator)
        merged_model = deepcopy(pretrained_model)
        merged_model.requires_grad_(False)


        for model_idx, model_name in tqdm(
            enumerate(model_names), desc="Processing models"
        ):
            task_model = modelpool.load_model(model_name).to(accelerator)
            # stats: dict = {}
            with timeit_context("Merging time"):
                for module_name, module in tqdm(list(task_model.named_modules()), desc=f"Processing {model_name}", leave=False):
                    if not is_leaf_module(module):
                        continue

                    merged_module = merged_model.get_submodule(module_name)

                    if isinstance(module, nn.Linear):
                        do_lora = False

                        if any(key in module_name for key in self._config.lora_layer):
                            do_lora = True

                        if not do_lora:
                            # skip everything else
                            continue

                        task_vector = task_model.get_submodule(module_name).weight.data - pretrained_model.get_submodule(module_name).weight.data
                        previous_merged_tv = merged_module.weight.data - pretrained_model.get_submodule(module_name).weight.data

                        filter_lora = FilterLoRA(
                            base_model=merged_module,
                            task_tv=task_vector,
                            previous_merged_tv=previous_merged_tv,
                            null_space=self.null_space,
                            rank=self.lora_r,
                            null_rank=self.null_r,
                            grad_rank=self.grad_r,
                            accelerator=accelerator,
                        )
                        merged_model.set_submodule(module_name, filter_lora)

                if self.lora_r > 0 and model_idx > 0:
                    merged_model = self.solve_lora(merged_model)

                merged_model = self.merge_model(merged_model)

            torch.cuda.empty_cache()

            if self.save_on_every_step:
                self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step or model_idx == len(model_names) - 1:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {n: self._test_datasets[n] for n in model_names[:model_idx + 1]}
                )
                if not self.evaluate_on_every_step: 
                    return merged_model
                report = self.taskpool.evaluate(deepcopy(merged_model.to(accelerator)))
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")
            torch.cuda.empty_cache()

        return merged_model

    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        torch.save(
            merged_model.state_dict(),
            Path(self.log_dir) / "checkpoints" / f"model_{step}.pth",
        )

    def solve_lora(self, model):
        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if p.requires_grad and 'gate' in n],
            lr=self._config.lr
        )

        steps = self._config.max_steps
        if self._config.get("fast_dev_run", False):
            steps = 1
            print("========= Optimized Parameters =========")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    print(f"Name: {n}, Shape: {p.shape}, Requires Grad: {p.requires_grad}")
            print("========================================")

        pbar = tqdm(range(steps), "Solve LoRA", dynamic_ncols=True)

        with gpu_mem_context("Testing GPU memory usage") as mem_tracker:
            with timeit_context("solving time"):
                for step_idx in pbar:
                    loss = 0
                    for name, module in list(model.named_modules()):
                        if isinstance(module, FilterLoRA):
                            loss += module.solve_lora()
                
                    loss.backward()
                    mem_tracker.update_peak_memory()
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.set_postfix({"loss": loss.item()})

        return model

    @torch.no_grad()
    def merge_model(self, model):
        for name, module in list(model.named_modules()):
            if isinstance(module, FilterLoRA):
                merged_w = module.merge_to_base()  
                base = module.base_model        

                new_base = deepcopy(base)
                new_base.weight.data = merged_w
                
                parent, attr = model, name
                if "." in name:
                    parent_name, attr = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                setattr(parent, attr, new_base)
        return model