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
from torch.utils.data import DataLoader

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
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


class NUFILTForT5(BaseAlgorithm, LightningFabricMixin):
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
        self.taskpool = self._program.taskpool  # generic, not casting

        pretrained_model = modelpool.load_pretrained_model().to(accelerator)
        merged_model = deepcopy(pretrained_model)
        merged_model.requires_grad_(False)

        for model_idx, model_name in tqdm(enumerate(model_names)):
            task_model = modelpool.load_model(model_name)
            task_model = task_model.to(accelerator)
            # stats: dict = {}

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
                    # _, _, task_V = svd(task_vector, full_matrices=False)
                    # merged_module.weight.data = task_model.get_submodule(module_name).weight.data

                    # stats[module_name] = task_V.cpu().numpy()

                    lora_moe = FilterLoRA(
                        base_model=merged_module,
                        task_tv=task_vector,
                        previous_merged_tv=previous_merged_tv,
                        null_space=self.null_space,
                        rank=self.lora_r,
                        null_rank=self.null_r,
                        grad_rank=self.grad_r,
                        accelerator=accelerator,
                    )
                    merged_model.set_submodule(module_name, lora_moe)

            if self.lora_r > 0 and model_idx > 0:
                merged_model = self.solve_lora(merged_model)

            merged_model = self.merge_model(merged_model)

            # save_path = Path(self.log_dir) / "motiv_exp" / f"tv_svd_{model_name}.npz"
            # save_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
            # np.savez(save_path, **stats)
            # print(f"[Saved] tv svd for {model_name} → {save_path}")
            # del stats
            
            # self.compute_and_save_svd(merged_model, model_name.replace("glue-", ""))

            torch.cuda.empty_cache()

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

    def save_merged_model(self, merged_model, step: int):
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

        for step_idx in pbar:
            loss = 0
            for name, module in list(model.named_modules()):
                if isinstance(module, FilterLoRA):
                    loss += module.solve_lora()
          
            loss.backward()

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
        

    # @torch.no_grad()
    # def compute_and_save_svd(self, model, model_name: str):
    #     task = self.taskpool.load_task(model_name)
    #     dataset = task.test_dataset

    #     loader = DataLoader(dataset, batch_size=self._config.batch_size,
    #                         shuffle=True, num_workers=0, pin_memory=False,
    #                         collate_fn=getattr(task, "test_loader").collate_fn
    #                         if hasattr(task, "test_loader") else None)
    
    #     cov_dict = {}
    #     handles = []
    
    #     def hook_fn(mod_name):
    #         def hook(module, inp, out):
    #             input_tensor = inp[0].detach()
    #             if input_tensor.dim() > 2:
    #                 input_tensor = input_tensor.view(-1, input_tensor.size(-1))
    #             elif input_tensor.dim() < 2:
    #                 return  # Skip if not suitable for SVD
    #             d = input_tensor.size(-1)
    #             if mod_name not in cov_dict:
    #                 cov_dict[mod_name] = torch.zeros(d, d, device=input_tensor.device)
    #             cov_dict[mod_name] += input_tensor.T @ input_tensor
    #         return hook
    
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Linear) and 'encoder' in name:
    #             if any(key in name for key in self._config.lora_layer):
    #                 handle = module.register_forward_hook(hook_fn(name))
    #                 handles.append(handle)
    #             else:
    #                 continue    

    #     for batch in loader:
    #         _ = self.compute_logits(model, batch)
    
    #     # Remove hooks
    #     for handle in handles:
    #         handle.remove()
    
    #     stats1: dict = {}
    #     stats2: dict = {}
    #     for name in cov_dict:
    #         C = cov_dict[name]
    #         U, S, _ = torch.svd(C)
    #         stats1[name] = U.cpu().numpy()  # Convert to numpy for saving
    #         stats2[name] = S.cpu().numpy()  # Convert to numpy for saving

    #     save_path = Path(self.log_dir) / "motiv_exp" / f"data_svd_U_{model_name}.npz"
    #     save_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
    #     np.savez(save_path, **stats1)
    #     save_path = Path(self.log_dir) / "motiv_exp" / f"data_svd_S_{model_name}.npz"
    #     save_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
    #     np.savez(save_path, **stats2)
    #     print(f"[Saved] data svd for {model_name} → {save_path}")



    # def compute_logits(
    #     self,
    #     module,
    #     batch,
    # ) -> Tensor:
    #     """
    #     Compute the logits for the given images and task.

    #     Args:
    #         module: The model module.
    #         images (Tensor): The input images.
    #     Returns:
    #         Tensor: The computed logits.
    #     """
    #     input_ids: Tensor = batch["input_ids"]
    #     attention_mask: Tensor = batch["attention_mask"]

    #     # remove padding tokens from the input
    #     while attention_mask[:, -1].eq(0).all():
    #         input_ids = input_ids[:, :-1].to(self.fabric.device)
    #         attention_mask = attention_mask[:, :-1].to(self.fabric.device)

    #     outputs = module(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=torch.ones(
    #             input_ids.size(0), 1, dtype=torch.long, device=input_ids.device
    #         ),
    #     )
    #     logits = outputs.logits[:, 0, :]
    #     return logits
