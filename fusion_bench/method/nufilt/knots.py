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

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method.ties_merging.ties_merging_utils import (
    state_dict_to_vector,
    ties_merging,
    vector_to_state_dict,
)
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_sub

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def subspace_alignment(
    delta_weights: list[torch.Tensor],
    svd_dtype: torch.dtype | None = torch.float64,
    eps: float = 1e-4,
):
    """
    Reference: Model merging with SVD to tie the Knots. http://arxiv.org/abs/2410.19735
    """
    if svd_dtype is None:
        svd_dtype = delta_weights[0].dtype
    original_dtype = delta_weights[0].dtype
    output_dim, input_dim = delta_weights[0].size()
    concat_task_vector = torch.cat(delta_weights, dim=1)
    U, S, Vh = torch.linalg.svd(concat_task_vector.to(svd_dtype), full_matrices=False)
    # Keep only supported basis components
    U = U[:, S > eps].to(original_dtype)
    Vh = Vh[S > eps].to(original_dtype)
    S = S[S > eps].to(original_dtype)
    Vhs = torch.split(Vh, input_dim, dim=1)
    return U, S, Vhs


class ContinualKnotsTiesForCLIP(BaseAlgorithm, LightningFabricMixin):
    def __init__(
        self,
        scaling_factor: float,
        threshold: float,
        remove_keys: Optional[List[str]] = None,
        merge_func: Literal["sum", "mean", "max"] = "sum",
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        **kwargs,
    ):
        """
        Continual Model Merging via Ties-Merging.

        Args:
            scaling_factor (float): the scaling factor to use.
            shuffle_order (bool): whether to shuffle the order of the models.
            seed (Optional[int]): the seed to use.
            save_on_every_step (bool): whether to save the merged model on every step.
            evaluate_on_every_step (bool): whether to evaluate the merged model on every step.
        """
        self.scaling_factor = scaling_factor
        self.threshold = threshold
        self.remove_keys = remove_keys if remove_keys is not None else []
        self.merge_func = merge_func
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
        self._test_datasets = deepcopy(self.taskpool._test_datasets)
        """Configuration for the test datasets"""

        # log the model names
        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")
            tensorboard_summarywriter: "SummaryWriter" = self.tensorboard_summarywriter
            tensorboard_summarywriter.add_text(
                "global/model_names", str(model_names), global_step=0
            )

        # get the average model
        pretrained_model = modelpool.load_pretrained_model()
        merged_model = deepcopy(pretrained_model)

        for model_idx, model_name in tqdm(
            enumerate(model_names), desc="Processing models"
        ):
            task_model = modelpool.load_model(model_name)

            for module_name, module in tqdm(
                list(task_model.named_modules()),
                desc=f"Processing {model_name}",
                leave=False,
            ):                    
                if not isinstance(module, nn.Linear):
                    continue

                task_param = module.weight
                merged_param = merged_model.get_submodule(module_name).weight
                pretrained_param = pretrained_model.get_submodule(module_name).weight
                merged_tv = merged_param - pretrained_param
                task_vector = task_param - pretrained_param

                if model_idx == 0:
                    new_param = merged_param + self.scaling_factor * task_vector
                else:
                    U, S, Vhs = subspace_alignment([merged_tv, task_vector])
                    flat_Vhs = [vh.flatten() for vh in Vhs]  
                    flat_Vhs = torch.stack(flat_Vhs, dim=0)  # (num_Vh, dim)
                    ties_merging_Vhs = ties_merging(
                        flat_Vhs,
                        reset_thresh=self.threshold,
                        merge_func=self.merge_func,
                    )
                    ties_merging_Vhs = ties_merging_Vhs.view(Vhs[0].shape) 
                    ties_merging_tv = U @ torch.diag(S) @ ties_merging_Vhs
                    new_param = merged_param + self.scaling_factor * ties_merging_tv

                merged_model.get_submodule(module_name).weight.data = new_param

            if self.save_on_every_step:
                self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {n: self._test_datasets[n] for n in model_names[: model_idx + 1]}
                )
                report = self.taskpool.evaluate(deepcopy(merged_model))
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

        return merged_model

    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        torch.save(
            merged_model.state_dict(),
            Path(self.log_dir) / "checkpoints" / f"model_{step}.pth",
        )
