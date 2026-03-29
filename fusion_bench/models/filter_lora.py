import functools
import logging
from typing import List
import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F
from copy import deepcopy
from fusion_bench.utils.type import StateDictType
from fusion_bench.method.opcm.utils import svd
log = logging.getLogger(__name__)


class ResidualProjGate(nn.Module):
    def __init__(self, d=768, r=1, U=None):
        super().__init__()
        self.U = U
        r = max(r, 1)
        self.A = nn.Parameter(torch.zeros(r, d))  # LoRA A
        self.B = nn.Parameter(torch.zeros(d, r))  # LoRA B
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)        
        
    def forward(self, x):               # x: [..., d]
        if self.U is not None:
            x_u = x @ self.U
            x_filt = x - x_u @ self.U.T
        else:
            x_filt = x
        delta  = (x @ self.B) @ self.A
        return x_filt + delta
    
    @torch.no_grad()
    def get_weight(self):
        d = self.A.shape[1]
        device = self.A.device
        dtype = self.A.dtype
        if self.U is not None:
            P = torch.eye(d, device=device, dtype=dtype) - self.U @ self.U.T
        else:
            P = torch.eye(d, device=device, dtype=dtype)
        W_right = P + self.B @ self.A
        return W_right

class FilterLoRA(nn.Module):
    # ---- Linear-like aliases (delegate to base_model) ----
    @property
    def weight(self):
        return self.base_model.weight

    @weight.setter
    def weight(self, v):
        self.base_model.weight = v

    @property
    def bias(self):
        return getattr(self.base_model, "bias", None)

    @bias.setter
    def bias(self, v):
        if hasattr(self.base_model, "bias"):
            self.base_model.bias = v

    @property
    def in_features(self):
        return getattr(self.base_model, "in_features", None)

    @property
    def out_features(self):
        return getattr(self.base_model, "out_features", None)

    def __init__(self,
                 base_model,
                 task_tv,
                 null_space,
                 previous_merged_tv,
                 rank,
                 null_rank,
                 grad_rank,
                 accelerator,
                 ):
        super().__init__()        
        self.base_model = base_model.requires_grad_(False)
        self.task_tv = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=False).to(accelerator)
        self.task_tv.weight.data = task_tv
        self.task_tv.requires_grad_(False)
        self.prev_tv = previous_merged_tv
        _, _, pre_V = svd(previous_merged_tv, full_matrices=False)
        _, _, task_V = svd(task_tv, full_matrices=False)

        self.pre_V = pre_V[:, :null_rank]
        if grad_rank > 0:
            self.task_V = task_V[:, :grad_rank]
        else:
            self.task_V = None
            
        self.U = self.pre_V if null_space else None
        self.gate = ResidualProjGate(self.in_features, rank, self.U)
        self.gate = self.gate.to(accelerator)
      
    def forward(self, hidden_states: Tensor):
        base_act = self.base_model(hidden_states)
        task_act = self.task_tv(self.gate(hidden_states))
        output_hidden_states = base_act+task_act
        return output_hidden_states

    def solve_lora(self, lambda1: float = 1.0, lambda2: float = 1.0):
        """
        L(A,B) = sum_i λ_i || T_i - (M0 + T@B@A) V_i ||_F^2
        where:
            M0 = prev_tv + T@P
            T1 = T@V1,  V1 = self.task_V
            T2 = prev_tv@V2,  V2 = self.pre_V
        """
        T = self.task_tv.weight.data          # [o, d]  (Δθ_t)
        A = self.gate.A                       # [r, d]
        B = self.gate.B                       # [d, r]
        prev = self.prev_tv                   # [o, d]  (Δθ_{≤t-1})
        U = self.gate.U                       # [d, nr] or None
        loss = torch.tensor(0.0, device=T.device, dtype=T.dtype)
        has_proj1 = getattr(self, "task_V", None) is not None and lambda1 != 0.0
        has_proj2 = getattr(self, "pre_V", None) is not None and lambda2 != 0.0
        if has_proj1:
            V = self.task_V                   # [d, k1]
            target = T @ V
            prev_v = prev @ V
            if U is not None:
                u_t_v = U.T @ V
                u_utv = U @ u_t_v
                p_v = V - u_utv
            else:
                p_v = V
            t_p_v = T @ p_v
            m0_v = prev_v + t_p_v
            residual = target - m0_v
            a_v = A @ V
            b_av = B @ a_v
            t_bav = T @ b_av
            r = residual - t_bav
            loss = loss + lambda1 * torch.sum(r * r)
        if has_proj2:
            V = self.pre_V                    # [d, k2]
            target = prev @ V
            prev_v = prev @ V
            if U is not None:
                u_t_v = U.T @ V
                u_utv = U @ u_t_v
                p_v = V - u_utv
            else:
                p_v = V
            t_p_v = T @ p_v
            m0_v = prev_v + t_p_v
            residual = target - m0_v
            a_v = A @ V
            b_av = B @ a_v
            t_bav = T @ b_av
            r = residual - t_bav
            loss = loss + lambda2 * torch.sum(r * r)

        if not has_proj1 and not has_proj2:
            if U is not None:
                t_u = T @ U
                t_u_ut = t_u @ U.T
                t_p = T - t_u_ut
            else:
                t_p = T
            m0 = prev + t_p
            t_b = T @ B
            delta = t_b @ A
            m = m0 + delta
            r_full1 = T - m
            r_full2 = prev - m
            loss = loss + torch.sum(r_full1 * r_full1) + torch.sum(r_full2 * r_full2)
        return loss

    def merge_to_base(self):
        gate_w = self.gate.get_weight()
        tv_w = self.task_tv.weight.data
        merged_w = tv_w @ gate_w + self.base_model.weight.data
        return merged_w

