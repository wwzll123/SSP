"""
对称自博弈（Self-Play）+ 在线偏好优化（DPO）for protein inverse folding（backbone -> sequence）。

本文件只封装“Model”（不写训练过程）：`SSP_model(nn.Module)` 的 `forward(g_batch)` 会完成：
- 用 A/B 在同一 backbone 上各采样 K 个候选序列（非自回归、逐位点独立采样）；
- 调用 oracle（ESMFold 折叠 + 与输入 backbone 对齐）得到 TM/RMSD/pLDDT/pTM；
- 置信过滤 + reward 标量化；
- 在线构造偏好对（top vs bottom + margin）；
- 对 A 和 B 分别计算 DPO loss（ref_model 做锚点）；
- 计算 A/B 分布的 JS divergence“排斥项”（防同质化，最大化 JS）；
- （可选）给 B 加熵奖励以鼓励探索；
- 返回包含 loss 与日志的 dict。

依赖对齐（参考 `single_SPIF_train.py` / `utils.py`）：
- base inverse folding 模型：`model(g_batch, recycle_steps=...) -> logits: (N,20)`
- g_batch: PyG Batch，至少包含 `batch` 和 `PDB_ID`（batch_size 建议保持 1）
- oracle: 需提供 `score(g_batch, sequence_str) -> metrics dict`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.tokenization import get_esm3_model_tokenizers
from esm.utils.constants import esm3 as esm3_constants
import re
from peft import get_peft_model,LoraConfig 
try:
    import utils  # MapDiff-main/utils.py（训练脚本里也这样 import）
except Exception:  # pragma: no cover
    utils = None


tokenizer_collection=get_esm3_model_tokenizers()



@dataclass
class Candidate:
    idx: torch.Tensor              # (N,) long，逐残基 aa index（对应 logits 的 N）
    seq: str                       # 单条序列字符串（当前实现假设 batch_size=1）
    source: str                    # "A" 或 "B"
    metrics: Dict[str, float]      # oracle 输出
    reward_sc: float               # 基于 TM/RMSD 的 reward（结构相容性）
    reward_pred: float             # 基于 pLDDT/pTM 的 reward（模型稳定性）


class ESMFoldOracle:
    """
    可选：一个与你当前 `single_SPIF_train.py` 对齐的 oracle 实现。
    如果你已有更完整的 oracle（例如直接和 g_batch/backbone 对齐），可以不用这个类。
    """

    def __init__(self, esmfold_model, pdb_dir: str = None):
        if utils is None:
            raise ImportError("未能 import utils.py；请确保运行时 PYTHONPATH 包含 MapDiff-main。")
        self.esmfold_model = esmfold_model
        self.pdb_dir = pdb_dir if pdb_dir is not None else '/root/autodl-tmp/cath_chain_dir/train'

    @torch.no_grad()
    def score_batch(self, batch_data: Dict[str, Any], sequences: List[str]) -> Dict[str, torch.Tensor]:
        """
        批量版 ESMFold 打分：
        输入 sequences: list[str]（batch），返回每个指标的 tensor，形状 (B,)。
        """
        pdb_id = batch_data['pdb_id']
        tm, rmsd, plddt, ptm, pae = utils.calculate_metrics(
            true_pdb_path=os.path.join(self.pdb_dir, f'{pdb_id}.pdb'),
            gen_sequences=sequences,
            esmfold_model=self.esmfold_model
        )
        
        if tm is None:
            # 失败兜底：返回全 nan，避免训练崩掉
            b = len(sequences)
            nan = torch.full((b,), float("nan"))
            return {"TM": nan, "RMSD": nan, "pLDDT": nan, "pTM": nan, "pAE": nan}

        # 统一为 1D tensor（在 CPU 上即可；训练侧只用它们做过滤/日志）
        return {"TM": tm, "RMSD": rmsd, "pLDDT": plddt, "pTM": ptm, "pAE": pae}


class SSP_model(nn.Module):
    """
    Self-Play + DPO 的“模型封装层”。

    你需要保证三个模型具有一致 forward：
      logits = model(g_batch, recycle_steps=...)  -> (N,20)
    """

    def __init__(
        self,
        model_A: nn.Module,
        model_B: nn.Module,
        ref_model: nn.Module,
        *,
        oracle: Optional[Any] = None,
        k_samples: int = 10,
        # 可选：让 ref_model 也生成候选（早期稳定候选质量/避免策略漂移后全不过滤）
        include_ref_candidates: bool = True,
        k_ref_samples: int = 5,
        temp_ref: float = 0.5,
        pair_margin: float = 0.05,
        max_pairs: int = 16,
        min_plddt: float = 0.45,
        min_ptm: float = 0.35,
        dpo_beta: float = 0.1,
        use_js: bool = True,           # 是否使用 JS divergence 损失
        lambda_js: float = 0.02,
        # SFT（交叉熵）loss 系数：希望小于 DPO 主项（隐式为 1），但大于 JS 的系数（例如 0.02）
        lambda_sft: float = 0.5,
        temp_A: float = 1.0,
        temp_B: float = 1.0,
        # 熵奖励：默认 A 与 B 使用相同系数；如需可单独指定 entropy_bonus_A
        entropy_bonus_B: float = 0.01,
        entropy_bonus_A: float =0.005,
        reward_w_tm: float = 0.4,
        reward_w_rmsd_term: float = 0.6,
        top_k: int = 10,
        top_p: float = 0.9,
        recycle_steps: int = 0,
        ema_decay: float = 0.995,
    ):
        super().__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.ref_model = ref_model
        self.oracle = oracle

        self.k_samples = int(k_samples)
        self.include_ref_candidates = bool(include_ref_candidates)
        self.k_ref_samples = int(k_ref_samples)
        self.temp_ref = float(temp_ref)
        self.pair_margin = float(pair_margin)
        self.max_pairs = int(max_pairs)
        self.min_plddt = float(min_plddt)
        self.min_ptm = float(min_ptm)
        self.dpo_beta = float(dpo_beta)
        self.use_js = bool(use_js)
        self.lambda_js = float(lambda_js)
        self.lambda_sft = float(lambda_sft)
        self.temp_A = float(temp_A)
        self.temp_B = float(temp_B)
        self.entropy_bonus_B = float(entropy_bonus_B)
        self.entropy_bonus_A = float(entropy_bonus_B) if entropy_bonus_A is None else float(entropy_bonus_A)
        self.reward_w_tm = float(reward_w_tm)
        self.reward_w_rmsd_term = float(reward_w_rmsd_term)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.recycle_steps = int(recycle_steps)
        self.ema_decay = float(ema_decay)

        self._aa_token_ids = self._get_aa_token_ids()
        self._freeze_ref()

    @staticmethod
    def _get_aa_token_ids() -> List[int]:
        vocab = {tok: i for i, tok in enumerate(esm3_constants.SEQUENCE_VOCAB)}
        aa_tokens = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        return [int(vocab[a]) for a in aa_tokens if a in vocab]

    def _freeze_ref(self) -> None:
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def _forward_logits(self, model: nn.Module, 
                        coord: torch.Tensor, #seq_length*37*3
                        structure_tokens: torch.Tensor) -> torch.Tensor: #seq_length
        
        if coord.dim()==3:
            coord=coord.unsqueeze(0) #seq_length*37*3 --> 1*seq_length*37*3
        if structure_tokens.dim()==1:
            structure_tokens=structure_tokens.unsqueeze(0) #seq_length --> 1*seq_length
        with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):
            outputs=model(structure_tokens=structure_tokens,
              structure_coords=coord)
        return outputs.sequence_logits.squeeze(0) #1*seq_length*20-->seq_length*20


    @staticmethod
    def _idx_to_seq(idx: torch.Tensor) -> str:
        # idx: (N,)
        seq=tokenizer_collection.sequence.decode(idx,skip_special_tokens=True)
        return seq.replace(' ','')

    @torch.no_grad()
    def _sample_indices(self, logits: torch.Tensor, 
                        *, temperature: float,
                        num_samples: int) -> List[torch.Tensor]:
        """
        非自回归逐位点采样：对每个残基独立从 softmax(logits/temp) 抽样。
        返回 num_samples 个 (N,) long。
        """
        if num_samples <= 0:
            return []
        t = float(temperature)
        if t <= 0:
            # 退化为 greedy
            idx = logits.argmax(dim=-1).long()
            return [idx.clone() for _ in range(num_samples)]

        logits_f = logits.float().clone()
        mask = torch.ones(logits_f.size(-1), dtype=torch.bool, device=logits_f.device)
        mask[self._aa_token_ids] = False
        logits_f[..., mask] = float("-inf")

        scaled = logits_f / t
        if self.top_k and self.top_k > 0 and self.top_k < scaled.size(-1):
            topk_vals, _ = torch.topk(scaled, self.top_k, dim=-1)
            min_topk = topk_vals[..., -1].unsqueeze(-1)
            scaled = torch.where(scaled < min_topk, torch.full_like(scaled, float("-inf")), scaled)

        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumprobs > self.top_p
            cutoff[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf"))
            scaled.scatter_(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probs, num_samples=num_samples, replacement=True)  # (N,K)
        return [sampled[:, k].long() for k in range(num_samples)]

    @staticmethod
    def _logprob_from_logits(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        非自回归 logp(seq|x)=sum_n log softmax(logits[n])[idx[n]]
        返回标量（tensor）。
        """
        logp = F.log_softmax(logits.float(), dim=-1)  # (N,20)
        token_lp = logp.gather(dim=-1, index=idx.view(-1, 1)).squeeze(-1)  # (N,)
        return token_lp.sum()

    @staticmethod
    def _logprob_batch_from_logits(logits: torch.Tensor, idx_batch: torch.Tensor) -> torch.Tensor:
        """
        批量版 logp：
        - logits: (N,20)
        - idx_batch: (P,N) long
        返回：(P,) 每条序列的 logp(seq|x)=sum_n log p(a_n|x)
        """
        logp = F.log_softmax(logits.float(), dim=-1)  # (N,20)
        logp = logp.unsqueeze(0).expand(idx_batch.size(0), -1, -1)  # (P,N,20)
        token_lp = logp.gather(dim=-1, index=idx_batch.unsqueeze(-1)).squeeze(-1)  # (P,N)
        return token_lp.sum(dim=-1)  # (P,)

    @staticmethod
    def _sft_ce_loss_from_targets(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        SFT 交叉熵 loss（mean）：
        - logits: (N,V)
        - targets: (N,) 或 (P,N)
        返回标量 loss。
        """
        logp = F.log_softmax(logits.float(), dim=-1)  # (N,V)

        if targets.dim() == 1:
            nll = -logp.gather(dim=-1, index=targets.long().view(-1, 1)).squeeze(-1)  # (N,)
            if not nll.numel():
                return logits.sum() * 0.0
            return nll.mean()
        if targets.dim() == 2:
            p, n = int(targets.size(0)), int(targets.size(1))
            if p == 0 or n == 0:
                return logits.sum() * 0.0
            logp_p = logp.unsqueeze(0).expand(p, -1, -1)  # (P,N,V) view
            nll = -logp_p.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)  # (P,N)
            return nll.mean()
        raise ValueError(f"targets must be 1D or 2D, got shape={tuple(targets.shape)}")

    def _dpo_loss_batched(
        self,
        logits_pi: torch.Tensor,
        logits_ref: torch.Tensor,
        idx_chosen: torch.Tensor,
        idx_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPO 的 batched 版本：
        - idx_chosen/idx_rejected: (P,N)
        返回：标量 loss(对 P 取 mean)
        """
        lp_pi_c = self._logprob_batch_from_logits(logits_pi, idx_chosen)      # (P,)
        lp_pi_r = self._logprob_batch_from_logits(logits_pi, idx_rejected)    # (P,)
        with torch.no_grad():
            lp_ref_c = self._logprob_batch_from_logits(logits_ref, idx_chosen)    # (P,)
            lp_ref_r = self._logprob_batch_from_logits(logits_ref, idx_rejected)  # (P,)

        delta = (lp_pi_c - lp_pi_r) - (lp_ref_c - lp_ref_r)  # (P,)
        return (-torch.log(torch.sigmoid(self.dpo_beta * delta))).mean()

    #plddt要大于70；pTM要大于0.5
    def _pass_confidence(self, metrics: Dict[str, float]) -> bool:
        return (metrics.get("pLDDT", 100.0) >= self.min_plddt) and (metrics.get("pTM", 1.0) >= self.min_ptm)

    #以sc得分作为标准的奖励函数，代表了和天然结构的相容性
    def _reward_sc(self, metrics: Dict[str, float]) -> float:
        tm = float(metrics.get("TM", float("nan")))
        rmsd = float(metrics.get("RMSD", float("nan")))
        if not (torch.isfinite(torch.tensor(tm)) and torch.isfinite(torch.tensor(rmsd))):
            return float("-inf")
        rmsd_term = 1.0 / (1.0 + max(0.0, rmsd))
        return self.reward_w_tm * tm + self.reward_w_rmsd_term * rmsd_term
    
    
    #以pLDD和pTM作为标准的奖励函数，代表了模型评估的稳定性
    def _reward_pred(self, metrics: Dict[str, float]) -> float:
        plddt=float(metrics.get("pLDDT", float("nan")))
        ptm=float(metrics.get("pTM", float("nan")))
        if not (torch.isfinite(torch.tensor(plddt)) and torch.isfinite(torch.tensor(ptm))):
            return float("-inf")
        return 0.5*plddt + 0.5*ptm

    def _build_pairs(self, cands: Sequence[Candidate], *, reward_key: str = "reward_sc") -> List[Tuple[int, int]]:
        """
        返回 pairs 的索引对 (i_chosen, i_rejected)。
        """
        n = len(cands)
        if n < 2:
            return []

        # 1) 排序：reward 高的在前
        if reward_key not in {"reward_sc", "reward_pred"}:
            raise ValueError(f"unknown reward_key: {reward_key}")
        order = sorted(range(n), key=lambda i: float(getattr(cands[i], reward_key)), reverse=True)

        # 2) top-k vs bottom-k：用更多组合构造偏好对，增加 num_pairs 的上限与稳定性
        #    经验上 top/bottom 各取 8，可以提供最多 64 个候选对，再按 gap 选前 max_pairs（默认 16）
        k = min(n // 2, 8)
        if k <= 0:
            return []
        top = order[:k]
        bottom = order[-k:]

        cand_pairs: List[Tuple[float, int, int]] = []  # (gap, hi, lo)
        for hi in top:
            for lo in bottom:
                if hi == lo:
                    continue
                gap = float(getattr(cands[hi], reward_key) - getattr(cands[lo], reward_key))
                if gap >= self.pair_margin:
                    cand_pairs.append((gap, hi, lo))

        if not cand_pairs:
            return []

        # gap 越大越“强偏好”，优先使用；最多取 max_pairs
        cand_pairs.sort(key=lambda x: x[0], reverse=True)
        out: List[Tuple[int, int]] = [(hi, lo) for _gap, hi, lo in cand_pairs[: self.max_pairs]]
        return out

    @staticmethod
    def _js_divergence_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        logits_p/logits_q: (N,20)
        返回 JS 的均值（标量）。
        """
        p = torch.softmax(logits_p.float(), dim=-1)
        q = torch.softmax(logits_q.float(), dim=-1)
        m = 0.5 * (p + q)
        kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
        kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
        js = 0.5 * (kl_pm + kl_qm)  # (N,)
        return js.mean()

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits.float(), dim=-1)
        return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()

    @torch.no_grad()
    def ref_ema_update_from_avg(self) -> None:
        """
        ref <- ema(ref, 0.5*(A+B))，不创建临时 teacher，直接做参数更新。
        支持 DDP 包装后的模型：通过 .module 访问原始模型。
        """
        decay = float(self.ema_decay)
        # 如果 model_A/B 是 DDP 包装的，访问 .module 获取原始模型
        model_A_inner = self.model_A.module if hasattr(self.model_A, 'module') else self.model_A
        model_B_inner = self.model_B.module if hasattr(self.model_B, 'module') else self.model_B
        # 只对 LoRA 参数做 EMA（避免对冻结 base 权重做无意义更新）
        # PEFT LoRA 常见命名包含 "lora_"，另外 modules_to_save 也可能是可训练块
        a_named = dict(model_A_inner.named_parameters())
        b_named = dict(model_B_inner.named_parameters())
        for name, p_ref in self.ref_model.named_parameters():
            if "lora" not in name:
                continue
            p_a = a_named.get(name, None)
            p_b = b_named.get(name, None)
            if p_a is None or p_b is None:
                continue
            teacher = 0.5 * (p_a.data + p_b.data)
            p_ref.data.mul_(decay).add_(teacher, alpha=(1.0 - decay))

    def forward(self, 
                batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        返回 dict（训练侧自行决定如何 step 两个 optimizer）：
        - loss_A / loss_B: 给各自模型的 DPO+regularizer
        - js: JS divergence（正数，越大越不一样）
        - repulse_loss: -js（用于最小化 = 最大化 JS）
        - num_pairs / num_kept: 在线产生偏好对与保留候选数
        - best_metrics: 当前 step 最优候选的指标（便于 log）
        """
        if self.oracle is None:
            raise ValueError("SSP_model 需要 oracle 才能在 forward 中做 ESMFold 打分；请在 init 里传入 oracle。")

        # 1) 三个模型 logits（ref 不参与梯度）,ESM3的序列头会输出64个维度，但是实际上只有31个被使用了
        #映射表可以在esm3的constant定义中找到
        logits_A = self._forward_logits(self.model_A, batch_data['structure_coords'],
                                        batch_data['structure_tokens'])  # (N,64)
        logits_B = self._forward_logits(self.model_B, batch_data['structure_coords'],
                                        batch_data['structure_tokens'])  # (N,64)
        with torch.no_grad():
            logits_ref = self._forward_logits(self.ref_model, batch_data['structure_coords'],
                                              batch_data['structure_tokens'])  # (N,64)

        # 2) 采样候选（A/B 各 K）
        idx_list_A = self._sample_indices(logits_A, temperature=self.temp_A, num_samples=self.k_samples)
        idx_list_B = self._sample_indices(logits_B, temperature=self.temp_B, num_samples=self.k_samples)
        idx_list_R: List[torch.Tensor] = []
        if self.include_ref_candidates and self.k_ref_samples > 0:
            # ref 候选不参与梯度（logits_ref 已在 no_grad 中计算）
            idx_list_R = self._sample_indices(logits_ref, temperature=self.temp_ref, num_samples=self.k_ref_samples)

        # 3) oracle 批量评分 + 过滤 + reward（一次性把候选序列列表送进 ESMFold）
        kept: List[Candidate] = []
        best_sc: Optional[Candidate] = None
        best_pred: Optional[Candidate] = None
        rewards_A_sc: List[float] = []
        rewards_B_pred: List[float] = []

        idx_all: List[torch.Tensor] = []
        src_all: List[str] = []
        for idx in idx_list_A:
            idx_all.append(idx)
            src_all.append("A")
        for idx in idx_list_B:
            idx_all.append(idx)
            src_all.append("B")
        for idx in idx_list_R:
            idx_all.append(idx)
            src_all.append("R")

        seq_all: List[str] = [self._idx_to_seq(idx) for idx in idx_all]
        num_total = len(seq_all)

        # oracle 支持 score_batch 
        metrics_batch = self.oracle.score_batch(batch_data, 
                                                seq_all)
        tm_t = metrics_batch["TM"]
        rmsd_t = metrics_batch["RMSD"]
        plddt_t = metrics_batch["pLDDT"]
        ptm_t = metrics_batch["pTM"]
        pae_t = metrics_batch["pAE"]


        def _nanmean(x: torch.Tensor) -> float:
            x = x.detach().float()
            return float(torch.nanmean(x).item()) if x.numel() else float("nan")

        metrics_dict = {
            "cand_num_total": int(num_total),
            "cand_num_pass_conf": int(((plddt_t >= self.min_plddt) & (ptm_t >= self.min_ptm)).sum().item()) if num_total else 0,
            "cand_mean_pLDDT": _nanmean(plddt_t),
            "cand_mean_pTM": _nanmean(ptm_t),
            "cand_mean_pAE": _nanmean(pae_t),
            "cand_mean_RMSD": _nanmean(rmsd_t),
            "cand_mean_TM": _nanmean(tm_t)
        }

        # 逐候选构造 Candidate（这里只做轻量 CPU 逻辑；DPO loss 仍在 GPU 上算 logits）
        for i in range(num_total):
            metrics = {
                "pLDDT": float(plddt_t[i].item()),
                "pTM": float(ptm_t[i].item()),
                "pAE": float(pae_t[i].item()),
                "TM": float(tm_t[i].item()),
                "RMSD": float(rmsd_t[i].item()),
            }
            passed = self._pass_confidence(metrics)
            
            r_sc = self._reward_sc(metrics) if passed else float("nan")
            r_pred = self._reward_pred(metrics) if passed else float("nan")
            if passed and np.isfinite(r_sc) and np.isfinite(r_pred):
                cand = Candidate(
                    idx=idx_all[i],
                    seq=seq_all[i],
                    source=src_all[i],
                    metrics=metrics,
                    reward_sc=float(r_sc),
                    reward_pred=float(r_pred),
                )
                kept.append(cand)
                # A 用 sc 奖励训练，B 用 pred 奖励训练：日志分别按各自目标统计
                if src_all[i] == "A":
                    rewards_A_sc.append(float(r_sc))
                elif src_all[i] == "B":
                    rewards_B_pred.append(float(r_pred))
                if best_sc is None or cand.reward_sc > best_sc.reward_sc:
                    best_sc = cand
                if best_pred is None or cand.reward_pred > best_pred.reward_pred:
                    best_pred = cand


        if len(kept) < 2:
            # 没法构造偏好对：返回 0 loss（trainer 可选择跳过 step）
            js = self._js_divergence_from_logits(logits_A, logits_B)
            repulse_loss = -js
            # 返回“带计算图”的 0，避免训练脚本 backward 报错
            zero_A = logits_A.sum() * 0.0
            zero_B = logits_B.sum() * 0.0
            return {
                "loss_A": zero_A,
                "loss_B": zero_B,
                "js": js.detach(),
                "repulse_loss": repulse_loss.detach(),
                "num_pairs": 0,
                "num_pairs_A": 0,
                "num_pairs_B": 0,
                "num_kept": len(kept),
                "reward_mean_A_sc": float("nan"),
                "reward_mean_B_pred": float("nan"),
                "reward_mean_A": float("nan"),
                "reward_mean_B": float("nan"),
                "reward_best_sc": best_sc.reward_sc if best_sc is not None else float("nan"),
                "reward_best_pred": best_pred.reward_pred if best_pred is not None else float("nan"),
                "reward_best": best_sc.reward_sc if best_sc is not None else float("nan"),
                "reward_gap_mean_A": float("nan"),
                "reward_gap_mean_B": float("nan"),
                "reward_gap_mean": float("nan"),
                "best_metrics_sc": best_sc.metrics if best_sc is not None else None,
                "best_seq_sc": best_sc.seq if best_sc is not None else None,
                "best_metrics": best_sc.metrics if best_sc is not None else None,
                "best_seq": best_sc.seq if best_sc is not None else None,
                "best_metrics_pred": best_pred.metrics if best_pred is not None else None,
                "best_seq_pred": best_pred.seq if best_pred is not None else None,
                # 候选统计（均值/数量）
                **metrics_dict
            }

        # 4) 分别构造偏好对：
        #    - A: reward_sc（结构相容性）
        #    - B: reward_pred（稳定性）
        pairs_A = self._build_pairs(kept, reward_key="reward_sc")
        pairs_B = self._build_pairs(kept, reward_key="reward_pred")
        if len(pairs_A) == 0 and len(pairs_B) == 0:
            js = self._js_divergence_from_logits(logits_A, logits_B)
            repulse_loss = -js
            zero_A = logits_A.sum() * 0.0
            zero_B = logits_B.sum() * 0.0
            return {
                "loss_A": zero_A,
                "loss_B": zero_B,
                "js": js.detach(),
                "repulse_loss": repulse_loss.detach(),
                "num_pairs": 0,
                "num_pairs_A": 0,
                "num_pairs_B": 0,
                "num_kept": len(kept),
                "reward_mean_A_sc": float(sum(rewards_A_sc) / max(1, len(rewards_A_sc))) if len(rewards_A_sc) else float("nan"),
                "reward_mean_B_pred": float(sum(rewards_B_pred) / max(1, len(rewards_B_pred))) if len(rewards_B_pred) else float("nan"),
                "reward_mean_A": float(sum(rewards_A_sc) / max(1, len(rewards_A_sc))) if len(rewards_A_sc) else float("nan"),
                "reward_mean_B": float(sum(rewards_B_pred) / max(1, len(rewards_B_pred))) if len(rewards_B_pred) else float("nan"),
                "reward_best_sc": best_sc.reward_sc if best_sc is not None else float("nan"),
                "reward_best_pred": best_pred.reward_pred if best_pred is not None else float("nan"),
                "reward_best": best_sc.reward_sc if best_sc is not None else float("nan"),
                "reward_gap_mean_A": float("nan"),
                "reward_gap_mean_B": float("nan"),
                "reward_gap_mean": float("nan"),
                "best_metrics_sc": best_sc.metrics if best_sc is not None else None,
                "best_seq_sc": best_sc.seq if best_sc is not None else None,
                "best_metrics": best_sc.metrics if best_sc is not None else None,
                "best_seq": best_sc.seq if best_sc is not None else None,
                "best_metrics_pred": best_pred.metrics if best_pred is not None else None,
                "best_seq_pred": best_pred.seq if best_pred is not None else None,
                **metrics_dict
            }

        # 5) DPO loss：A/B 分别用自己的 pairs
        loss_A = logits_A.sum() * 0.0
        loss_B = logits_B.sum() * 0.0
        reward_gap_mean_A = float("nan")
        reward_gap_mean_B = float("nan")

        # 5.1) SFT loss：优先用偏好对里的 chosen 作为 CE 目标；如果没有 chosen，再 fallback 到真实标签（sequence_tokens）
        true_targets = batch_data.get("sequence_tokens", None)
        if true_targets is not None:
            true_targets = true_targets.to(logits_A.device).long().view(-1)

        sft_loss_A = logits_A.sum() * 0.0
        sft_loss_B = logits_B.sum() * 0.0
        sft_src_A = "none"
        sft_src_B = "none"

        if len(pairs_A) > 0:
            reward_gap_mean_A = float(
                sum(float(kept[i_c].reward_sc - kept[i_r].reward_sc) for i_c, i_r in pairs_A) / max(1, len(pairs_A))
            )
            idx_chosen_A = torch.stack([kept[i_c].idx for i_c, _ in pairs_A], dim=0)    # (P,N)
            idx_rejected_A = torch.stack([kept[i_r].idx for _, i_r in pairs_A], dim=0)  # (P,N)
            loss_A = self._dpo_loss_batched(logits_A, logits_ref, idx_chosen_A, idx_rejected_A)
            sft_loss_A = self._sft_ce_loss_from_targets(logits_A, idx_chosen_A.detach())
            sft_src_A = "pair_chosen"
        elif true_targets is not None and true_targets.numel() == logits_A.size(0):
            sft_loss_A = self._sft_ce_loss_from_targets(logits_A, true_targets.detach())
            sft_src_A = "true_label"

        if len(pairs_B) > 0:
            reward_gap_mean_B = float(
                sum(float(kept[i_c].reward_pred - kept[i_r].reward_pred) for i_c, i_r in pairs_B) / max(1, len(pairs_B))
            )
            idx_chosen_B = torch.stack([kept[i_c].idx for i_c, _ in pairs_B], dim=0)    # (P,N)
            idx_rejected_B = torch.stack([kept[i_r].idx for _, i_r in pairs_B], dim=0)  # (P,N)
            loss_B = self._dpo_loss_batched(logits_B, logits_ref, idx_chosen_B, idx_rejected_B)
            sft_loss_B = self._sft_ce_loss_from_targets(logits_B, idx_chosen_B.detach())
            sft_src_B = "pair_chosen"
        elif true_targets is not None and true_targets.numel() == logits_B.size(0):
            sft_loss_B = self._sft_ce_loss_from_targets(logits_B, true_targets.detach())
            sft_src_B = "true_label"

        # 6) 防同质化：JS divergence
        # 用于日志的 JS（无梯度）
        with torch.no_grad():
            js = self._js_divergence_from_logits(logits_A, logits_B)
        repulse_loss = -js
        
        # 如果启用 JS 损失，分开计算避免 DDP 中梯度交叉
        if self.use_js:
            # 对于 A：让 A 远离当前的 B（B 的 logits detached）
            # 对于 B：让 B 远离当前的 A（A 的 logits detached）
            js_for_A = self._js_divergence_from_logits(logits_A, logits_B.detach())
            js_for_B = self._js_divergence_from_logits(logits_A.detach(), logits_B)
            repulse_loss_A = -js_for_A
            repulse_loss_B = -js_for_B
        else:
            repulse_loss_A = logits_A.new_tensor(0.0)
            repulse_loss_B = logits_B.new_tensor(0.0)

        # 7) 可选：给 A/B 加熵奖励（最小化 -H == 最大化 H）
        entropy_A = self._entropy_from_logits(logits_A)
        entropy_loss_A = -self.entropy_bonus_A * entropy_A if self.entropy_bonus_A > 0 else logits_A.new_tensor(0.0)
        entropy_B = self._entropy_from_logits(logits_B)
        entropy_loss_B = -self.entropy_bonus_B * entropy_B if self.entropy_bonus_B > 0 else logits_B.new_tensor(0.0)

        # 关键：loss_A 只依赖 model_A，loss_B 只依赖 model_B
        # 这样分开 backward 时不会有梯度交叉问题
        total_A = loss_A + self.lambda_sft * sft_loss_A + self.lambda_js * repulse_loss_A + entropy_loss_A
        total_B = loss_B + self.lambda_sft * sft_loss_B + self.lambda_js * repulse_loss_B + entropy_loss_B

        return {
            "loss_A": total_A,
            "loss_B": total_B,
            "dpo_loss_A": loss_A.detach(),
            "dpo_loss_B": loss_B.detach(),
            "sft_loss_A": sft_loss_A.detach(),
            "sft_loss_B": sft_loss_B.detach(),
            "sft_src_A": sft_src_A,
            "sft_src_B": sft_src_B,
            "js": js.detach(),
            "repulse_loss": repulse_loss.detach(),
            "entropy_A": entropy_A.detach(),
            "entropy_B": entropy_B.detach(),
            "num_pairs": int(len(pairs_A) + len(pairs_B)),
            "num_pairs_A": int(len(pairs_A)),
            "num_pairs_B": int(len(pairs_B)),
            "num_kept": len(kept),
            "reward_mean_A_sc": float(sum(rewards_A_sc) / max(1, len(rewards_A_sc))) if len(rewards_A_sc) else float("nan"),
            "reward_mean_B_pred": float(sum(rewards_B_pred) / max(1, len(rewards_B_pred))) if len(rewards_B_pred) else float("nan"),
            "reward_mean_A": float(sum(rewards_A_sc) / max(1, len(rewards_A_sc))) if len(rewards_A_sc) else float("nan"),
            "reward_mean_B": float(sum(rewards_B_pred) / max(1, len(rewards_B_pred))) if len(rewards_B_pred) else float("nan"),
            "reward_best_sc": best_sc.reward_sc if best_sc is not None else float("nan"),
            "reward_best_pred": best_pred.reward_pred if best_pred is not None else float("nan"),
            "reward_best": best_sc.reward_sc if best_sc is not None else float("nan"),
            "reward_gap_mean_A": reward_gap_mean_A,
            "reward_gap_mean_B": reward_gap_mean_B,
            "reward_gap_mean": reward_gap_mean_A,
            "best_metrics_sc": best_sc.metrics if best_sc is not None else None,
            "best_seq_sc": best_sc.seq if best_sc is not None else None,
            "best_metrics": best_sc.metrics if best_sc is not None else None,
            "best_seq": best_sc.seq if best_sc is not None else None,
            "best_metrics_pred": best_pred.metrics if best_pred is not None else None,
            "best_seq_pred": best_pred.seq if best_pred is not None else None,
            **metrics_dict
        }


def bulid_ESM3_model4LoRA(model:ESM3,
                          fine_tune_layer_num,
                          LoRA_config:Dict[str,Any],
                          model_type:str='lora'):
    total_layers=48
    if model_type=='base':
        for name, param in model.named_parameters():
            if 'encoder' in name:param.requires_grad=False
            if name.startswith('transformer.blocks'):
                match = re.search(r'\.([0-9]+)\.', name)
                layer_num=int(match.group(1))
                if layer_num<total_layers-fine_tune_layer_num:param.requires_grad=False
    elif model_type=='lora':
        update_num = total_layers - fine_tune_layer_num
        target_layers = [f'transformer.blocks.{i}.attn.layernorm_qkv.1' for i in range(update_num, total_layers)] + \
                        [f'transformer.blocks.{i}.attn.out_proj' for i in range(update_num, total_layers)] + \
                        [f'transformer.blocks.{i}.ffn.1' for i in range(update_num, total_layers)] + \
                        [f'transformer.blocks.{i}.ffn.3' for i in range(update_num, total_layers)] + \
                        ['output_heads.sequence_head.0', 'output_heads.sequence_head.3'] 
                               
        lora_config = LoraConfig(
            r=LoRA_config['r'],
            # task_type=TaskType.TOKEN_CLS,
            lora_alpha=LoRA_config['lora_alpha'],
            lora_dropout=LoRA_config['lora_dropout'],
            target_modules=target_layers,
            bias="lora_only"
        )

        model = get_peft_model(model, lora_config).to(torch.bfloat16)
        model.print_trainable_parameters()

        print('Transfer LoRA model Over')
    else:
        raise ValueError(f'Model type {model_type} not supported!')
    return model


if __name__ == '__main__':
    model=ESM3.from_pretrained("esm3_sm_open_v1").to('cuda:0')
    model=bulid_ESM3_model4LoRA(model,16,LoRA_config={'r':8,'lora_alpha':32,'lora_dropout':0.2})
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)