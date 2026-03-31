"""
DDP 版本的 SSP-ESM3 训练脚本。
支持多 GPU 并行训练，每个进程持有完整的 model_A, model_B, ref_model 和 ESMFold Oracle。

启动方式:
    torchrun --nproc_per_node=2 SSP_ESM3_train_ddp.py
    或
    python -m torch.distributed.launch --nproc_per_node=2 SSP_ESM3_train_ddp.py
"""

import sys
import os
import time
import json
import numpy as np
import logging
from contextlib import nullcontext
import wandb
import hydra
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from transformers import EsmForProteinFolding
from peft import PeftModel
import utils
from utils import set_seed
from ESM3RL_DataSet import ESM3RL_DataSet, collate_fn
from SSP_ESM3 import SSP_model, ESMFoldOracle, bulid_ESM3_model4LoRA
from esm.tokenization import get_esm3_model_tokenizers

tokenizer_collection = get_esm3_model_tokenizers()
logger = logging.getLogger(__name__)

def _is_ddp_enabled(world_size: int) -> bool:
    """world_size>1 且进程组已初始化才认为 DDP 真正启用。"""
    return int(world_size) > 1 and dist.is_available() and dist.is_initialized()

def _unwrap_model(m):
    """兼容 DDP 与非 DDP：返回原始模型。"""
    return m.module if hasattr(m, "module") else m


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 单卡模式的 fallback
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, local_rank, device


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否是主进程（rank 0）"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_dataloaders(cfg: DictConfig, rank: int, world_size: int):
    """创建支持 DDP 的 DataLoader"""
    max_length = getattr(cfg.dataset, 'max_length', None)
    debug_samples = getattr(cfg.dataset, 'debug_samples', None)
    
    train_dataset = ESM3RL_DataSet(
        cfg.dataset.train_dir,
        cfg.dataset.structure_token_dir,
        max_length=max_length,
        max_samples=debug_samples
    )
    test_dataset = ESM3RL_DataSet(
        cfg.dataset.test_dir,
        cfg.dataset.structure_token_dir,
        max_length=max_length,
        max_samples=debug_samples
    )

    if is_main_process():
        print(
            f"Train on CATH dataset with {len(train_dataset)} training data, "
            f"{len(test_dataset)} test data (no val). max_length={max_length}"
        )
    
    batch_size = 1  # 保持 batch_size=1，每个进程处理一个样本
    num_workers = int(getattr(cfg.train, "num_workers", 4))

    # 使用 DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # 丢弃不完整的 batch，避免进程间数据量不一致
    ) if world_size > 1 else None
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # 如果用了 sampler 就不能 shuffle
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader, train_sampler


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    esmfold_model,
    device,
    pdb_dir:str,
    eval_num: int = 0,
    split: str = "test",
    skip_esmfold: bool = False,
):
    """
    评估函数，与原版保持一致。
    注意：在 DDP 下每个进程独立评估自己的数据子集，
    如果需要汇总结果，可以在调用后手动 all_gather。
    
    Args:
        skip_esmfold: 如果为 True，跳过 ESMFold 评估（节省显存）
    """
    model.eval()

    total_loss_sum = 0.0
    total_res = 0
    total_correct = 0
    seq_recoveries = []
    pTM_list = []
    sc_RMSD_list = []
    sc_TM_list = []
    pLDDT_list = []
    pAE_list = []
    gen_seq = {}
    

    index = 0
    pbar = tqdm(dataloader, desc=f"eval {split}", leave=False, disable=not is_main_process())
    for batch_data in pbar:
        index += 1
        aa_label = batch_data["sequence_tokens"].to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 如果 model 是 DDP，需要访问 .module
            actual_model = _unwrap_model(model)
            outputs = actual_model(
                structure_tokens=batch_data["structure_tokens"].to(device),
                structure_coords=batch_data["structure_coords"].to(device),
            )
        logits = outputs.sequence_logits if hasattr(outputs, "sequence_logits") else outputs
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        pred_idx = logits.argmax(dim=-1)
        seq = tokenizer_collection.sequence.decode(pred_idx, skip_special_tokens=True)
        seq = seq.replace(" ", "")
        sequence = [seq]

        # ESMFold 评估（可选，需要大量显存）        
        if not skip_esmfold:
            tm_t, rmsd_t, plddt_t, ptm_t, pae_t = utils.calculate_metrics(
                true_pdb_path=os.path.join(pdb_dir, batch_data['pdb_id'] + '.pdb'),
                gen_sequences=sequence,
                esmfold_model=esmfold_model,
            )
            # 每次 ESMFold 推理后清理缓存
            torch.cuda.empty_cache()
            
            if tm_t is not None:
                pTM_list.append(float(ptm_t[0].item()))
                pLDDT_list.append(float(plddt_t[0].item()))
                pAE_list.append(float(pae_t[0].item()))
                sc_TM_list.append(float(tm_t[0].item()))
                sc_RMSD_list.append(float(rmsd_t[0].item()))

        gen_seq[batch_data['pdb_id']] = sequence[0]
        loss = F.cross_entropy(logits.float(), aa_label, reduction="mean")
        pred = logits.argmax(dim=-1)
        correct = (pred == aa_label).sum().item()
        seq_recoveries.append(float(correct / aa_label.numel()))

        n = int(aa_label.numel())
        total_loss_sum += float(loss.item()) * n
        total_res += n
        total_correct += int(correct)

        if eval_num > 0 and index == eval_num:
            break

    mean_loss = total_loss_sum / max(1, total_res)
    residue_acc = total_correct / max(1, total_res)
    mean_seq_recovery = float(sum(seq_recoveries) / max(1, len(seq_recoveries)))

    return {
        "loss": mean_loss,
        "residue_acc": residue_acc,
        "mean_seq_recovery": mean_seq_recovery,
        "num_res": total_res,
        "num_seq": len(seq_recoveries),
        "pTM": np.nanmean(pTM_list) if pTM_list else float("nan"),
        "pLDDT": np.nanmean(pLDDT_list) if pLDDT_list else float("nan"),
        "pAE": np.nanmean(pAE_list) if pAE_list else float("nan"),
        "sc_TM": np.nanmean(sc_TM_list) if sc_TM_list else float("nan"),
        "sc_RMSD": np.nanmean(sc_RMSD_list) if sc_RMSD_list else float("nan"),
        "test_gen_seq": gen_seq,
    }


def gather_metrics(metrics: dict, world_size: int) -> dict:
    """
    跨进程汇总评估指标（简单版：求均值）
    """
    if world_size <= 1 or not dist.is_initialized():
        return metrics
    
    # 汇总数值型指标
    keys_to_gather = ["loss", "residue_acc", "mean_seq_recovery", "pTM", "pLDDT", "pAE", "sc_TM", "sc_RMSD"]
    gathered = {}
    
    for key in keys_to_gather:
        val = metrics.get(key, 0.0)
        if np.isnan(val):
            val = 0.0
        tensor = torch.tensor([val], device='cuda')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        gathered[key] = tensor.item() / world_size
    
    # 直接复制其他字段
    for key in metrics:
        if key not in keys_to_gather:
            gathered[key] = metrics[key]
    
    return gathered


@hydra.main(version_base=None, config_path="./", config_name="esm3_config")
def main(cfg: DictConfig):
    # 1. 初始化分布式
    rank, world_size, local_rank, device = setup_distributed()
    ddp_enabled = _is_ddp_enabled(world_size)
    
    # 设置随机种子（每个进程用不同种子以获得不同采样）
    set_seed(seed=1024 + rank)
    
    if is_main_process():
        print(f"=== DDP Training with {world_size} GPUs ===")
        print(OmegaConf.to_yaml(cfg))
    
        # 2. 输出目录和日志（只在主进程）
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Output directory: {output_dir}")
        logger.info(f"Logger initialized in {output_dir}/train.log")
        
        wandb.init(
            project="SSP-ESM3-DDP",
            name=f"SSP-ESM3-DDP-{world_size}GPU-{time.strftime('%Y%m%d%H%M%S')}",
            config={
                "epochs": cfg.train.train_epochs,
                "world_size": world_size,
                "max_length": getattr(cfg.dataset, 'max_length', None),
            }
        )

    # 3. 创建数据加载器
    train_loader, test_loader, train_sampler = get_dataloaders(cfg, rank, world_size)

    # 4. 构建模型
    def _build_base_model(LoRA_Dir):
        from esm.models.esm3 import ESM3
        LoRA_config = {'r': 8, 'lora_alpha': 32, 'lora_dropout': 0.2}
        base_esm3 = ESM3.from_pretrained("esm3_sm_open_v1")
        
        if LoRA_Dir is not None:
            LoRA_ESM3 = PeftModel.from_pretrained(base_esm3, LoRA_Dir,is_trainable=True)
            print('Load existing LoRA model from:', LoRA_Dir)
        else:
            print('Build new LoRA model')
            LoRA_ESM3 = bulid_ESM3_model4LoRA(
                model=base_esm3,
                fine_tune_layer_num=cfg.model.fine_tune_layer_num,
                LoRA_config=LoRA_config,
                model_type='lora'
            )
        return LoRA_ESM3

    # 构建三个模型
    model_A = _build_base_model(cfg.model.model_A_LoRA_weight_dir).to(device)
    model_B = _build_base_model(cfg.model.model_B_LoRA_weight_dir).to(device)
    ref_model = _build_base_model(cfg.model.Ref_Model_LoRA_weight_dir).to(device)

    # 5. DDP 包装（仅在 torchrun 多卡时启用；python 单卡不走 DDP）
    # 注意：SSP_model 内部如果使用 DDP 模型可以触发梯度同步；单卡时直接用原始模型即可。
    if ddp_enabled:
        find_unused = True  # LoRA 模型需要
        model_A_ddp = DDP(model_A, device_ids=[local_rank], find_unused_parameters=find_unused)
        model_B_ddp = DDP(model_B, device_ids=[local_rank], find_unused_parameters=find_unused)
        # 设置静态图：告诉 DDP 计算图不会在训练过程中改变
        model_A_ddp._set_static_graph()
        model_B_ddp._set_static_graph()
    else:
        model_A_ddp = model_A
        model_B_ddp = model_B

    # 6. 构建 ESMFold Oracle（每个进程独立持有）
    esmfold_model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1"
    ).to(device)
    esmfold_model.eval()

    oracle = ESMFoldOracle(
        esmfold_model=esmfold_model,
        pdb_dir=cfg.dataset.train_dir
    )

    # 7. 构建 SSP_model
    # 关键：传入 DDP 包装后的模型，这样 forward 时才能触发 DDP 的梯度同步
    ssp_model = SSP_model(
        model_A=model_A_ddp,  # 使用 DDP 包装后的模型
        model_B=model_B_ddp,  # 使用 DDP 包装后的模型
        ref_model=ref_model,   # ref 不需要梯度，不用 DDP
        oracle=oracle,
        use_js=cfg.ssp.use_js,
        temp_A=1,
        temp_B=1,
        temp_ref=0.8,
        k_samples=6,
        k_ref_samples=5,
        entropy_bonus_A=cfg.ssp.entropy_bonus_A,
        entropy_bonus_B=cfg.ssp.entropy_bonus_B
    )  # 不需要 .to(device)，因为子模型已经在 device 上

    # 8. 优化器（注意：用原始模型的 parameters，不是 DDP 包装后的）
    lr = float(cfg.train.lr)
    weight_decay = float(getattr(cfg.train, "weight_decay", 0.01))
    clip_grad_norm = float(getattr(cfg.train, "clip_grad_norm", 1.0))
    epochs = int(cfg.train.train_epochs)

    # model_A_ddp.module 就是原始的 model_A
    
    optimizer_A = AdamW(filter(lambda p: p.requires_grad, _unwrap_model(model_A_ddp).parameters()), lr=lr, weight_decay=weight_decay)
    optimizer_B = AdamW(filter(lambda p: p.requires_grad, _unwrap_model(model_B_ddp).parameters()), lr=lr, weight_decay=weight_decay)

    # 9. 训练状态
    best_test_A_rmsd = float("inf")
    best_test_B_rmsd = float("inf")
    log_step_interval = int(getattr(cfg.train, "log_step_interval", 10))
    
    global_forward = 0
    global_backward = 0
    global_optim_step = 0
    global_ref_ema_step = 0

    update_every = int(getattr(cfg.train, "update_every", 12))
    ref_update_every = int(getattr(cfg.train, "ref_update_every", 10))
    accum_backward = 0

    optimizer_A.zero_grad(set_to_none=True)
    optimizer_B.zero_grad(set_to_none=True)

    # 10. 训练循环
    for epoch in range(1, epochs + 1):
        ssp_model.train()
        
        # DDP: 每个 epoch 设置 sampler 的 epoch，确保 shuffle 正确
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        sums = {
            "loss_A": 0.0,
            "loss_B": 0.0,
            "reward_mean_A": 0.0,
            "reward_mean_B": 0.0,
            "reward_best": 0.0,
            "js": 0.0,
            "num_pairs": 0.0,
            "num_kept": 0.0,
        }
        reward_count_A = 0
        reward_count_B = 0
        reward_best_count = 0
        epoch_iter = 0
        valid_backward_count = 0  # 有效 backward 计数

        pbar = tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch}/{epochs}", 
                    disable=not is_main_process())
        
        for batch_data in pbar:
            batch_data['structure_coords'] = batch_data['structure_coords'].to(device)
            batch_data['structure_tokens'] = batch_data['structure_tokens'].to(device)
            
            # Forward
            out = ssp_model(batch_data)
            loss_A = out["loss_A"]
            loss_B = out["loss_B"]
            global_forward += 1
            epoch_iter += 1

            # 准备日志数据
            logout = {}
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    logout[f"train_{k}"] = float(v.detach().item())
                elif isinstance(v, (float, int)):
                    logout[f"train_{k}"] = float(v)

            # 只在主进程记录 wandb
            if is_main_process():
                logout.update({
                    "train_global_forward": float(global_forward * world_size),  # 全局 forward 次数
                    "train_global_backward": float(global_backward * world_size),
                    "train_global_optim_step": float(global_optim_step),
                    "train_global_ref_ema_step": float(global_ref_ema_step),
                    "train_epoch_iter": float(epoch_iter),
                })
                wandb.log({
                    "epoch": epoch,
                    "global_forward": global_forward * world_size,
                    "global_optim_step": global_optim_step,
                    **logout
                }, step=global_forward * world_size)

            num_pairs = int(out.get("num_pairs", 0))
            
            # 决定是否需要同步梯度
            # 在梯度累积期间用 no_sync() 跳过同步，最后一次累积时正常同步
            is_last_accum = ((accum_backward + 1) % update_every == 0) if num_pairs > 0 else False
            
            # 使用 DDP 的 no_sync 上下文管理器
            # 单卡/非 DDP 时没有 no_sync()
            can_no_sync_A = hasattr(model_A_ddp, "no_sync")
            can_no_sync_B = hasattr(model_B_ddp, "no_sync")
            sync_context_A = nullcontext() if (is_last_accum or not ddp_enabled or not can_no_sync_A) else model_A_ddp.no_sync()
            sync_context_B = nullcontext() if (is_last_accum or not ddp_enabled or not can_no_sync_B) else model_B_ddp.no_sync()
            
            # 关键：分开 backward，避免 DDP 中同一参数被标记两次
            # loss_A 只依赖 model_A，loss_B 只依赖 model_B
            with sync_context_A:
                (loss_A / update_every).backward()
            with sync_context_B:
                (loss_B / update_every).backward()
            
            if num_pairs > 0:
                global_backward += 1
                accum_backward += 1
                valid_backward_count += 1

                # 累积够 update_every 次有效 backward 再更新
                if accum_backward % update_every == 0:
                    # 调试：检查梯度是否存在
                    if is_main_process() and global_optim_step < 3:
                        base_A = _unwrap_model(model_A_ddp)
                        base_B = _unwrap_model(model_B_ddp)
                        grad_norm_A = sum(p.grad.norm().item() for p in base_A.parameters() if p.grad is not None)
                        grad_norm_B = sum(p.grad.norm().item() for p in base_B.parameters() if p.grad is not None)
                        num_grad_A = sum(1 for p in base_A.parameters() if p.grad is not None)
                        num_grad_B = sum(1 for p in base_B.parameters() if p.grad is not None)
                        print(f"[DEBUG] Before step {global_optim_step}: grad_norm_A={grad_norm_A:.4f} ({num_grad_A} params), grad_norm_B={grad_norm_B:.4f} ({num_grad_B} params)", flush=True)
                    
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(_unwrap_model(model_A_ddp).parameters(), float(clip_grad_norm))
                        torch.nn.utils.clip_grad_norm_(_unwrap_model(model_B_ddp).parameters(), float(clip_grad_norm))

                    optimizer_A.step()
                    optimizer_B.step()
                    optimizer_A.zero_grad(set_to_none=True)
                    optimizer_B.zero_grad(set_to_none=True)
                    global_optim_step += 1

                    # ref EMA 更新
                    # 注意：因为 A/B 的参数在 DDP 下是同步的，所以每个进程独立做 EMA 结果一致
                    if ref_update_every > 0 and (global_optim_step % ref_update_every == 0):
                        ssp_model.ref_ema_update_from_avg()
                        global_ref_ema_step += 1

                # 统计
                sums["loss_A"] += float(loss_A.item())
                sums["loss_B"] += float(loss_B.item())
                rA = float(out.get("reward_mean_A", float("nan")))
                rB = float(out.get("reward_mean_B", float("nan")))
                rbest = float(out.get("reward_best", float("nan")))
                if np.isfinite(rA):
                    sums["reward_mean_A"] += rA
                    reward_count_A += 1
                if np.isfinite(rB):
                    sums["reward_mean_B"] += rB
                    reward_count_B += 1
                if np.isfinite(rbest):
                    sums["reward_best"] += rbest
                    reward_best_count += 1
                sums["js"] += float(out.get("js", 0.0))
                sums["num_pairs"] += float(out.get("num_pairs", 0))
                sums["num_kept"] += float(out.get("num_kept", 0))
                        

            if is_main_process():
                pbar.set_postfix({
                    "loss_A": sums["loss_A"] / max(1, valid_backward_count),
                    "loss_B": sums["loss_B"] / max(1, valid_backward_count),
                    "r_best": sums["reward_best"] / max(1, reward_best_count),
                    "pairs": sums["num_pairs"] / max(1, epoch_iter),
                })

            if is_main_process() and global_forward % log_step_interval == 0:
                logger.info(
                    f"epoch {epoch}, iter {epoch_iter}, |"
                    f"global_forward: {global_forward * world_size}, |"
                    f"global_backward: {global_backward * world_size}, |"
                    f"global_optim_step: {global_optim_step}, |"
                    f"global_ref_ema_step: {global_ref_ema_step}, |"
                    f"Loss_A: {sums['loss_A'] / max(1, valid_backward_count):.4f}, |"
                    f"Loss_B: {sums['loss_B'] / max(1, valid_backward_count):.4f}, |"
                    f"Reward_best: {sums['reward_best'] / max(1, reward_best_count):.4f}, |"
                    f"Pairs(avg/iter): {sums['num_pairs'] / max(1, epoch_iter)}"
                )
                

        # Epoch 结束前：同步 CUDA 操作并清理缓存
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Epoch 结束：保存模型（只在主进程）
        if is_main_process():
            train_metrics = {
                "loss_A": sums["loss_A"] / max(1, valid_backward_count),
                "loss_B": sums["loss_B"] / max(1, valid_backward_count),
                "reward_mean_A": sums["reward_mean_A"] / max(1, reward_count_A),
                "reward_mean_B": sums["reward_mean_B"] / max(1, reward_count_B),
                "reward_best": sums["reward_best"] / max(1, reward_best_count),
                "js": sums["js"] / max(1, epoch_iter),
                "num_pairs": sums["num_pairs"] / max(1, epoch_iter),
                "num_kept": sums["num_kept"] / max(1, epoch_iter),
            }

            # 保存模型（使用 DDP 包装后的 .module 访问原始模型）
            print(f"Saving models for epoch {epoch}...")
            _unwrap_model(model_A_ddp).save_pretrained(os.path.join(output_dir, f"model_A_epoch{epoch}"))
            _unwrap_model(model_B_ddp).save_pretrained(os.path.join(output_dir, f"model_B_epoch{epoch}"))
            ref_model.save_pretrained(os.path.join(output_dir, f"ref_model_epoch{epoch}"))
            print(f"Models saved for epoch {epoch}")

        # 同步所有进程，确保保存完成
        if ddp_enabled:
            dist.barrier()

        # 评估前清理显存
        torch.cuda.empty_cache()
        
        # 评估（每个进程独立评估自己的数据子集，然后汇总）
        # 传入 DDP 模型，evaluate 内部会通过 .module 访问原始模型
        
        skip_esmfold = getattr(cfg.dataset, 'skip_esmfold_eval', False)
        print('skip_esmfold_eval:', skip_esmfold, flush=True)
        print(f"[rank:{rank}]-->Evaluating {cfg.dataset.debug_eva_samples} samples for A and B (skip_esmfold={skip_esmfold})", flush=True)
        
        test_metrics_A = evaluate(model_A_ddp, test_loader, esmfold_model, device,cfg.dataset.pdb_chain_dir+os.sep+'test',
                                  eval_num=cfg.dataset.debug_eva_samples, skip_esmfold=skip_esmfold)
        torch.cuda.empty_cache()
        test_metrics_B = evaluate(model_B_ddp, test_loader, esmfold_model, device,cfg.dataset.pdb_chain_dir+os.sep+'test',
                                  eval_num=cfg.dataset.debug_eva_samples, skip_esmfold=skip_esmfold)

        # 汇总评估结果
        test_metrics_A = gather_metrics(test_metrics_A, world_size)
        test_metrics_B = gather_metrics(test_metrics_B, world_size)

        if is_main_process():
            gen_seq_A = test_metrics_A.get("test_gen_seq", {})
            gen_seq_B = test_metrics_B.get("test_gen_seq", {})
            
            # 移除 gen_seq 字段以便写入 CSV
            test_metrics_A_csv = {k: v for k, v in test_metrics_A.items() if k != "test_gen_seq"}
            test_metrics_B_csv = {k: v for k, v in test_metrics_B.items() if k != "test_gen_seq"}

            row = {"epoch": epoch}
            row.update(utils._prefix_metrics("train_", train_metrics))
            row.update(utils._prefix_metrics("test_A_", test_metrics_A_csv))
            row.update(utils._prefix_metrics("test_B_", test_metrics_B_csv))
            utils.append_metrics_csv(output_dir, row)

            # 保存最佳生成序列
            if float(test_metrics_A.get("sc_RMSD", float("inf"))) < best_test_A_rmsd:
                best_test_A_rmsd = float(test_metrics_A["sc_RMSD"])
                with open(os.path.join(output_dir, "best_test_A_gen_seq.json"), "w") as f:
                    json.dump(gen_seq_A, f)

            if float(test_metrics_B.get("sc_RMSD", float("inf"))) < best_test_B_rmsd:
                best_test_B_rmsd = float(test_metrics_B["sc_RMSD"])
                with open(os.path.join(output_dir, "best_test_B_gen_seq.json"), "w") as f:
                    json.dump(gen_seq_B, f)

            logger.info(
                f"Epoch {epoch} done. "
                f"Test A: sc_TM={test_metrics_A.get('sc_TM', 0):.4f}, sc_RMSD={test_metrics_A.get('sc_RMSD', 0):.4f} | "
                f"Test B: sc_TM={test_metrics_B.get('sc_TM', 0):.4f}, sc_RMSD={test_metrics_B.get('sc_RMSD', 0):.4f}"
            )

        # 同步所有进程
        if ddp_enabled:
            dist.barrier()

    # 训练结束
    if is_main_process():
        wandb.finish()
        logger.info("Training completed!")

    cleanup_distributed()


if __name__ == "__main__":
    main()

