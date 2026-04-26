import torch
from torch.nn import functional as F
from torch import sin, cos, atan2, acos
import os
import numpy as np
import math
import torch.distributed as dist
import random
import csv
from tmtools import tm_align
from esm.tokenization import get_esm3_model_tokenizers
from tmtools.io import get_structure, get_residue_data

tokenizer_collection=get_esm3_model_tokenizers()

def set_seed(seed=1024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers


@torch.no_grad()
def compute_ptm_per_sample(
    ptm_logits: torch.Tensor,
    residue_weights: torch.Tensor | None = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    计算每个样本一个 pTM（形状 [B]），用于对比 transformers 内置 compute_tm()

    ptm_logits: [B, L, L, no_bins]
    """
    if residue_weights is None:
        # [L]
        residue_weights = ptm_logits.new_ones(ptm_logits.shape[-2])

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=ptm_logits.device)
    bin_centers = _calculate_bin_centers(boundaries)  # [no_bins]

    n = ptm_logits.shape[-2]
    clipped_n = max(int(n), 19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(ptm_logits, dim=-1)  # [B, L, L, no_bins]
    tm_per_bin = 1.0 / (1.0 + (bin_centers**2) / (d0**2))     # [no_bins]
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [B, L, L]

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())  # [L]
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)  # [B, L]

    weighted = per_alignment * residue_weights  # [B, L]
    best_i = weighted.argmax(dim=-1)            # [B]
    b_idx = torch.arange(per_alignment.shape[0], device=per_alignment.device)
    return per_alignment[b_idx, best_i]         # [B]



@torch.no_grad()
def logits2AA_seq(
    pred_logits: torch.Tensor,
    batch: torch.Tensor | None = None,
    decode_method: str = "greedy",
    temperature: float = 1.0,
):
    """
    将模型输出的 residue-level logits 解码为 AA 序列字符串。

    Args:
        pred_logits: (N, 20) float tensor
        batch: (N,) long tensor，表示每个残基属于哪条序列；若为 None，则视为单条序列
        decode_method: 'greedy' 或 'temperature'
        temperature: 温度采样用，>0；越大越随机，越小越接近 argmax

    Returns:
        seqs: list[str]，按 batch id 从小到大排列的序列
        idxs: list[torch.Tensor]，每条序列对应的 (L,) 预测类别索引
    """
    if pred_logits.dim() != 2 or pred_logits.size(-1) != 20:
        raise ValueError(f"pred_logits must be (N,20), got {tuple(pred_logits.shape)}")

    decode_method = decode_method.lower()
    if decode_method not in {"greedy", "temperature"}:
        raise ValueError("decode_method must be 'greedy' or 'temperature'")

    if decode_method == "greedy" or temperature is None or float(temperature) <= 0:
        pred_idx = pred_logits.argmax(dim=-1)
    else:
        t = float(temperature)
        probs = torch.softmax(pred_logits / t, dim=-1)
        pred_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

    if batch is None:
        batch = torch.zeros((pred_idx.size(0),), dtype=torch.long, device=pred_idx.device)

    seqs: list[str] = []
    idxs: list[torch.Tensor] = []
    for b in batch.unique(sorted=True):
        sel = (batch == b)
        idx = pred_idx[sel]
        seq=tokenizer_collection.sequence.decode(idx)
        seq = seq.replace(' ','').replace('<cls>','').replace('<eos>','')
        seqs.append(seq)
        idxs.append(idx)
    return seqs, idxs


@torch.no_grad()
def refolding_structure(esmfold_model,
                        sequence:str
                        ):
    #推理长度：180
    #大于180，用bf16推理,否则用普通的
    with torch.no_grad(),torch.amp.autocast(device_type="cuda",
                                            dtype=torch.bfloat16):
        output=esmfold_model.infer(sequence)#8*batch_size*seq_len*14*3
    
    #batch*seq_len*3
    ca=output.positions[-1,:, :,1, :].detach().cpu().numpy()
    
    #batch
    plddt=output.plddt.mean((1,2)).detach().cpu()

    #batch
    pae=output.predicted_aligned_error.mean((1,2)).detach().cpu()
    
    #batch
    ptm = compute_ptm_per_sample(output.ptm_logits).detach().cpu()
 
    return ca,plddt,ptm,pae


#不读pdb文件，直接从坐标计算
def calculate_metrics_from_coord(true_seq,
                                 gen_seq_list,
                                 true_coord,
                                 esmfold_model):
    tm_scores=[]
    rmsds=[]
    try:
        pred_ca_coord,plddt,ptm,pae=refolding_structure(esmfold_model,gen_seq_list)       
        for one_seq in gen_seq_list:
            if len(one_seq)!=true_coord.shape[0]:
                print(f"Sequence length mismatch: true_coord.shape[0]={true_coord.shape[0]}, len(one_seq)={len(one_seq)}")
        
        #只取Ca原子的坐标
        #esmfold_coords_Ca=pred_coord[:,1,:]
        for i,one_sequence in enumerate(gen_seq_list):
            tm_score = tm_align(true_coord,pred_ca_coord[i], true_seq,one_sequence)
            tm_scores.append(tm_score.tm_norm_chain1)
            rmsds.append(tm_score.rmsd)

    except Exception as e:
        print(f"Error calculating TM-score and RMSD: {e}")
        return None,None,None,None,None
    #batch,batch,batch,batch,batch
    return torch.tensor(tm_scores), torch.tensor(rmsds),plddt,ptm,pae
    


def calculate_metrics(true_pdb_path:str,
                                 gen_sequences:list[str],
                                 esmfold_model,
                                 pdb_id:str=''
                                 ):
    tm_scores=[]
    rmsds=[]
    try:
        pred_ca_coord,plddt,ptm,pae=refolding_structure(esmfold_model,gen_sequences)
        true_structure = get_structure(true_pdb_path)
        true_chain = next(true_structure.get_chains())
        true_coords, true_seq = get_residue_data(true_chain)
        #只取Ca原子的坐标
        #esmfold_coords_Ca=pred_coord[:,1,:]
        for i,one_sequence in enumerate(gen_sequences):
            tm_score = tm_align(true_coords,pred_ca_coord[i], true_seq,one_sequence)
            tm_scores.append(tm_score.tm_norm_chain1)
            rmsds.append(tm_score.rmsd)

    except Exception as e:
        #print(f"Error calculating TM-score and RMSD: {e}")
        # NOTE: 不依赖 torch.distributed，保证单进程/未初始化进程组时也不会在异常路径崩溃
        print(f"Error calculating TM-score and RMSD: {e}")
        print(f"true_pdb_path: {true_pdb_path}")
        if gen_sequences:
            print(f"sequences length: {len(gen_sequences[0])}")
        return None,None,None,None,None
    #batch,batch,batch,batch,batch
    return torch.tensor(tm_scores), torch.tensor(rmsds),plddt,ptm,pae


def _prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def append_metrics_csv(
    output_dir: str,
    row: dict,
    filename: str = "metrics_by_epoch.csv",
    float_digits: int = 4,
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # 稳定列顺序：epoch 相关放前面，其余按字母序
    keys = list(row.keys())
    front = [k for k in ["epoch", "recycle_steps"] if k in row]
    rest = sorted([k for k in keys if k not in set(front)])
    fieldnames = front + rest

    # 写入时统一格式化 float，保持 CSV 展示为固定小数位（例如 0.1000）。
    # 数据源端仍保留原始 float，便于后续计算。
    fmt = f"{{:.{int(float_digits)}f}}"
    formatted_row = {}
    for k, v in row.items():
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                formatted_row[k] = ""
            else:
                formatted_row[k] = fmt.format(float(v))
        else:
            formatted_row[k] = v

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(formatted_row)
        


def entropy_aware_focal_loss(
    logits,
    target,
    mask,
    gamma=0.5,
    beta=0.1,
    eps=1e-8,
):
    """
    logits: [B, L, 21]
    target: [B, L]
    mask:   [B, L]  (1 = compute loss)
    """

    logp = torch.log_softmax(logits, dim=-1)
    p = torch.exp(logp)

    # CE
    ce = torch.nn.functional.nll_loss(
        logp.view(-1, logp.size(-1)),
        target.view(-1),
        reduction="none"
    ).view_as(target)

    # p_t
    pt = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    # entropy
    entropy = -(p * logp).sum(dim=-1)

    # focal + entropy
    loss = ((1 - pt) ** gamma) * (1 + beta * entropy) * ce

    return (loss * mask).sum() / (mask.sum() + eps)



if __name__ == "__main__":
    from transformers import  EsmForProteinFolding
    import time
    model:EsmForProteinFolding = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to("cuda:0")
    model.eval()
    pdb_path="/root/autodl-tmp/cath_chain_dir/test/5e3x.A.pdb"
    seq1='MEELKSYYKRVAKYYSAAALLYWDMQTYMPKDAGPYRAEVLSEIGTYAFKQITDDALGKLLETAQPQSEIDEKLVYVGKKEYYKYKKVPPELFQEIMITSTMLEQKWEIAKPRGDFEEVRPLLEKIVDLSRKYADILGYEGEPYNALLDLYEPGMKAEEVDQIFSKVRDFIVEVLEKIERLPKSEDPFNREIGVDKQKEFSNWLLHYLKYDFTKGRLDVSAHPFTNPIGLNDVRITTRYIVNDIRNSIYSTIHEFGHALYALSIPTEFYGLPIGSSASYGFDESQSRFWENVVGRSLAFWKGIYSKFIEIVPEMRGYSVEELWRAVNRVQRSFIRTEADEVTYNLHIIIRFEIERELINGELSVKDVPDKWNELYKKYLGLDVPNNTLGCMQDPHWFGGNFGYFPTYALGNLYAAQIFEKLKEEINFEEVVSAGNFEIIKNFLKEKIHSKGKMYEPSDLIKIVTGKPLSYESFVRYIKDKYSKVYEIEL'
    seq2='MNNLKSLLKRVAKYYSAAALLYWDMQTYMPKDAGPYRAEVLSEIGTYAFKQITDDALGKLLETAQPQSEIDEKLVYVGKKEYYKYKKVPPELFQEIMITSTMLEQKWEIAKPRGDFEEVRPLLEKIVDLSRKYADILGYEGEPYNALLDLYEPGMKAEEVDQIFSKVRDFIVEVLEKIERLPKSEDPFNREIGVDKQKEFSNWLLHYLKYDFTKGRLDVSAHPFTNPIGLNDVRITTRYIVNDIRNSIYSTIHEFGHALYALSIPTEFYGLPIGSSASYGFDESQSRFWENVVGRSLAFWKGIYSKFIEIVPEMRGYSVEELWRAVNRVQRSFIRTEADEVTYNLHIIIRFEIERELINGELSVKDVPDKWNELYKKYLGLDVPNNTLGCMQDPHWFGGNFGYFPTYALGNLYAAQIFEKLKEEINFEEVVSAGNFEIIKNFLKEKIHSKGKMYEPSDLIKIVTGKPLSYESFVRYIKDKYSKVYEIEL'
    seq3='MGGLKSRRKRVAKYYSAAALLYWDMQTYMPKDAGPYRAEVLSEIGTYAFKQITDDALGKLLETAQPQSEIDEKLVYVGKKEYYKYKKVPPELFQEIMITSTMLEQKWEIAKPRGDFEEVRPLLEKIVDLSRKYADILGYEGEPYNALLDLYEPGMKAEEVDQIFSKVRDFIVEVLEKIERLPKSEDPFNREIGVDKQKEFSNWLLHYLKYDFTKGRLDVSAHPFTNPIGLNDVRITTRYIVNDIRNSIYSTIHEFGHALYALSIPTEFYGLPIGSSASYGFDESQSRFWENVVGRSLAFWKGIYSKFIEIVPEMRGYSVEELWRAVNRVQRSFIRTEADEVTYNLHIIIRFEIERELINGELSVKDVPDKWNELYKKYLGLDVPNNTLGCMQDPHWFGGNFGYFPTYALGNLYAAQIFEKLKEEINFEEVVSAGNFEIIKNFLKEKIHSKGKMYEPSDLIKIVTGKPLSYESFVRYIKDKYSKVYEIEL'
    start_time=time.time()
    #with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):
    for i in range(10):
        index=calculate_metrics(pdb_path,[seq1,seq2,seq3],model)
        #index=refolding_structure(model,seq)
    
    end_time=time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(index)
