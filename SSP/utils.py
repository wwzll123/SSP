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

class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        if self.betas.device != t_int.device:
            self.betas = self.betas.to(t_int.device)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        if self.alphas_bar.device != t_int.device:
            self.alphas_bar = self.alphas_bar.to(t_int.device)
        return self.alphas_bar[t_int.long()]


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def seq_recovery(data, pred_seq):
    '''
    data.x is nature sequence

    '''
    ind = (data.x.argmax(dim=1) == pred_seq.argmax(dim=1))
    recovery = ind.sum() / ind.shape[0]
    return recovery, ind.cpu()


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def set_seed(seed=1024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def place_fourth_atom(
        a_coord: torch.Tensor,
        b_coord: torch.Tensor,
        c_coord: torch.Tensor,
        length: torch.Tensor,
        planar: torch.Tensor,
        dihedral: torch.Tensor,
) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral)
    ]

    d_coord = c_coord + sum([m * d for m, d in zip(m_vec, d_vec)])

    return d_coord


def place_missing_cb(atom_positions):
    cb_coords = place_fourth_atom(atom_positions[:, 2], atom_positions[:, 0],
                                  atom_positions[:, 1], torch.tensor(1.522),
                                  torch.tensor(1.927), torch.tensor(-2.143))
    cb_coords = torch.where(torch.isnan(cb_coords), torch.zeros_like(cb_coords), cb_coords)

    # replace all vitural cb coords
    atom_positions[:, 3] = cb_coords
    return atom_positions


def place_missing_o(atom_positions, missing_mask):
    o_coords = place_fourth_atom(
        torch.roll(atom_positions[:, 0], shifts=-1, dims=0), atom_positions[:, 1],
        atom_positions[:, 2], torch.tensor(1.231), torch.tensor(2.108),
        torch.tensor(-3.142))
    o_coords = torch.where(torch.isnan(o_coords), torch.zeros_like(o_coords), o_coords)

    atom_positions[:, 4][missing_mask == 0] = o_coords[missing_mask == 0]
    return atom_positions


def cal_stats_metric(metric_list):
    mean_metric = np.mean(metric_list)
    median_metric = np.median(metric_list)
    return mean_metric, median_metric


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_entropy(log_probs):
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -1 * p_log_p.mean(dim=-1)
    return entropy


def fuse_logits_by_log_probs(log_prob_list, logits_list, temp=1.):
    entropy_list = [get_entropy(log_probs) for log_probs in log_prob_list]
    entropy = torch.stack(entropy_list, dim=0)
    entropy = torch.nn.functional.softmax(-1 * entropy / temp, dim=0)

    # fuse by entropy
    logits_list = torch.stack(logits_list, dim=0)
    logits = (entropy.unsqueeze(-1) * logits_list).sum(dim=0)

    return logits


def sin_mask_ratio_adapter(beta_t_bar, max_deviation=0.2, center=0.5):
    adjusted = beta_t_bar * torch.pi * 0.5
    sine = torch.sin(adjusted)
    adjustment = sine * max_deviation
    mask_ratio = center + adjustment
    return mask_ratio.squeeze(1)


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
    为什么会返回一个标量。

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