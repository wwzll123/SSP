import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import random
from esm.sdk.api import GenerationConfig, ESMProtein
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization import get_esm3_model_tokenizers
from tqdm import tqdm

tokenizer_collection = get_esm3_model_tokenizers()


def collate_fn(batch):
    return batch[0]


def random_crop_sequence(
    sequence: str,
    coord: torch.Tensor,
    structure_tokens: torch.Tensor,
    sequence_tokens: torch.Tensor,
    max_length: int,
) -> tuple:
    """
    随机截断序列到 max_length。
    按照连续的序列索引截断（随机选择起始位置）。
    
    Args:
        sequence: 原始序列字符串
        coord: 坐标张量 [seq_len, 37, 3] 或 [1, seq_len, 37, 3]
        structure_tokens: 结构 token [seq_len] 或 [1, seq_len]
        sequence_tokens: 序列 token [seq_len]
        max_length: 最大长度
        
    Returns:
        截断后的 (sequence, coord, structure_tokens, sequence_tokens)
    """
    seq_len = len(sequence)
    if seq_len <= max_length:
        return sequence, coord, structure_tokens, sequence_tokens
    
    # 随机选择起始位置
    start_idx = random.randint(0, seq_len - max_length)
    end_idx = start_idx + max_length
    
    # 截断序列字符串
    cropped_sequence = sequence[start_idx:end_idx]
    
    # 截断坐标（处理不同维度）
    if coord.dim() == 4:  # [1, seq_len, 37, 3]
        cropped_coord = coord[:, start_idx:end_idx, :, :]
    else:  # [seq_len, 37, 3]
        cropped_coord = coord[start_idx:end_idx, :, :]
    
    # 截断结构 token（处理不同维度）
    if structure_tokens.dim() == 2:  # [1, seq_len]
        cropped_structure_tokens = structure_tokens[:, start_idx:end_idx]
    else:  # [seq_len]
        cropped_structure_tokens = structure_tokens[start_idx:end_idx]
    
    # 截断序列 token
    cropped_sequence_tokens = sequence_tokens[start_idx:end_idx]
    
    return cropped_sequence, cropped_coord, cropped_structure_tokens, cropped_sequence_tokens


class ESM3RL_DataSet(Dataset):
    def __init__(self, pdb_dir, structure_token_dir, max_length: int = None, max_samples: int = None):
        """
        Args:
            pdb_dir: PDB 文件目录
            structure_token_dir: 预计算的结构 token 目录
            max_length: 最大序列长度，超过则随机截断。None 表示不截断。
            max_samples: 最大样本数量（用于调试），None 或 0 表示使用全部数据。
        """
        self.data_path = pdb_dir
        self.structure_token_dir = structure_token_dir
        self.max_length = max_length
        self.pdb_file = os.listdir(pdb_dir)
        
        # 限制样本数量（用于快速调试）
        if max_samples and max_samples > 0:
            self.pdb_file = self.pdb_file[:max_samples]
        
    def __len__(self):
        return len(self.pdb_file)
    
    def __getitem__(self, idx):
        pdb_file = self.pdb_file[idx]
        pdb_path = os.path.join(self.data_path, pdb_file)
        esm_protein = ESMProtein.from_pdb(pdb_path)
        esm_protein.coordinates[:, 4:, :] = torch.nan  # backbone informed
        coord = esm_protein.coordinates
        sequence = esm_protein.sequence
        
        sequence_tokens_lab = torch.tensor(
            tokenizer_collection.sequence.encode(
                sequence, add_special_tokens=False), dtype=torch.long
        )
        structure_tokens = torch.load(
            os.path.join(self.structure_token_dir, pdb_file.replace('.pdb', '.pt'))
        )
        
        # 应用长度截断
        if self.max_length is not None and len(sequence) > self.max_length:
            sequence, coord, structure_tokens, sequence_tokens_lab = random_crop_sequence(
                sequence=sequence,
                coord=coord,
                structure_tokens=structure_tokens,
                sequence_tokens=sequence_tokens_lab,
                max_length=self.max_length,
            )
        
        return {
            'pdb_id': pdb_file[:-4],
            'structure_coords': coord if coord.dim() == 4 else coord.unsqueeze(0),
            'structure_tokens': structure_tokens if structure_tokens.dim() == 2 else structure_tokens.unsqueeze(0),
            'sequence_tokens': sequence_tokens_lab,
            'true_sequence': sequence,
        }

        



