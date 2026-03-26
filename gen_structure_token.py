import traceback
import argparse
import numpy as np
import os
from tqdm import tqdm
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.tokenization import get_esm3_model_tokenizers
from esm.pretrained import ESM3_structure_encoder_v0
from esm.utils import encoding

parser = argparse.ArgumentParser()
parser.add_argument('--protein_list_file',
                    type=str,
                    default='./protein_list.txt')
parser.add_argument('--pdb_dir',
                    type=str,
                    default='./pdb_dir')
parser.add_argument('--target_dir',
                    type=str,
                    default='./token_dir')
config = parser.parse_args()

# Initialize structure encoder and tokenizer
structure_encoder = ESM3_structure_encoder_v0().cuda()
tokenizer_collection = get_esm3_model_tokenizers()
str_tokenizer = tokenizer_collection.structure

def generate_structure_token(protein_id):
    pdb_path = config.pdb_dir + os.sep + protein_id + '.pdb'
    target_path = config.target_dir + os.sep + f"{protein_id}_tokens.npz"

    chain = ProteinChain.from_pdb(pdb_path)
    protein = ESMProtein.from_protein_chain(chain)

    # Tokenize structure only
    coordinates, _, structure_tokens = encoding.tokenize_structure(
        coordinates=protein.coordinates.cuda(),
        structure_encoder=structure_encoder,
        structure_tokenizer=str_tokenizer,
        reference_sequence=protein.sequence,
        add_special_tokens=True,
    )

    np.savez_compressed(target_path,
                        structure=np.uint16(structure_tokens.cpu().detach().numpy()))

if __name__ == '__main__':
    protein_ids = np.loadtxt(config.protein_list_file, dtype=str)
    for one_protein_id in tqdm(protein_ids):
        target_path = config.target_dir + os.sep + f"{one_protein_id}_tokens.npz"
        pdb_path = config.pdb_dir + os.sep + one_protein_id + '.pdb'
        if not os.path.exists(pdb_path) or os.path.exists(target_path):
            continue
        try:
            generate_structure_token(one_protein_id)
        except Exception as e:
            traceback.print_exc()
