import os
import argparse
import torch
from esm.models.esm3 import ESM3, ESMProteinTensor
from esm.sdk.api import GenerationConfig, ESMProtein
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization import get_esm3_model_tokenizers
from esm.utils import encoding
from peft import PeftModel
import gemmi

def main():
    parser = argparse.ArgumentParser(description="ESM3 + LoRA inference: generate sequences from a single PDB")
    parser.add_argument("--pdb", type=str, required=True, help="Path to input PDB file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of sequences to generate per PDB (default: 1)")
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory containing LoRA weights")
    parser.add_argument("--output", type=str, required=True, help="Output FASTA file path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.pdb):
        print(f"Error: PDB file does not exist: {args.pdb}")
        return
    if not os.path.isdir(args.lora_dir):
        print(f"Error: LoRA directory does not exist: {args.lora_dir}")
        return

    # Initialize model components
    print("Initializing ESM3 model...")
    device = torch.device(args.device)
    structure_encoder = ESM3_structure_encoder_v0().to(device)
    tokenizer_collection = get_esm3_model_tokenizers()
    str_tokenizer = tokenizer_collection.structure
    base_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)

    # Load LoRA weights
    print(f"Loading LoRA weights from: {args.lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.lora_dir).to(device)
    model.eval()

    # Extract PDB ID from filename
    pdb_id = os.path.basename(args.pdb).replace(".pdb", "").replace(".cif", "")
    print(f"Processing PDB: {pdb_id}")

    # Load protein structure
    if args.pdb.endswith(".cif"):
        doc = gemmi.cif.read_file(args.pdb)
        block = doc.sole_block()
        structure = gemmi.make_structure_from_block(block)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_pdb = tmp.name
        structure.write_pdb(tmp_pdb)
        esmprotein = ESMProtein.from_pdb(tmp_pdb)
        os.unlink(tmp_pdb)
    else:
        esmprotein = ESMProtein.from_pdb(args.pdb)

    # Prepare input tensors
    coord = torch.full_like(esmprotein.coordinates, torch.nan).to(device)  # [seq_len, 37, 3]
    coord[:, 0:4, :] = esmprotein.coordinates[:, 0:4, :].to(device)

    # Tokenize structure
    coordinates, _, structure_tokens = encoding.tokenize_structure(
        coordinates=coord,
        structure_encoder=structure_encoder,
        structure_tokenizer=str_tokenizer,
        reference_sequence='',
        add_special_tokens=True,
    )

    num_steps = coord.shape[0] // 4
    if num_steps < 1:
        num_steps = 1

    # Generate multiple samples and compute pLDDT/pTM
    results = []  # list of (seq, plddt, ptm)
    for sample_idx in range(args.num_samples):
        print(f"Generating sequence {sample_idx+1}/{args.num_samples}")
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Step 1: generate sequence from structure
            gen_protein_tensor: ESMProteinTensor = model.generate(
                ESMProteinTensor(
                    coordinates=coordinates,
                    structure=structure_tokens
                ),
                config=GenerationConfig(
                    track="sequence",
                    num_steps=num_steps,
                    temperature=args.temperature,
                )
            )
            seq_tokens = gen_protein_tensor.sequence[1:-1].detach().cpu().numpy()
            seq = tokenizer_collection.sequence.decode(seq_tokens, skip_special_tokens=True).replace(' ', '')

            # Step 2: generate structure from the generated sequence to get pLDDT and pTM
            gen_structure = model.generate(
                ESMProtein(sequence=seq),
                config=GenerationConfig(
                    track="structure",
                    num_steps=num_steps,
                    temperature=args.temperature,
                )
            )
            plddt = gen_structure.plddt.mean().item() if hasattr(gen_structure, 'plddt') else 0.0
            ptm = gen_structure.ptm.item() if hasattr(gen_structure, 'ptm') else 0.0

        results.append((seq, plddt, ptm))

    # Sort results by pLDDT descending, then by pTM descending
    results.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Write output FASTA with pLDDT and pTM in header
    with open(args.output, 'w') as f:
        for seq, plddt, ptm in results:
            f.write(f">{pdb_id}|plddt:{plddt:.4f}|ptm:{ptm:.4f}\n{seq}\n")

    print(f"Done. Generated {len(results)} sequences, sorted by pLDDT then pTM, saved to {args.output}")

if __name__ == "__main__":
    main()