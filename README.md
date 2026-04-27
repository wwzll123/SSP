# 🔥 A Symmetric Self-play Online Preference Optimization Framework for Protein Inverse Folding

![SSP](main.png)


# 🏛️ Installation:
- 1.Download the source code in this repository.
- 2.Download the weights of all SSP models at [https://huggingface.co/XXX/SSP](https://huggingface.co/XXX/SSP/tree/main). For ESM3, we use the [peft](https://github.com/huggingface/peft); for ESM-IF1, we use the [minlora](https://github.com/changjonathanc/minLoRA); for ProteinMPNN, we provide complete weights.
- 3.Unzip all .zip packages.
- 4.Prepare the environment. Please note that this environment is prepared for ESM3. If you need to use [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) and [ESM-IF1](https://github.com/facebookresearch/esm), please create their proprietary environment..
 ```
pip install requirements.txt
```

# 🔍 Inference

Once you have prepared the pdb/cif files, you can run the inference script directly.

 ```
 $ python run_design.py \
           --pdb example/1a7l.A.pdb \
           --temperature 1 \
           --num_samples 10 \
           --lora_dir "YOUR LOCAL MODEL WEIGHT PATH" \
           --output "SAVE FASTA PATH" \
           --device cuda:0
```


# ⚔️ Training
- 1.Before starting the training, you should first generate the [Structure Token](./gen_structure_token.py).
- 2.[Configure](./SSP/esm3_config.yaml) your PDB, token, and weight path.
- 3.Run the following command depend on the number of GPUs available to you.
 ```
bash run_ddp.sh NUM_GPU
```
