# 🔥 A Symmetric Self-play Online Preference Optimization Framework for Protein Inverse Folding

![SSP](main.png)


# 🏛️ Installation:
- 1.Download the source code in this repository.
- 2.Download the weights of all SSP models at [https://huggingface.co/zengwenwu/SSP](https://huggingface.co/zengwenwu/SSP/tree/main). ESM3 uses the [peft](https://github.com/huggingface/peft); ESM-IF1 uses the [minlora](https://github.com/changjonathanc/minLoRA); ProteinMPNN provides complete weights.
- 3.Unzip all .zip packages.
- 4.Prepare the environment. Please note that this environment is prepared for ESM3. If you need to use ProteinMPNN and ESM-IF1, please create a different environment.
 ```
pip install requirements.txt
```

# 🔍 Inference

Once you have prepared the pdb/cif files, you can run the inference script directly..

 ```
 $ python run_design.py \
           --pdb example/1a7l.A.pdb \
           --temperature 1 \
           --num_samples 10 \
           --lora_dir "YOUR LOCAL PATH" \
           --output "SAVE FASTA PATH" \
           --device cuda:0
```


# ⚔️ Training
- 1.Before starting the training, you should first read the [Tokenization Tutorial](./Tokenization) and prepare the tokens.
- 2.Run the following command depend on the number of GPUs available to you.
 ```
bash run_ddp.sh NUM_GPU
```

# 📄 Citation
 ```
pass
 ```
# ☎ Contact
Please feel free to contact us at wwz_cs@hnu.edu.cn, if you have any question!
