# 🔥 A Symmetric Self-play Online Preference Optimization Framework for Protein Inverse Folding

![SSP](main.png)


# 🏛️ Installation:
- 1.Download the source code in this repository.
- 2.Download the weights of all SSP models at [https://huggingface.co/zengwenwu/SSP](https://huggingface.co/zengwenwu/SSP/tree/main). ESM3 uses the peft; ESM-IF1 uses the minlora; ProteinMPNN provides complete weights.
- 3.Unzip all .zip packages.
- 4.Prepare the environment. Please note that this environment is prepared for ESM3. If you need to use ProteinMPNN and ESM-IF1, please create a different environment.
 ```
pip install requirements.txt
```

# 🔍 Inference

EiRA inherits the flexible multi-modal editing feature of ESM3 and supports the free combination of multiple tracks of prompts.

 ```
 $ python run_EiRA.py \
           --weight_dir "The local path of the downloaded weight file"
           --SRC_PDB_path "The path of your template"
           --designed_seq_save_path "Result path"
           --design_num "Number of designed sequences"
           --inform_position "The constant residue indices in the template, like: 0,1,2,3,5,6,7,8,9"
           --device cuda:0
           --chain Template chain (like "A")
```


# ⚔️ Training
- 1.Before starting the training, you should first read the [Tokenization Tutorial](./Tokenization) and prepare the tokens.
- 2.Run the following command depend on the number of GPUs available to you.
 ```
 $ torchrun --nproc_per_node=8 --master_port=29512 pretrain.py \
           --gpu 0,1,2,3,4,5,6,7
           --token_dir "The path contains all token in .npz format"
           --save_path "Checkpoint path to be saved"
           --batch_size 20
           --fine_tuning_num 16
           --epochs 5
           --prefetch_factor 30
           --num_workers 16
```

# 📄 Citation
 ```
pass
 ```
# ☎ Contact
Please feel free to contact us at wwz_cs@hnu.edu.cn, if you have any question!
