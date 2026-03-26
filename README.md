# 🔥 A Symmetric Self-play Online Preference Optimization Framework for Protein Inverse Folding

![SSP](main.png)



# 🏛️ Installation:
- 1.Download the source code in this repository.
- 2.Download the weights of all SSP models at [https://huggingface.co/zengwenwu/EiRA](https://huggingface.co/zengwenwu/EiRA/tree/main).
- 3.Unzip all .zip packages.

|CheckPoint Name | Description | Size |
|:--------|:--------:|-------:|
| DNAbinder_lora_ft32_DNAlen50_cross_att_DNAtransformer_DNAbinder.pth | DNA transformer and Cross attention weights of DNA-informed EiRA | 2.23G |
| EiRA_checkpoint_DNAbinder_lora_ft32_DNAlen50.zip | LoRA weighht of DNA-informed EiRA | 23.9M |
| EiRA_checkpoint_vanilla_lora_ft32_repeat_penalty.zip | LoRA wight of EiRA without DPO | 10.1M |
| DPO_checkpoint_VanillaLora_part_data_no_repeat.zip | LoRA wight of EiRAD with DPO | 12.1M |

# 🔍 Protein Generation like ESM3

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
