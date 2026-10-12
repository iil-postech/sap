# SAP: Spectral-Aligned Pruning for Universal Error-Correcting Code Transformers

Implementation of SAP.

Paper link: https://openreview.net/forum?id=Y5hbfFtOH1

## Method

SAP represents each code using the two largest algebraic eigenvalues of its parity-check bipartite adjacency matrix. It retrieves the nearest library signature by Euclidean distance and reuses its head/FFN masks when $exp(-0.1 \times \text{distance}) \geq 0.5$. Otherwise, it derives a target-specific Fisher mask and adds it to the library before retraining.

Masks physically remove attention heads and FFN channels. Masks physically remove attention heads and FFN channels. Recovery freezes the pruned backbone and trains LoRA adapters on the attention projection matrices ($W_Q$, $W_K$, $W_V$, and $W_O$) in all decoder layers, using BCE + KD loss from the unpruned pretrained teacher.

## Installation

Python with PyTorch, NumPy, and loralib:

```bash
pip install torch numpy loralib
```

A CUDA GPU is recommended for the full experiments. The scripts also support CPU execution.

## Running the code

Run the following commands from this directory. Paths are resolved relative to the scripts.

A pretrained checkpoint is included at [`checkpoints/fecct_pretrained.pt`](checkpoints/fecct_pretrained.pt). **You can run the experiments without pretraining:** build the library, then retrain your target code using the commands below.

Both scripts automatically use `pretrained_fecct.pt` if you have trained a new backbone; otherwise, they use the provided checkpoint. No checkpoint argument or manual file copying is needed. The included weights match the default architecture: 6 decoder layers, dimension 128, and 8 heads.

### 1. Build the library

```bash
python Build_Library.py --gpus 0 --model_prun_ratio 0.4
```

### 2. Retrieve, prune, and retrain a target code

```bash
python Main_DC.py --do_what prune_and_retrain_lora --gpus 0 --target_code BCH_N63_K36 --lora_r 8 --epochs 100
python Main_DC.py --do_what prune_and_retrain_lora --gpus 0 --target_code LDPC_N121_K70 --lora_r 8 --epochs 100
```

The script checks library compatibility, retrieves a mask, physically prunes the pretrained model, and retrains LoRA on the **target code**. 

Retrieval prints only `Library reuse succeeded.` or `Library reuse failed.` Low spectral similarity triggers target-specific mask derivation and library expansion. A missing/invalid library or checkpoint, a model mismatch, or a pruning-ratio mismatch instead stops execution with a nonzero exit code before retraining.

**Pruning ratios must match exactly.** `--model_prun_ratio 0.4` removes 40% of the attention/FFN MAC budget when deriving each reference mask. Reusing that mask on a different-length code can change its realized MAC reduction. A library built at another ratio is never searched, extended, or overwritten by these commands. To use another ratio, move the existing `pruning_library.pt` aside and rebuild it with the same ratio you will use for retraining. Changing the pretrained checkpoint also requires rebuilding the library.

### Optional: pretrain a new FECCT backbone

```bash
python Main_DC.py --do_what train --gpus 0 --epochs 4000
```

This optional command starts from random initialization and saves the lowest-training-loss state dictionary as `pretrained_fecct.pt`. The provided checkpoint is preserved. Once the new checkpoint exists, both library construction and retraining use it automatically. Rebuild `pruning_library.pt` before retraining with the new backbone; a library derived from different checkpoint weights is rejected.

## Code arguments

Only `train` and `prune_and_retrain_lora` are available through `--do_what`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--target_code` | `BCH_N63_K36` | Target code in `Codes_DB`; `--test_code` is an alias |
| `--lora_r` | `8` | LoRA rank |
| `--epochs` | `100` | Pretraining or retraining epochs |
| `--model_prun_ratio` | `0.4` | Pruning ratio shared by library construction and retrieval |
| `--gpus` | `0` | CUDA device IDs; examples select GPU 0 |
| `--workers` | `4` | Data-loader workers |
| `--batch_size` | `256` | Training/calibration batch size |
| `--test_batch_size` | `2048` | Test batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--seed` | `42` | Random seed |
| `--N_dec`, `--d_model`, `--h` | `6`, `128`, `8` | Backbone dimensions; must match the checkpoint/library |
| `--lora_alpha` | `16` | LoRA scaling parameter |
| `--alpha` | `1` | KD loss weight |


## Citation

```bibtex
@inproceedings{Cho2026Spectral,
  title={Spectral-Aligned Pruning for Universal Error-Correcting Code Transformers},
  author={Sanghyeon Cho and Taewoo Park and Seong-Joon Park and Dae-Young Yun and Hee-Youl Kwak and Sang-Hyo Kim and Yongjune Kim},
  booktitle="Advances in Neural Information Processing Systems (NeurIPS)",
  year={2026},
  url={https://openreview.net/forum?id=Y5hbfFtOH1}
}
```
## License

Codes are available only for non-commercial research purposes.

## Acknowledgement

This implementation follows the FECCT architecture proposed by Choukroun and Wolf ([ICLR 2024](https://openreview.net/forum?id=7KDuQPrAF3)), which builds on ECCT (Choukroun and Wolf, NeurIPS 2022). The Fisher-based structured pruning follows Kwon et al. (NeurIPS 2022), and LoRA adapters are implemented with [`loralib`](https://github.com/microsoft/LoRA) (Hu et al., ICLR 2022).
