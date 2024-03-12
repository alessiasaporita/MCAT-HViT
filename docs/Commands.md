Commands for Running Ablation Experiments.
===========
### SNN
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode omic --model_type snn --fusion None --reg_type omic 
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode omic --model_type snn --fusion None --reg_type omic --met
```

### MCAT
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat --fusion concat --apply_sig 
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat --fusion concat --apply_sig --met
```

### MCAT-ViT
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat_vit --fusion None --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat_vit --fusion None --apply_sig --met
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat_vit --fusion None --apply_sig --visual_dropout 0.3
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat_vit --fusion None --apply_sig --met --visual_dropout 0.3
```

### MCAT-HViT
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat_hvit --fusion None --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat_hvit --fusion None --apply_sig --met
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat_hvit --fusion None --apply_sig --visual_dropout 0.3
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ovary --mode coattn --model_type mcat_hvit --fusion None --apply_sig --met --visual_dropout 0.3
```