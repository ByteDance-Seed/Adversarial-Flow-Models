# Adversarial Flow Models

This repository contains the official PyTorch implementation of both discrete and continuous Adversarial Flow Models.

> [**Adversarial Flow Models**](https://www.arxiv.org/abs/2511.22475)<br>
> [Shanchuan Lin](https://scholar.google.com/citations?user=EDWUw7gAAAAJ), [Ceyuan Yang](https://scholar.google.com/citations?user=Rfj4jWoAAAAJ), [Zhijie Lin](https://scholar.google.com/citations?user=xXMj6_EAAAAJ),  [Hao Chen](https://scholar.google.com/citations?user=QMuIRLYAAAAJ), [Haoqi Fan](https://scholar.google.com/citations?user=76B8lrgAAAAJ)
> <br>ByteDance Seed<br>

> [**Continuous Adversarial Flow Models**](https://arxiv.org/abs/2604.11521)<br>
> [Shanchuan Lin](https://scholar.google.com/citations?user=EDWUw7gAAAAJ), [Ceyuan Yang](https://scholar.google.com/citations?user=Rfj4jWoAAAAJ), [Zhijie Lin](https://scholar.google.com/citations?user=xXMj6_EAAAAJ),  [Hao Chen](https://scholar.google.com/citations?user=QMuIRLYAAAAJ), [Haoqi Fan](https://scholar.google.com/citations?user=76B8lrgAAAAJ)
> <br>ByteDance Seed<br>

## Colab Notebooks

* Try [Adversarial Flow Models](https://colab.research.google.com/drive/1qsRIKIVQgGm2YpkDxRHs-xIJzDvWKq_V?usp=sharing) on 1D Gaussian mixture.
* Try [Continuous Adversarial Flow Models](https://colab.research.google.com/drive/1gWVuCjsVA8Knq7pak5n-jjsHBYwdxpJA?usp=sharing) on 1D Gaussian mixture.


## AFMs

### Train

1. Install requirements `pip install -r requirements_afm.txt`.
2. Download VAE and other misc [checkpoints](https://huggingface.co/ByteDance-Seed/Adversarial-Flow-Models/tree/main/misc) to the root directory.
3. Download [dit.py](https://github.com/facebookresearch/DiT/blob/main/models.py) from the original DiT repo and place it under `models/afm/dit/dit.py`.
4. Configure your dataset. Instruction is provided in the next section.
5. Run the training configurations provided in `configs/train/afm`.
    * Replace `TORCHRUN` with your `torchrun` command with your GPU configuration.
    * Make sure `exp.gpu` is equal to your total amount of GPUs for the current per-rank batch size calculation.
    * You can set smaller `exp.bsz` for local debugging.
    * The training schedule is provided in Table 11 of the AFM paper. The current approach still requires more manual intervention. This is a limitation we hope to improve in future work.

```bash
TORCHRUN main.py configs/train/train_1nfe.yaml
```

### Evaluate
1. Download [pre-trained AFMs checkpoints](https://huggingface.co/ByteDance-Seed/Adversarial-Flow-Models), or use your own.
2. Generate 50K samples for FID evaluation.
```bash
TORCHRUN main.py configs/generate/afm/generate_1nfe.yaml
```
3. Use `/misc/pack_npz.py` to pack npz.
4. Use [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to evaluate FID.



## CAFMs

### Train
1. Install requirements `pip install -r requirements_cafm.txt`.
2. Download VAE and other misc [checkpoints](https://huggingface.co/ByteDance-Seed/Adversarial-Flow-Models/tree/main/misc) to the root directory.
3. Download model files.
    * Download [sit.py](https://github.com/willisma/SiT/blob/main/models.py) from the original SiT repo and place it under `models/cafm/sit/sit.py`.
    * Download [jit.py](https://github.com/LTH14/JiT/blob/main/model_jit.py) from the original JiT repo and place it under `models/cafm/jit/jit.py`.
    * No need to download model code for Z-Image.
4. Configure your dataset. Instruction is provided in the next section.
5. Download pre-trained checkpoints.
    * Download the [official pre-trained SiT-XL/2 checkpoint](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0).
    * Download the [official pre-trained JiT-H/16 checkpoint](https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/ALQzc8CWONX6fVvej_kfjY0/jit-h-16?dl=0&preview=checkpoint-last.pth&rlkey=14gjrblmljewpl6ygxzlr3njm&subfolder_nav_tracking=1). Use `/misc/convert_from_jit_format.py` to convert to our format.
    * Download [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) can place it under root directory as `/Z-Image`.
6. Run the training configurations provided in `configs/train/cafm`.
    * Replace `TORCHRUN` with your `torchrun` command with your GPU configuration.
    * Make sure `exp.gpu` is equal to your total amount of GPUs for the current per-rank batch size calculation.
    * You can set smaller `exp.bsz` for local debugging.
```bash
TORCHRUN main.py configs/train/cafm/train_cafm_sit.yaml
TORCHRUN main.py configs/train/cafm/train_cafm_jit.yaml
TORCHRUN main.py configs/train/cafm/train_cafm_zimage.yaml
```

### Evaluate
1. Download [pre-trained CAFMs checkpoints](https://huggingface.co/ByteDance-Seed/Adversarial-Flow-Models), or use your own.
2. We do not provide generation/evaluation code. Please use SiT/JiT codebase for generation and evaluation.
    * You may need `/misc/convert_to_jit_format.py` to convert our JiT saved ckpt to their format.
    * **Note that the FID logged by our training script is only a rough estimate. You will get better FID using their official evaluation code!**


## Dataloading
For our official training we pack imagenet and t2i datasets into parquet format. The dataloading code is provided in `/data` only for reference purposes. You can implement your own dataset loading logic.

For ImageNet, implement it as a IterableDataset with a forever loop that returns a dictionary with keys `image` and `label`. The `image` should be a PyTorch tensor of shape (3, H, W) with range [0, 1]. The dataset class should accept our `transform` to handle resize, cropping, and normalization to [-1, 1]. The `label` should be the class index with range [0, 999].

For ImageNet, CAFM SiT training also supports using offline dataloading. It a dictionary with keys `latent` and `label`. The `latent` should be a PyTorch tensor of shape (4, 32, 32). The train script will automatically skip VAE encoding and use the offline latents.

For T2I, implement it as a IterableDataset with a forever loop that returns a dictionary with keys `image` and `text`. The `image` should be a tensor of shape (3, H, W) and we support batching of different aspect ratios. The `text` should be a string.

Note that the IterableDataset must internally check the current rank and worker id to handle distributed partitioning. Otherwise the sample will be repeatedly seen in all ranks.

## Citation

```bibtex
@article{afm,
  title={Adversarial Flow Models},
  author={Lin, Shanchuan and Yang, Ceyuan and Lin, Zhijie and Chen, Hao and Fan, Haoqi},
  journal={arXiv preprint arXiv:2511.22475},
  year={2025}
}

@article{cafm,
  title={Continuous Adversarial Flow Models},
  author={Lin, Shanchuan and Yang, Ceyuan and Lin, Zhijie and Chen, Hao and Fan, Haoqi},
  journal={arXiv preprint arXiv:2604.11521},
  year={2026}
}
```

## About [ByteDance Seed Team](https://seed.bytedance.com/)
Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.