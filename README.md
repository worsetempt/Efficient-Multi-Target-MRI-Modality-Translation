# MRI-Translator

## Overview

Modular MRI modality translation repository with an MRM encoder, MDN latent generator, latent translator, CCU-Net image translator, and latent decoder.

## Pipeline

1. `mrm` trains the encoder and extracts latent artifacts.
2. `mdn` generates class-conditional latents from MRM outputs.
3. `latent\\\_translator` predicts target latents from source latents.
4. `ccunet` trains on real MRM latents and infers with generated latents.
5. `decoder` reconstructs images from latent vectors.

## Structure

```text
configs/
src/
  datasets/
  models/
  training/
  evaluation/
  inference/
  utils/
experiments/
outputs/
notebooks/
```

## Setup

```bash
conda env create -f environment.yml
conda activate mri-translator
pip install -r requirements.txt
```

## Configs

* `configs/mrm.yaml`
* `configs/mdn.yaml`
* `configs/latent\\\_translator.yaml`
* `configs/ccunet.yaml`
* `configs/decoder.yaml`

## VSCode terminal commands

```bash
# =========================

\# 0. from repo root

\# =========================

cd MRI-Translator



\# optional: create env

conda env create -f environment.yml

conda activate mri-translator



\# or pip

pip install -r requirements.txt





\# =========================

\# 1. MRM

\# =========================



\# train

python -m src.mrm\_main --config configs/mrm.yaml --mode train



\# eval

python -m src.mrm\_main --config configs/mrm.yaml --mode eval



\# extract latents for train/val/test

python -m src.mrm\_main --config configs/mrm.yaml --mode extract





\# =========================

\# 2. MDN

\# =========================



\# train

python -m src.mdn\_main --config configs/mdn.yaml --mode train



\# generate latents

python -m src.mdn\_main --config configs/mdn.yaml --mode generate



\# eval generated latents

python -m src.mdn\_main --config configs/mdn.yaml --mode eval





\# =========================

\# 3. latent\_translator

\# =========================



\# train

python -m src.latent\_translator\_main --config configs/latent\_translator.yaml --mode train



\# generate translated latents

python -m src.latent\_translator\_main --config configs/latent\_translator.yaml --mode generate



\# eval translated latents

python -m src.latent\_translator\_main --config configs/latent\_translator.yaml --mode eval





\# =========================

\# 4. CCU-Net

\# =========================



\# train using MRM latents

python -m src.ccunet\_main --config configs/ccunet.yaml --mode train



\# infer using generated latents from MDN or latent\_translator

python -m src.ccunet\_main --config configs/ccunet.yaml --mode infer



\# eval

python -m src.ccunet\_main --config configs/ccunet.yaml --mode eval





\# =========================

\# 5. decoder

\# =========================



\# train

python -m src.decoder\_main --config configs/decoder.yaml --mode train



\# eval / decode generated latents

python -m src.decoder\_main --config configs/decoder.yaml --mode eval```

