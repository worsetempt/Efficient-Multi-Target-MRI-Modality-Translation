# MRI-Translator

## Overview
Modular MRI modality translation repository with an MRM encoder, MDN latent generator, latent translator, CCU-Net image translator, and latent decoder.

## Pipeline
1. `mrm` trains the encoder and extracts latent artifacts.
2. `mdn` generates class-conditional latents from MRM outputs.
3. `latent_translator` predicts target latents from source latents.
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
- `configs/mrm.yaml`
- `configs/mdn.yaml`
- `configs/latent_translator.yaml`
- `configs/ccunet.yaml`
- `configs/decoder.yaml`

## VSCode terminal commands
```bash
python -m src.mrm_main train --config configs/mrm.yaml
python -m src.mrm_main eval --config configs/mrm.yaml
python -m src.mrm_main extract --config configs/mrm.yaml

python -m src.mdn_main train --config configs/mdn.yaml
python -m src.mdn_main generate --config configs/mdn.yaml
python -m src.mdn_main eval --config configs/mdn.yaml

python -m src.latent_translator_main train --config configs/latent_translator.yaml
python -m src.latent_translator_main generate --config configs/latent_translator.yaml
python -m src.latent_translator_main eval --config configs/latent_translator.yaml

python -m src.ccunet_main train --config configs/ccunet.yaml
python -m src.ccunet_main infer --config configs/ccunet.yaml
python -m src.ccunet_main eval --config configs/ccunet.yaml

python -m src.decoder_main train --config configs/decoder.yaml
python -m src.decoder_main eval --config configs/decoder.yaml
```
