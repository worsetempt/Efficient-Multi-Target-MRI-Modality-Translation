# MRI-Translator

A modular MRI modality translation repository built around a latent-first pipeline.

The project starts with **MRM**, which learns image representations and extracts latent embeddings from MRI slices. Those latents are then used by downstream modules for latent generation, latent translation, conditional image translation, and latent decoding.

All notebooks in notebooks/ implements the baseline model for 2 classes/modalities

## Pipeline Summary

The repository follows this pipeline:

1. **MRM**

   * trains the masked reconstruction encoder
   * evaluates reconstruction and latent quality
   * extracts latents for `train`, `val`, and `test`

2. **MDN**

   * consumes MRM latent outputs
   * learns a class-conditional latent generator
   * generates modality-conditioned latents
   * evaluates generated latents with UMAP + FD

3. **latent_translator**

   * consumes MRM latent outputs
   * learns source-to-target latent translation
   * generates translated target latents
   * evaluates translated latents with UMAP + FD

4. **CCU-Net**

   * trains on source images + real MRM target latents
   * performs inference/eval using generated latents from either MDN or latent_translator
   * evaluates image outputs with SSIM, PSNR, and L1

5. **decoder**

   * trains a latent-to-image decoder
   * decodes generated latents from MDN or latent_translator
   * evaluates decoded outputs with SSIM and PSNR

## Repository Structure

```text
MRI-Translator/
├── README.md
├── .gitignore
├── environment.yml
├── requirements.txt
├── configs/
│   ├── mrm.yaml
│   ├── mdn.yaml
│   ├── latent_translator.yaml
│   ├── ccunet.yaml
│   └── decoder.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── latents/
├── experiments/
│   ├── mrm/
│   ├── mdn/
│   ├── latent_translator/
│   ├── ccunet/
│   └── decoder/
├── notebooks/
├── outputs/
└── src/
    ├── __init__.py
    ├── mrm_main.py
    ├── mdn_main.py
    ├── latent_translator_main.py
    ├── ccunet_main.py
    ├── decoder_main.py
    ├── datasets/
    ├── models/
    ├── training/
    ├── evaluation/
    ├── inference/
    └── utils/
```

## Setup

### Conda

```bash
conda env create -f environment.yml
conda activate mri-translator
```

### Pip

```bash
pip install -r requirements.txt
```

## Config Files

* `configs/mrm.yaml`
* `configs/mdn.yaml`
* `configs/latent_translator.yaml`
* `configs/ccunet.yaml`
* `configs/decoder.yaml`

Each module is driven by its YAML config instead of large CLI argument lists.

## VSCode Terminal Commands

Run all commands from the repository root:

```bash
cd MRI-Translator
```

### MRM

Train:

```bash
python -m src.mrm_main --config configs/mrm.yaml --mode train
```

Evaluate:

```bash
python -m src.mrm_main --config configs/mrm.yaml --mode eval
```

Extract latents:

```bash
python -m src.mrm_main --config configs/mrm.yaml --mode extract
```

### MDN

Train:

```bash
python -m src.mdn_main --config configs/mdn.yaml --mode train
```

Generate latents:

```bash
python -m src.mdn_main --config configs/mdn.yaml --mode generate
```

Evaluate:

```bash
python -m src.mdn_main --config configs/mdn.yaml --mode eval
```

### latent_translator

Train:

```bash
python -m src.latent_translator_main --config configs/latent_translator.yaml --mode train
```

Generate translated latents:

```bash
python -m src.latent_translator_main --config configs/latent_translator.yaml --mode generate
```

Evaluate:

```bash
python -m src.latent_translator_main --config configs/latent_translator.yaml --mode eval
```

### CCU-Net

Train:

```bash
python -m src.ccunet_main --config configs/ccunet.yaml --mode train
```

Infer:

```bash
python -m src.ccunet_main --config configs/ccunet.yaml --mode infer
```

Evaluate:

```bash
python -m src.ccunet_main --config configs/ccunet.yaml --mode eval
```

### decoder

Train:

```bash
python -m src.decoder_main --config configs/decoder.yaml --mode train
```

Evaluate:

```bash
python -m src.decoder_main --config configs/decoder.yaml --mode eval
```

## Expected Execution Order

A typical end-to-end workflow is:

1. Train MRM
2. Evaluate MRM
3. Extract MRM latents
4. Train MDN and/or latent_translator
5. Generate downstream latents
6. Evaluate generated latents
7. Train CCU-Net on real MRM latents
8. Run CCU-Net inference/eval with generated latents
9. Train decoder
10. Decode/evaluate generated latents with decoder

## Outputs

All module outputs are saved under `experiments/`.

Typical artifacts include:

* checkpoints
* metrics
* visualizations
* extracted latents
* generated latents
* translated latents
* decoded outputs

## Notes

* MRM is the canonical upstream latent source.
* MDN and latent_translator must consume MRM latent outputs using the shared latent contract.
* CCU-Net inference/eval must accept generated latents from either MDN or latent_translator.
* decoder eval must decode generated latents from either MDN or latent_translator.
* Keep modality naming, split naming, and config keys consistent across all modules.

## Contributors

* William Liu — UC San Diego  
* Advaith Modali — UC San Diego  
* Nitin Venkatesan — UC San Diego

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
