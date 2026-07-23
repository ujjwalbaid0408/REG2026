# Environment Setup

Three environments are involved. Only the first is needed to reproduce the submitted model;
the other two are for the offline report metric and for the container build.

| Environment | Purpose | Definition |
|---|---|---|
| **Training / extraction** | Tile extraction, encoder features, MIL training | `requirements-train.txt` |
| **Container runtime** | What actually runs on Grand Challenge | `repo_template/algorithm_submission_template/requirements.txt` |
| **Report-metric eval** *(optional)* | Faithful offline replica of the report sub-metric | see [Report-metric environment](#4-report-metric-environment-optional) |

---

## 1. System dependencies

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y libopenslide0 build-essential git

# RHEL / Amazon Linux
sudo dnf install -y openslide gcc git
```

`libopenslide0` is the C library behind `openslide-python`. If you cannot install system
packages, the pinned `openslide-bin==4.0.0.8` wheel bundles it — but note that the fallback
strip decoder (`tifffile` + `imagecodecs`) is a hard requirement either way, because ~8.7% of
the corpus cannot be opened by OpenSlide at all.

Python **3.11** is what we used (3.10+ should work).

## 2. Training / extraction environment

```bash
git clone https://github.com/ujjwalbaid0408/REG2026.git
cd REG2026

python -m venv venv && source venv/bin/activate
pip install --upgrade pip

# torch FIRST, from the CUDA 12.8 index (see GPU note below)
pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements-train.txt
pip install -e .                    # installs the `reg2026` package
```

### CONCH (vendored, gated)

CONCH is not on PyPI and its weights are gated on Hugging Face:

```bash
git clone https://github.com/mahmoodlab/CONCH repo_conch
pip install -e repo_conch
huggingface-cli login               # token must have access to MahmoodLab/CONCH
```

Weight download is covered in [`../MODEL_WEIGHTS.md`](../MODEL_WEIGHTS.md).

### GPU note — this bit is not optional

| GPU | Arch | Works with |
|---|---|---|
| A10G (deployment target), L4 | sm_86, sm_89 | any recent CUDA build |
| **RTX PRO 6000** | **sm_120 (Blackwell)** | **cu128 only** |
| **B200** | **sm_100 (Blackwell)** | **cu128 only** |

A `cu124` build contains no Blackwell kernels and fails at the first matmul with
`CUDA error: no kernel image is available for execution on the device`. This cost us a full
debugging cycle; if you are on Blackwell, install the cu128 wheels above.

Verify before launching anything long:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
# expect: 2.9.1+cu128 True [... 'sm_100', 'sm_120']
```

### Verify the environment end to end

```bash
python -c "
from reg2026.encoder import load_encoder
m, pre = load_encoder('conch'); print('CONCH ok', m._emb_mean, m._emb_std)
"
python scripts/sanity_check_embeddings.py     # shapes, dtype, placeholder detection
```

A subtle failure worth guarding against: if the CONCH or UNI2-h import throws, the predictor
**catches it and silently falls back** to the template baseline (~0.25) rather than crashing.
Always confirm the encoders load rather than assuming a clean run means a correct one.

## 3. Container runtime environment

Built from `pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime`; the base image already provides
torch/torchvision, so `repo_template/algorithm_submission_template/requirements.txt` only adds
the rest. It pins deliberately and differently from training:

| Pin | Why |
|---|---|
| `timm==1.0.27` | UNI2-h arch guard. An older resolve makes the UNI2-h build raise → silent baseline fallback. |
| `transformers==4.46.3` | Vendored CONCH hard-imports it. Missing → `conch` import fails → silent baseline fallback. |
| `tifffile`, `imagecodecs` | Striped-TIFF strip decoder. Missing → ~8.7% of slides read as all-zero features. |

All three failure modes are **silent** — the container still emits schema-valid output, just from
the wrong path. `docs/HOST_BUILD_INSTRUCTIONS_REV4.txt` STEP 1a shows the log line to grep for.

See [`CONTAINER_BUILD.md`](CONTAINER_BUILD.md) for the build itself.

## 4. Report-metric environment (optional)

Only needed to run `scripts/diag_report.py` / `scripts/optimize_report_templates.py`, which
replicate the challenge's report sub-metric offline. It needs spaCy + scispaCy + PubMedBERT and
conflicts with the training pins, so keep it separate:

```bash
python -m venv eval_venv --system-site-packages
source eval_venv/bin/activate
pip install spacy==3.7.5 scispacy==0.5.4 transformers==4.57
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

`en_core_sci_lg` is distributed only from that S3 release URL — not PyPI, not Hugging Face. The
embedding scorer additionally pulls `NeuML/pubmedbert-base-embeddings`. CPU is fine.

## Known environment gotchas

- **User-site packages.** Our cluster venv had `~/.local` packages on the path. Do **not** set
  `PYTHONNOUSERSITE=1` — it hides `safetensors` and others. The venv's torch correctly precedes
  `~/.local` in resolution order.
- **SLURM (our cluster).** Submit from a non-`/group` filesystem; request GPUs with `--gpus=N`,
  not `--gres=gpu:` (the plugin rejects the latter as "no GPU requested"). Adjust
  partition/account in `slurm/*.sbatch` for your site.
- **Shared-filesystem contention.** More than ~12 tile-reader workers against a shared mount
  produced transient `Unsupported or missing image file` errors on slides that are perfectly
  readable. Keep `TILE_WORKERS <= 12`.
