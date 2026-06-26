from setuptools import setup, find_packages

setup(
    name="reg2026",
    version="0.1.0",
    description="REG2026 Pathologist Reasoning-Guided Report Generation — training & inference",
    packages=find_packages(include=["reg2026", "reg2026.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "tifffile",
        "imagecodecs",
        "openslide-python",
        "pillow",
        "h5py",
        "scikit-learn",
        "tqdm",
        "timm>=1.0",
        "einops",
        "open_clip_torch",
        "huggingface_hub",
    ],
)
