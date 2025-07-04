```
nerf_demo/
├── configs/
│   ├── eval.yaml              # Config for Evaluation
│   └── train.yaml             # Config for Training
│
├── data/
│   ├── __init__.py
│   ├── loader.py              # NeRFDataset: images + poses + intrinsics
│   └── rays.py                # RayGenerator: camera->world ray origins & dirs
│
├── models/
│   ├── __init__.py
│   ├── nerf.py                # TinyNeRF MLP + (later) SHLighting module
│   └── posenc.py              # positional encoding functions
│
├── rendering/
│   ├── __init__.py
│   ├── sampling.py            # stratified or hierarchical sampling utilities
│   └── volume_render.py       # volume_render() composites σ, RGB → pixel
│
├── train/
│   ├── __init__.py
│   ├── train_loop.py          # orchestrates epochs, batches, logging, checkpoints
│   └── eval.py                # novel‐view synthesis + PSNR/SSIM metrics
│
├── scripts/
│   ├── export_dataset.sh      # the blender command wrapper
│   └── run_training.sh        # sample `python train_loop.py --config configs/train.yaml`
│
├── notebooks/
│   └── demo.ipynb             # interactive exploration & visualization
│
├── utils/
│   ├── __init__.py
│   ├── io.py                  # JSON load/save, image utilities
│   └── visualization.py       # matplotlib helpers for plotting loss, renders
│
├── requirements.txt           # torch, torchvision, PyYAML, etc.
└── README.md                  # overview, setup, and usage instructions
```
