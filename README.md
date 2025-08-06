# 🌱 Tiny-NeRF

**Tiny-NeRF** is a minimal PyTorch implementation of [Neural Radiance Fields (NeRF)](https://arxiv.org/abs/2003.08934). It provides a clean, educational base for understanding how NeRF works, including positional encoding, ray generation, volume rendering, and training pipelines.

---

## 📦 Features

- ✅ Tiny MLP with positional encoding for spatial and directional inputs  
- 🎯 Ray generation from camera intrinsics and poses to world-space rays  
- 🎨 Differentiable image synthesis with stratified sampling and volume rendering  
- 📈 Lightweight training pipeline with checkpointing, plotting, and visualization tools

---

## 🗂️ Directory Structure

```
Tiny-NeRF/
├── README.md
├── configs/
│   └── train.yaml                 # Training/evaluation configuration
├── data/
│   ├── loader.py                 # NeRFDataset for images, poses, intrinsics
│   └── rays.py                   # Ray generation utilities
├── models/
│   ├── nerf.py                   # TinyNeRF MLP
│   └── posenc.py                 # Sinusoidal positional encoding
├── rendering/
│   └── volume_render.py          # Stratified sampling, alpha compositing
├── train/
│   ├── train_loop.py             # Training loop with checkpointing
│   └── eval.py                   # Novel-view rendering tools
├── utils/
│   ├── io.py                     # Checkpoint, dataset loading
│   ├── model_utils.py            # Model loading for inference
│   └── visualization.py         # Plotting and rendering utilities
└── main.py                       # Demo stub
```

---

## 🔧 Requirements

Install dependencies using `requirements.txt` provided:

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Format

Prepare your dataset with the following structure:

```
<root_dir>/
├── images/
│   ├── 000.png
│   ├── 001.png
│   └── ...
├── intrinsics.json
└── poses.json
```

- `intrinsics.json` should contain:  
  `resolution_x`, `resolution_y`, `f_mm`, `sensor_width_mm` (optionally `sensor_height_mm`)

- `poses.json` must map numeric keys (`"0"`, `"1"`, …) to camera-to-world matrices

- You can download a small [Demo Dataset Here](https://drive.google.com/drive/folders/1qgHIg90g84B13LspqQbpzkUURM0HbuD6?usp=drive_link)

---

## ⚙️ Configuration

All settings live in [`configs/train.yaml`](configs/train.yaml):

```yaml
data:
  root_dir: path/to/dataset
  intrinsics: path/to/dataset/intrinsics.json

training:
  device: cuda
  lr: 5e-4
  num_epochs: 100
  scheduler: step
  step_size: 20
  gamma: 0.5
  T_max: 100
  checkpoint_interval: 1
  eval_interval: 2
  plot_interval: 1
  checkpoint_dir: ./checkpoints

model_params:
  body_depth: 4
  color_head_depth: 2
  width: 64
  pos_freqs: 10
  dir_freqs: 4
  skip_layer: 3

render_params:
  num_samples: 32
  near: 2.0
  far: 6.0
  chunk_size: 256
  background_color: [0, 0, 0]
  normalize_weights: false

output:
  checkpoint_dir: ./checkpoints
  log_dir: ./logs

eval:
  root_dir: path/to/dataset
  output_dir: ./eval_renders
  num_samples: 64
  near: 2.0
  far: 6.0
```

---

## 🏋️‍♂️ Training

1. Prepare the dataset as described above  
2. Adjust `configs/train.yaml` to match your setup  
3. Run the training script:

```bash
python train/train_loop.py --config configs/train.yaml
```

### Optional Flags:
- `--resume path/to/checkpoint.pth` to continue training
- `--device cpu` to force CPU mode

During training:
- Rays are volume-rendered through the NeRF model with MSE loss  
- Checkpoints and loss plots are saved at configured intervals  
- Optional novel-view renders are produced via `train/eval.py`

---

## 🎥 Evaluation & Novel View Rendering

Use the script:

```bash
python train/eval.py --config configs/train.yaml
```

Or, in code:

```python
from models.nerf import TinyNeRF
from utils.model_utils import load_model_for_inference

model = load_model_for_inference(
    checkpoint_path='checkpoints/checkpoint_epoch_0002.pth',
    model_class=TinyNeRF,
    model_kwargs={
        'body_depth': 4, 'color_head_depth': 2,
        'width': 64, 'pos_freqs': 10, 'dir_freqs': 4, 'skip_layer': 3
    },
    device='cuda'
)
```

---

## 🛠️ Utilities

- `utils/io.py`  
  - `load_dataset()` – Loads intrinsics/poses and image tensors  
  - `save_checkpoint()`, `load_checkpoint()` – Save/load model states  
  - `save_predicted_image()` – Export predicted images

- `utils/model_utils.py`  
  - `load_model_for_inference()` – Load model + weights in one step

- `utils/visualization.py`  
  - `plot_loss_curve()` – Plot/save training loss  
  - `show_render_comparison()` – Compare predicted vs. ground truth

---

## 📝 Notes

- `main.py` is just a stub and **not part of the core pipeline**  
- Code is minimal and may require polishing (e.g. missing imports)  
- Use GPUs with enough VRAM for training; adjust `chunk_size` if needed  
- This repo is ideal for **learning, extension, and experimentation**  

---

Feel free to contribute, suggest improvements, or extend this further into full NeRF variants like mip-NeRF, NeRF++ or Gaussian Splatting.
