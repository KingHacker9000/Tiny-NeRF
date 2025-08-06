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