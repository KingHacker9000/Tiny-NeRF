import json, os
import torch
from PIL import Image

def load_json(file_path) -> dict:
    """Load a JSON file and return its content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = dict(json.load(file))
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path}: {e}")

def load_dataset(file_path: str = 'dataset/') -> dict:
    """Load data from a file and return it as a dictionary."""
    try:
        intrinsics = load_json(f'{file_path}/intrinsics.json')
        print("Intrinsics loaded successfully:", intrinsics)
        poses = load_json(f'{file_path}/poses.json')
        #print("Poses loaded successfully:", poses)
        image_indexes = [x for x in poses.keys() if x.isdigit()]
        print("Image indexes:", image_indexes)
        # Convert poses to a dictionary of tensors
        poses = {k: torch.tensor(v, dtype=torch.float32) for k, v in poses.items() if k.isdigit()}
        print("Poses converted to tensors successfully.")

        return {
            'intrinsics': intrinsics,
            'poses': poses,
            'image_indexes': image_indexes
        }

    except Exception as e:
        print(f"An error occurred: {e}")

def save_predicted_image(tensor: torch.Tensor, path: str):
    tensor = tensor.clamp(0,1)
    img = (tensor * 255).byte().cpu().permute(1,2,0).numpy()
    Image.fromarray(img).save(path)


def save_checkpoint(
    model,               # torch.nn.Module
    optimizer,           # torch.optim.Optimizer
    epoch: int,          # current epoch number
    checkpoint_dir: str, # where to save files
    scheduler=None       # optional LR scheduler
) -> str:
    """
    - Ensure checkpoint_dir exists.
    - Build a dict with keys: 'epoch', 'model_state', 'optimizer_state', (and 'scheduler_state' if provided).
    - torch.save that dict to a file named e.g. checkpoint_epoch_{epoch:04d}.pth.
    - Return the full path to the saved file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
    
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    
    if scheduler is not None:
        ckpt['scheduler_state'] = scheduler.state_dict()
    
    torch.save(ckpt, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str,           # path to a .pth file
    model,               # torch.nn.Module
    optimizer=None,      # optional torch.optim.Optimizer
    scheduler=None,      # optional LR scheduler
    map_location=None    # e.g. 'cpu' or device
) -> dict:
    """
    - torch.load the file (with map_location if given).
    - model.load_state_dict( ckpt['model_state'] )
    - if optimizer and 'optimizer_state' in ckpt: optimizer.load_state_dict(...)
    - if scheduler and 'scheduler_state' in ckpt: scheduler.load_state_dict(...)
    - return the loaded ckpt dict (so you can read ckpt['epoch']).
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file {path} does not exist.")
    
    ckpt = torch.load(path, map_location=map_location)
    
    model.load_state_dict(ckpt['model_state'])
    
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    
    if scheduler is not None and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    
    return ckpt


if __name__ == "__main__":
    # Example usage
    dataset = load_dataset('dataset/')
    if dataset:
        print(f"Dataset loaded with {len(dataset['image_indexes'])} images.")
        print("Intrinsics:", dataset['intrinsics'])
        print("First pose tensor:", dataset['poses'][dataset['image_indexes'][0]])
        print("Image indexes:", dataset['image_indexes'], type(dataset['image_indexes'][0]))
    else:
        print("Failed to load dataset.")

