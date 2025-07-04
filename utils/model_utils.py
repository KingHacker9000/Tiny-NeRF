import torch
import os

def load_model_for_inference(
    checkpoint_path: str,
    model_class: type,         # e.g. TinyNeRF
    model_kwargs: dict,        # same dict you pass into model constructor
    device: str = "cpu"
) -> torch.nn.Module:
    """
    1. Instantiates your model: model = model_class(**model_kwargs)
    2. Loads the checkpoint (torch.load with map_location=device)
    3. Calls model.load_state_dict(checkpoint['model_state'])
    4. Sets model.eval() and moves it to the device
    5. Returns the ready-to-run model
    """
    # Step 1: Instantiate the model
    model = model_class(**model_kwargs)
    
    # Step 2: Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Step 3: Load the model state dict
    model.load_state_dict(checkpoint['model_state'])
    
    # Step 4: Set to eval mode and move to device
    model.eval()
    model.to(device)
    
    return model