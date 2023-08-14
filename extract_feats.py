import torch
from typing import Iterable
import numpy as np
from torchvision import transforms
import warnings
from PIL import Image

def load_pretrained_weights(model: torch.nn.Module, pretrained_weights: str, checkpoint_key: str):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    ### Added by ashutosh ###
    # remove head weights if present
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
    ### This is needed for full model checkpoint which includes classification head ###
    
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)


def extract_features(model: torch.nn.Module, images) -> np.ndarray:
    '''
    Extract features from images using a pretrained model.
    Args:
        model: a pretrained model
        images: a list of images or a list of image paths
    Returns:
        features: numpy array of shape (len(images), feature_dim)
    '''
    if len(images) == 0:
        warnings.warn('Length of images is 0', UserWarning)
        return np.array([])
        
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
    # if input is a list of image paths then load images
    if isinstance(images[0], str):
        images = [Image.open(x) for x in images]
    # convert images to tensors
    images = torch.stack([transform(x) for x in images])
    # extract features
    with torch.no_grad():
        features = model(images).squeeze()
    features = features.cpu().numpy()
    return features

if __name__ == '__main__':
    ### Load model ###
    vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    # load checkpoint
    load_pretrained_weights(vits8, 'dino_checkpoint.pth', 'teacher')
    # extract features
    features = extract_features(vits8, ['life.jpeg', 'oreos.jpeg'])
    print(features.shape)
