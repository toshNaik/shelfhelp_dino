# helper functions to load dino model and extract features
import torch
from typing import Iterable
import numpy as np
from torchvision import transforms
import sys
import warnings
from PIL import Image

### DINO SPECIFIC FUNCTIONS ###

class Dino:
    def __init__(self, device = 'cuda', model_path: str = 'dino_checkpoint.pth'):
        '''
        Loads dino model and it's weights stored in model_path
        '''
        if device not in ['cuda', 'cpu']:
            raise ValueError('Device must be either cuda or cpu')

        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device)
        self.load_pretrained_weights(model_path, 'teacher')
        # needs to be popped since when calling torch.hub.load,
        # if other model dir has a .py file with the same name it causes conflicts.
        # Here caused by utils.py present in both dino and yolo directories
        # https://github.com/pytorch/hub/issues/243
        sys.modules.pop('utils')

    def load_pretrained_weights(self, pretrained_weights: str, checkpoint_key: str):
        '''
        Loads pretrained weights; specifically for the dino model
        '''
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
        msg = self.model.load_state_dict(state_dict, strict=False)

    def extract_features(self, images) -> np.ndarray:
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
            
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
        # if input is a list of image paths then load images
        if isinstance(images[0], str):
            temp_list = []
            for x in images:
                try:
                    temp_list.append(Image.open(x).convert('RGB'))
                except:
                    pass
            images = temp_list
        # convert images to tensors
        images = torch.stack([transform(x) for x in images]).cuda()
        # extract features
        with torch.no_grad():
            features = self.model(images).squeeze()
        features = features.cpu().numpy()
        return features

### YOLO SPECIFIC FUNCTIONS ###

def yolo_load():
    '''
    Loads yolo model and it's weights stored in best_100.pt
    '''
    yolo = torch.hub.load("ultralytics/yolov5", "custom", "best_100.pt")
    # needs to be popped since when calling torch.hub.load,
    # if other model dir has a .py file with the same name it causes conflicts.
    # Here caused by utils.py present in both dino and yolo directories
    # https://github.com/pytorch/hub/issues/243
    sys.modules.pop('utils')
    return yolo

def yolo_forward(yolo, image, save_crops=False, crop_dir='cropped/'):
    '''
    Runs yolo forward pass on image and returns patches
    '''
    pred = yolo(image.copy())
    table = pred.pandas().xyxy[0]
    xmin = table['xmin'].tolist()
    xmax = table['xmax'].tolist()
    ymin = table['ymin'].tolist()
    ymax = table['ymax'].tolist()
    scores = table['confidence'].tolist()
    crops = []
    # get patches
    for idx, patch in enumerate(zip(xmin, xmax, ymin, ymax, scores)):
        if patch[4] > 0.65:
            x1, x2, y1, y2 = int(patch[0]), int(patch[1]), int(patch[2]), int(patch[3])
            crop = image[y1:y2, x1:x2]
            pil_img = Image.fromarray(crop)
            if save_crops:
                pil_img.save(crop_dir+img.split('.')[0]+"_"+str(idx)+'.jpeg')
            crops.append(pil_img)
    return crops


if __name__ == '__main__':
    ### Load model ###
    vits8 = dino_load()
    # extract features
    features = extract_features(vits8, ['orig.jpeg', 'sierramistsoda0.png'])
    print(features.shape)
