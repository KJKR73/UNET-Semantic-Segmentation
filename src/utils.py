import torch
import cv2
from collections import Counter
import pandas as pd
from tqdm import tqdm
import albumentations as alb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def get_augments(config):
    """Fetches the training and validation augmentations
    """
    train_augments = alb.Compose([
        alb.Resize(config.IMG_SIZE, config.IMG_SIZE),
        alb.HorizontalFlip(p=0.5),
        alb.Normalize(),
        ToTensorV2(),
    ])
    
    validation_augments = alb.Compose([
        alb.Resize(config.IMG_SIZE, config.IMG_SIZE),
        alb.Normalize(),
        ToTensorV2(),
    ])
    
    return train_augments, validation_augments


def save_prediction_mask(truth_mask, pred_mask):
    """Save the side by side plot of the image and the predicted mask to the disk

    Args:
        truth_mask (torch.tensor): ground truth mask
        pred_mask (torch.tensor): predicted mask
    """
    # Create the subplot data
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    # Convert the data
    truth_mask = truth_mask.detach().cpu().numpy()
    pred_mask = torch.argmax(pred_mask.cpu(), dim=0).numpy()
    
    # Plot the data
    axes[0].imshow(truth_mask)
    axes[0].set_title("Groud truth mask")
    axes[1].imshow(pred_mask)
    axes[1].set_title("Predicted mask")
    
    # Save the figure
    fig.savefig("output.png")
    
    
def get_new_pixel_map(files, config):
    """Get the new pixel map

    Args:
        files (list): File list
    """
    # Placeholder
    data = []
    
    # Loop and collect
    for f in tqdm(files[:2000]):
        mask = cv2.imread(config.PATH_TO_MASKS + f  + ".png", cv2.IMREAD_GRAYSCALE)
        data.extend(list(pd.Series(mask.reshape(-1).tolist()).value_counts().index))
    
    # Make the counter
    list_data = sorted(Counter(data).keys())
    map_data = {i : list_data[i] for i in range(len(list_data))}
    
    # Return the map
    return map_data
    
    
    