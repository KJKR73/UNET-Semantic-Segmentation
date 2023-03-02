import cv2
import torch
import numpy as np
import albumentations

class PascalDataset(object):
    """Creates the Pascal Dataset

    Args:
        object (object): Generic Python Object
    """
    def __init__(self, split_list, do_transforms, transforms, img_size):
        # Initialize the instance variables
        self.img_size = img_size
        self.split_list = split_list
        self.transforms = transforms
        self.do_transforms = do_transforms
        
    def __getitem__(self, index):
        # Get the image and the mask
        file_name = self.split_list[index].split(".")[0]
        image = cv2.imread(f"{file_name}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(f"{file_name}.png", cv2.IMREAD_GRAYSCALE)
        
        # Augment the mask and the images
        if self.do_transforms:
            augmented_data = self.transforms(image = image, mask = mask)
            image = augmented_data["image"]
            mask = augmented_data["mask"]
        
        return (torch.tensor(image),
                torch.tensor(mask))
        
    def __len__(self):
        # Return the length of the data
        return len(self.split_list)