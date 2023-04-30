import cv2
import torch

class SegmentationDataset(object):
    """Creates the Segmentation Dataset

    Args:
        object (object): Generic Python Object
    """
    def __init__(self, split_list, do_transforms,
                 transforms, config, map_transform):
        # Initialize the instance variables
        self.config = config
        self.split_list = split_list
        self.transforms = transforms
        self.map_transform = map_transform
        self.do_transforms = do_transforms
        
    def __getitem__(self, index):
        # Get the image and the mask
        file_name = self.split_list[index].split(".")[0]
        image = cv2.imread(f"{self.config.PATH_TO_IMAGES}{file_name}.{self.config.IMG_EXT}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(f"{self.config.PATH_TO_MASKS}{file_name}.{self.config.MASK_EXT}",
                          cv2.IMREAD_GRAYSCALE)
        
        # Transform the map
        if self.map_transform != None:
            for key, value in self.map_transform.items():
                mask[mask==value] = key
        
        # Augment the mask and the images
        if self.do_transforms:
            augmented_data = self.transforms(image = image, mask = mask)
            image = augmented_data["image"]
            mask = augmented_data["mask"]
        
        return image.float(), mask.long()
        
    def __len__(self):
        # Return the length of the data
        return len(self.split_list)