import torch
from src.dataset import *

def get_loaders(train_list, validation_list, config, transforms):
    """Fetches the dataloaders for the data

    Args:
        train_list (list): Image name for training
        validation_list (list): Image names for validation
        config (dict): Global config for pipeline
        transforms (albumentation.Compose): Train and augments
    """
    # Define the dataset objects
    train_dataset = SegmentationDataset(split_list=train_list,
                                        do_transforms=True,
                                        transforms=transforms)
    validation_dataset = SegmentationDataset(split_list=validation_list,
                                             do_transforms=True,
                                             transforms=transforms)
    
    # Make the dataloaders
    train_loader = torch.util.data.DataLoader(train_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              num_workers=config.NUM_WORKERS,
                                              shuffle=True)
    validation_loader = torch.util.data.DataLoader(validation_dataset,
                                                   batch_size=config.BATCH_SIZE,
                                                   num_workers=config.NUM_WORKERS,
                                                   shuffle=False)
    
    return train_loader, validation_loader
    