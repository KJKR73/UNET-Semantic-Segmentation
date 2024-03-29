import os
import json
import glob
import torch
import random
import numpy as np
from src.dataloader import *
from src.dataset import *
from src.train import *
from src.utils import *
from src.validate import *
from src.unet import *
from src.res_unet import *

class CONFIG:
    TRAIN_SIZE_PERCENT = 0.9
    DATA_PATH = f"{os.getcwd()}/data/"
    PATH_TO_IMAGES = f"{os.getcwd()}/data/images/"
    PATH_TO_MASKS = f"{os.getcwd()}/data/masks/"
    DEVICE = "cuda"
    IMG_SIZE = 256
    NUM_EPOCHS = 10
    BATCH_SIZE = 4
    AMP = True
    SEED = 10
    MODEL_VARIANT = "res-unet"
    IN_CHANNELS = 3
    NUM_WORKERS = 2
    WEIGHT_DECAY = 1e-8
    MAP_NEEDED = True
    OUT_CHANNELS = 151
    LEARNING_RATE = 1e-3
    UNET_CHANNELS = [64, 128, 256, 512, 1024]
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def engine(config):
    """Run the unet implementation

    Args:
        config (object): Contains the config for the whole file
    """
    # Seed everything
    seed_everything(config.SEED)   
     
    # Get the names of the files
    file_names = [str(i) for i in glob.glob(f"{config.PATH_TO_IMAGES}**/*", recursive=True)]
    file_names = [i.split("/")[-1].split(".")[0] for i in file_names]
    random.shuffle(file_names)
    print(f"Total Images to train : {len(file_names)}")
    
    # Collect the extensions
    config.IMG_EXT = [str(i) for i in glob.glob(f"{config.PATH_TO_IMAGES}**/*", recursive=True)][0].split(".")[-1] 
    config.MASK_EXT = [str(i) for i in glob.glob(f"{config.PATH_TO_MASKS}**/*", recursive=True)][0].split(".")[-1]
    
    train_files = file_names[ : int(config.TRAIN_SIZE_PERCENT * len(file_names))]
    validation_files = file_names[len(train_files):]
    print(f"Training on {len(train_files)} image/mask pairs")
    print(f"Validating on {len(validation_files)} image/mask pairs")
    
    # Calcuale the map
    map_transform = None
    if config.MAP_NEEDED:
        # Try to find the map in the directory
        f_path = config.DATA_PATH + "map.json"
        if os.path.isfile(f_path):
            with open(f_path, 'r') as f:
                map_transform = json.load(f)
        else:
            map_transform = get_new_pixel_map(files = train_files + validation_files, config=config)
            with open(f_path, 'w') as json_file:
                json.dump(map_transform, json_file)
                
        # Update the channels
        config.OUT_CHANNELS = len(map_transform)
        
    print(f"Total classes found : {config.OUT_CHANNELS}")
        
    # Define the model and its requirment
    model_dict = {
        "unet": UNET,
        "res-unet": RES_UNET,
    }
    
    unet_model = model_dict[config.MODEL_VARIANT](in_channels=config.IN_CHANNELS,
                                                  out_channels=config.OUT_CHANNELS,
                                                  channels=config.UNET_CHANNELS)
    unet_model.to(config.DEVICE)
    print(f"Loaded {config.MODEL_VARIANT} with {sum(p.numel() for p in unet_model.parameters())} parameters")
    
    # Get the dataloaders
    print("Fetching dataloaders....")
    t_t, v_t = get_augments(config=config)
    train_loader, validation_loader = get_loaders(train_list=train_files,
                                                  validation_list=validation_files,
                                                  t_t=t_t,
                                                  v_t=v_t,
                                                  m_t=map_transform,
                                                  config=config)
    print("Dataloaders fetched....")
    
    # Define the loss function and other stuff
    scaler = torch.cuda.amp.GradScaler()
    loss_fxn = torch.nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = torch.optim.AdamW(params=unet_model.parameters(),
                                  lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=1e-3, 
                                                    epochs=config.NUM_EPOCHS,
                                                    steps_per_epoch=len(train_loader))
    
    # Train the model
    for epoch_no in range(config.NUM_EPOCHS):
        # Train one epoch of the model
        t_loss, t_metrics = train_one_epoch(model=unet_model, config=config,
                                            epoch=epoch_no + 1, loader=train_loader,
                                            loss_fxn=loss_fxn, optimizer=optimizer,
                                            scaler=scaler, scheduler=scheduler)
        accuracy_t = t_metrics["pixel_accuracy"]
        
        # Validate one epoch of the model
        v_loss, v_metrics = validate_one_epoch(model=unet_model, config=config,
                                               epoch=epoch_no + 1,
                                               loader=validation_loader,
                                               loss_fxn=loss_fxn)
        accuracy_v = v_metrics["pixel_accuracy"]
        print(f"TRAINING || Epoch : {epoch_no + 1} | Loss : {t_loss} | Accuracy : {sum(accuracy_t) / len(accuracy_t)}")
        print(f"VALIDATION || Epoch : {epoch_no + 1} | Loss : {v_loss} | Accuracy : {sum(accuracy_v) / len(accuracy_v)}")
        print("\n")
        
    
    
if __name__ == "__main__":
    engine(CONFIG())
    