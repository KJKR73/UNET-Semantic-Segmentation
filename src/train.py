import torch
from tqdm import tqdm
from src.metrics import *
from src.utils import AverageMeter
from collections import defaultdict

def train_one_epoch(model, config, epoch, loader, loss_fxn, optimizer, scaler):
    """Trains one epoch for the semantic segmentation model

    Args:
        model (nn.Module): UNET model
        loss_fxn (_type_): Loss function
        optimizer (_type_): Optimizer type
    """
    # Put the model in the training mode
    model.train()
    
    # Placeholders
    metrics = defaultdict(list)
    
    # Initialize the Averge meter
    meter = AverageMeter()
    
    # Create the bar and loop
    bar = tqdm(enumerate(loader, 1), total=len(loader))
    for batch_no, (images, masks) in bar:
        # Load the data to the device
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)
        
        # Zero the grads
        optimizer.zero_grad()
        
        # Pass the data through the model
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = loss_fxn(images, masks)
            
        # update the average meter
        meter.update(loss.item(), images.shape[0])
            
        # scale and backward the loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update the bar
        bar.set_description(f"Epoch/Batch : {epoch} : {batch_no} | Loss : {round(loss.item(), ndigits=4)}")
        
        # Add the metrics (only accuracy for now)
        metrics["pixel_accuracy"].append(pixel_accuracy(truth=masks, preds=output))
    
    # Return the average metrics
    return meter.avg, metrics
    