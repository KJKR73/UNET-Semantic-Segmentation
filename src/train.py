import torch
from tqdm import tqdm
from src.metrics import *
from src.utils import AverageMeter
from collections import defaultdict
from src.utils import save_prediction_mask

def train_one_epoch(model, config, epoch, loader, loss_fxn, optimizer, scaler, scheduler):
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
        with torch.cuda.amp.autocast(enabled=config.AMP):
            output = model(images)
            loss = loss_fxn(output, masks)
            
        # update the average meter
        meter.update(loss.item(), images.shape[0])
            
        # scale and backward the loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        # Update the bar
        bar.set_description(f"Epoch/Batch : {epoch} : {batch_no} | Loss : {round(meter.avg, ndigits=4)}")
        
        # Add the metrics (only accuracy for now)
        pred_temp = torch.argmax(output.detach().cpu(), dim=1)
        mask_temp = masks.detach().cpu()
        metrics["pixel_accuracy"].append(pixel_accuracy(truth=mask_temp,
                                                        preds=pred_temp))
        
        if batch_no % 1 == 0:
            save_prediction_mask(truth_mask=mask_temp[0], pred_mask=pred_temp[0])


    # Return the average metrics
    return meter.avg, metrics
    