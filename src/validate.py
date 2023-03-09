import torch
from tqdm import tqdm
from src.metrics import *
from src.utils import AverageMeter
from collections import defaultdict
from src.utils import save_prediction_mask

@torch.no_grad()
def validate_one_epoch(model, config, epoch, loader, loss_fxn):
    """Trains one epoch for the semantic segmentation model

    Args:
        model (nn.Module): UNET model
        verbose (int): Print frequency
        loss_fxn (_type_): Loss function
        scheduler (_type_): Learning rate scheduler
        optimizer (_type_): Optimizer type
    """
    # Put the model in the eval mode
    model.eval()
    
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
        
        # Get the outputs
        output = model(images)
        loss = loss_fxn(output, masks)
            
        # update the average meter
        meter.update(loss.item(), images.shape[0])
            
        # Update the bar
        bar.set_description(f"Epoch/Batch : {epoch} : {batch_no} | Loss : {round(meter.avg, ndigits=4)}")

        # Add the metrics (only accuracy for now)
        metrics["pixel_accuracy"].append(pixel_accuracy(truth=masks.detach().cpu(),
                                                        preds=torch.argmax(output.cpu(), dim=1)))
        
        if batch_no % 10 == 0:
            save_prediction_mask(truth_mask=masks[0], pred_mask=output[0])
    
    # Return the average metrics
    return meter.avg, metrics
    