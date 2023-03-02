import torch

def pixel_accuracy(truth, preds):
    """Finds the pixel level accuracy

    Args:
        truth (torch.tensor): Ground truth mask
        pred (torch.tensor): Predicted mask
    """
    # Find the accuracy
    total_corrects = (truth == preds).sum().item()
    total_pixels = torch.numel(truth)
    
    # Return the accruacy
    return round(total_corrects / total_pixels, ndigits=3)

