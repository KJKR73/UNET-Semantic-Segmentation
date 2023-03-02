import time 
import torch
import torch.nn as nn

# Define the double convolution network
class DoubleConvolution(nn.Module):
    """Creates the double convolution layer

    Args:
        nn (Module): super class for any Neural Network
    """
    def __init__(self, in_channels, out_channels):
        # Init the super class
        super(DoubleConvolution, self).__init__()
        
        # Define the double conv layer
        self.double_conv_module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        """Forward Function for the double convolution block

        Args:
            x (torch.tensor): Tensor of input image (BATCH_SIZE, CHANNELS, IMG_HEIGHT, IMG_WIDTH)

        Returns:
            torch.tensor: Tensor output of double convolution block (BATCH_SIZE, OUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH)
        """
        return self.double_conv_module(x)
    
    
    
class UNET(nn.Module):
    """Creates the UNET model

    Args:
        nn (Module): super class for any Neural Network
    """
    def __init__(self, in_channels, out_channels, channels=[64, 128, 256, 512, 1024]):
        # Init the super class 
        super(UNET, self).__init__()
        
        # Define and populate the up and the down layer
        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        # Loop and populate the down layers
        for idx in range(len(channels)):
            self.down_layers.append(DoubleConvolution(in_channels=in_channels,
                                                      out_channels=channels[idx]))
            self.down_layers.append(nn.MaxPool2d(kernel_size=2,
                                                 stride=2))
            in_channels = channels[idx]
        # (BATCH_SIZE, 512, H, W)
        
        self.center_layer = DoubleConvolution(in_channels=channels[-1],
                                              out_channels=channels[-1] * 2)
        # (BATCH_SIZE, 1024, H, W)
            
        # Add the bottleneck layer
        for idx in range(len(channels))[::-1]:
            self.up_layers.append(nn.ConvTranspose2d(in_channels=channels[idx] * 2,
                                                     out_channels=channels[idx],
                                                     kernel_size=2, stride=2))
            self.up_layers.append(DoubleConvolution(in_channels=channels[idx] * 2,
                                                    out_channels=channels[idx]))
        
        # Define the last layer to convert data to 3d image
        self.last_layer = nn.Conv2d(in_channels=channels[0], out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # skip layers
        skips = []
        
        # Loop and collect skips and data
        for lyr in self.down_layers:
            x = lyr(x)
            if isinstance(lyr, DoubleConvolution):
                skips.append(x.detach())

        # Add the center layer
        x = self.center_layer(x)
        
        # Loop and add skips to the up sample layers
        idx = len(skips) - 1
        for lyr in self.up_layers:
            if isinstance(lyr, DoubleConvolution):
                # Compute the difference
                skip_x = skips[idx]
                diffX = skip_x.size()[2] - x.size()[2]
                diffY = skip_x.size()[3] - x.size()[3]

                # Pad the image
                x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                                diffY // 2, diffY - diffY // 2])
                x = torch.cat((skips[idx], x), dim=1)
                idx -= 1
                
            # Pass through the layer
            x = lyr(x)
            
        return self.last_layer(x)
    
    
    
if __name__ == "__main__":
    # Make the sample data
    sample_data = torch.randn((32, 3, 128, 128)).to("cuda")
    
    # Make the model
    model = UNET(in_channels=sample_data.shape[1], out_channels=sample_data.shape[1])
    model.to("cuda")
    print(f"Total parameters : {sum(p.numel() for p in model.parameters())}")
    start = time.time()
    print(model(sample_data).shape)
    print(f"Total time : {time.time() - start}")
    
    
    
    