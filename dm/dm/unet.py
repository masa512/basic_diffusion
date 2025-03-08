
import torch.functional as F 
import torch.nn as nn
######## Pure Unet Implementation ###############

class cbr_block(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,include_relu = True ,include_bn = True):

        super(cbr_block,self).__init__()

        # Initialize CBR Block with just conv
        self.cbr = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride = 1,
                padding = (kernel_size-1)//2
            )
        )

        # Add relu if needed
        if include_relu:
            self.cbr.append(
                nn.ReLU()
            )

        # Add bn if needed
        if include_bn:
            self.cbr.append(
                nn.BatchNorm2d(num_features=out_channels)
            )


